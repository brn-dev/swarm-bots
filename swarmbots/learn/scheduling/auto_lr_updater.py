from typing import Any, Optional

import numpy as np

from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRateUpdateResult, AutomaticLearningRateUpdater
from swarmbots.learn.summary_statistics import SummaryStatistics

def make_auto_lr_updater(
    warmup_iterations: int,
    cold_lr: float,
    warm_lr: float,
) -> AutomaticLearningRateUpdater:
    def auto_lr_updater(
            old_lr: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> AutomaticLearningRateUpdateResult:
        if early_stop_epoch is not None and early_stop_epoch == 0:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = 0.75
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'epoch={early_stop_epoch}',
                'event': 'zero_epoch_hit'
            }

        if early_stop_kl_div is not None and early_stop_kl_div > 0.0125:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = np.clip(0.95 - early_stop_kl_div, 0.4, 0.95)
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'kl={early_stop_kl_div:.3f}',
                'event': 'max_kl_hit'
            }

        clip_frac_stats: Optional[SummaryStatistics] = metrics.get('clip_frac', None)
        if clip_frac_stats and clip_frac_stats.mean > 0.19:
            state['counter'] = 0
            state['warmup'] = False
            clip_frac = clip_frac_stats.mean
            decay_factor = np.clip(1 - clip_frac, 0.5, 0.9)
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'{clip_frac=:.3f}',
                'event': 'max_clip_frac_hit'
            }

        if early_stop_epoch is not None and early_stop_epoch < 4:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = {
                1: 0.8, 2: 0.85, 3: 0.9,
            }[early_stop_epoch]
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'epoch={early_stop_epoch}',
                'event': 'min_epoch_hit'
            }

        warmup: bool = state.get('warmup', warmup_iterations > 0) and n_iterations <= warmup_iterations
        state['warmup'] = warmup
        if warmup:
            new_lr = cold_lr + (warm_lr - cold_lr) * n_iterations / warmup_iterations
            return {
                'new_lr': new_lr,
                'msg': f'Warmup ({n_iterations}/{warmup_iterations})',
                'event': 'warmup'
            }

        counter = state.get('counter', 0) + 1

        if counter >= 2:
            state['counter'] = 0
            return {'new_lr': old_lr * 1.1}

        state['counter'] = counter
        return {'new_lr': None}

    return auto_lr_updater