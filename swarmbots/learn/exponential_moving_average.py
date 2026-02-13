from typing import Optional

class ExponentialMovingAverage:

    def __init__(self, alpha: float):
        if not (0 <= alpha <= 1):
            raise ValueError(f'Alpha must be between 0 and 1, got {alpha}')

        self.alpha = alpha
        self.ema: Optional[float] = None

    def update(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
            return self.ema

        self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

    def get(self) -> Optional[float]:
        return self.ema


class HybridEMA(ExponentialMovingAverage):
    def __init__(self, alpha: float, n_pre_exponential_samples: int = 100):
        super().__init__(alpha=alpha)

        if n_pre_exponential_samples < 0:
            raise ValueError("n_pre_exponential_samples must be >= 0")

        self.n_pre_exponential_samples = n_pre_exponential_samples

        self.pre_exponential_count: int = 0
        self.pre_exponential_sum: float = 0.0


    def update(self, value: float) -> float:
        if self.pre_exponential_count < self.n_pre_exponential_samples:
            self.pre_exponential_count += 1
            self.pre_exponential_sum += value

            self.ema = self.pre_exponential_sum / self.pre_exponential_count
            return self.ema

        return super().update(value=value)
