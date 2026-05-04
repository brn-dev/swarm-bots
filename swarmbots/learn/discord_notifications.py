from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, TypeVar

from loguru import logger


DISCORD_WEBHOOK_ENV_VAR = "SWARMBOTS_DISCORD_WEBHOOK_URL"
DISCORD_CONTENT_LIMIT = 2000
DISCORD_USER_AGENT = "swarm-bots-training-notifier/0.1"
T = TypeVar("T")


def run_with_discord_notification(
        *,
        run_name: str,
        run_dir: str | Path,
        total_timesteps: int,
        algorithm: Any,
        run: Callable[[], T],
) -> T:
    error: BaseException | None = None
    try:
        return run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        notify_training_run_finished(
            run_name=run_name,
            run_dir=run_dir,
            total_timesteps=total_timesteps,
            algorithm=algorithm,
            error=error,
        )


def notify_training_run_finished(
        *,
        run_name: str,
        run_dir: str | Path,
        total_timesteps: int,
        algorithm: Any,
        error: BaseException | None,
) -> bool:
    webhook_url = os.environ.get(DISCORD_WEBHOOK_ENV_VAR)
    if not webhook_url:
        return False

    status = _get_run_status(
        current_timesteps=int(algorithm.n_total_timesteps),
        total_timesteps=total_timesteps,
        error=error,
    )
    content = _format_training_run_finished_message(
        run_name=run_name,
        status=status,
        run_dir=run_dir,
        total_timesteps=total_timesteps,
        algorithm=algorithm,
        error=error,
    )
    return send_discord_message(content=content, webhook_url=webhook_url)


def send_discord_message(*, content: str, webhook_url: str) -> bool:
    payload = json.dumps({"content": content[:DISCORD_CONTENT_LIMIT]}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": DISCORD_USER_AGENT,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace").strip()
        message = f"Failed to send Discord notification: HTTP {exc.code} {exc.reason}"
        if response_body:
            message = f"{message}: {response_body[:500]}"
        logger.error(message)
        return False
    except urllib.error.URLError as exc:
        logger.error(f"Failed to send Discord notification: {exc.reason}")
        return False
    except OSError as exc:
        logger.error(f"Failed to send Discord notification: {type(exc).__name__}: {exc}")
        return False

    return True


def _get_run_status(
        *,
        current_timesteps: int,
        total_timesteps: int,
        error: BaseException | None,
) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if error is not None:
        return "failed"
    if current_timesteps < total_timesteps:
        return "stopped"
    return "finished"


def _format_training_run_finished_message(
        *,
        run_name: str,
        status: str,
        run_dir: str | Path,
        total_timesteps: int,
        algorithm: Any,
        error: BaseException | None,
) -> str:
    lines = [
        f"{run_name} {status}",
        f"run_dir: {Path(run_dir).as_posix()}",
        f"timesteps: {algorithm.n_total_timesteps:,} / {total_timesteps:,}",
        f"iterations: {algorithm.n_total_iterations:,}",
        f"updates: {algorithm.n_total_updates:,}",
    ]
    best_return_ema = getattr(algorithm, "_best_return_ema", None)
    if best_return_ema is not None:
        lines.append(f"best_ep_rew_ema: {best_return_ema:.6g}")

    if error is not None:
        lines.append(f"error: {type(error).__name__}: {error}")

    return "\n".join(lines)
