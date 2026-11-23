import math
from typing import Any, Optional


class ExponentialMovingAverage:

    def __init__(self, alpha: float):
        if not (0 <= alpha <= 1):
            raise ValueError(f'Alpha must be between 0 and 1, got {alpha}')

        self.alpha = alpha
        self.ema: Optional[float] = None

    def update(self, value: float) -> Optional[float]:
        average = value if self.ema is None else self.alpha * value + (1 - self.alpha) * self.ema
        if not math.isfinite(average):
            raise ValueError("Moving average must be finite")
        self.ema = average
        return self.ema

    def get(self) -> Optional[float]:
        return self.ema


class HybridEMA(ExponentialMovingAverage):
    def __init__(
            self,
            alpha: float,
            n_pre_exponential_samples: int = 100,
            return_pre_exponential_mean: bool = False
    ) -> None:
        super().__init__(alpha=alpha)

        if type(n_pre_exponential_samples) is not int or n_pre_exponential_samples < 0:
            raise ValueError("n_pre_exponential_samples must be a nonnegative integer")
        if type(return_pre_exponential_mean) is not bool:
            raise ValueError("return_pre_exponential_mean must be a boolean")

        self.n_pre_exponential_samples = n_pre_exponential_samples

        self.return_pre_exponential_mean = return_pre_exponential_mean

        self.pre_exponential_count: int = 0
        self.pre_exponential_sum: float = 0.0

    def state_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "n_pre_exponential_samples": self.n_pre_exponential_samples,
            "return_pre_exponential_mean": self.return_pre_exponential_mean,
            "pre_exponential_count": self.pre_exponential_count,
            "pre_exponential_sum": self.pre_exponential_sum,
            "ema": self.ema,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "HybridEMA":
        result = cls(state["alpha"], state["n_pre_exponential_samples"], state["return_pre_exponential_mean"])
        count, total, ema = state["pre_exponential_count"], state["pre_exponential_sum"], state["ema"]
        if type(count) is not int or not 0 <= count <= result.n_pre_exponential_samples:
            raise ValueError("Invalid moving-average sample count")
        if not math.isfinite(total) or (ema is not None and not math.isfinite(ema)):
            raise ValueError("Invalid moving-average checkpoint values")
        if count == 0 and total != 0:
            raise ValueError("Moving-average sum must be zero before the first sample")
        if result.n_pre_exponential_samples > 0:
            has_average = count > 0 and (result.return_pre_exponential_mean or count == result.n_pre_exponential_samples)
            if (ema is not None) != has_average:
                raise ValueError("Moving-average checkpoint value disagrees with its sample count")
        result.pre_exponential_count = count
        result.pre_exponential_sum = total
        result.ema = ema
        return result

    def update(self, value: float) -> Optional[float]:
        if self.pre_exponential_count < self.n_pre_exponential_samples:
            total = self.pre_exponential_sum + value
            if not math.isfinite(total):
                raise ValueError("Moving-average sample sum must be finite")
            self.pre_exponential_sum = total
            self.pre_exponential_count += 1

            if self.return_pre_exponential_mean or self.pre_exponential_count == self.n_pre_exponential_samples:
                self.ema = self.pre_exponential_sum / self.pre_exponential_count
                return self.ema
            return None

        return super().update(value=value)
