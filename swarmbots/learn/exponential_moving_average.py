from typing import Optional

class ExponentialMovingAverage:

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema: Optional[float] = None

    def update(self, value: float, weight: Optional[float] = None) -> float:
        if self.ema is None:
            self.ema = value
            return self.ema

        if weight is None:
            effective_alpha = self.alpha
        else:
            effective_alpha = 1 - (1 - self.alpha) ** weight

        self.ema = effective_alpha * value + (1 - effective_alpha) * self.ema
        return self.ema

    def get(self) -> Optional[float]:
        return self.ema


class HybridEMA:
    def __init__(self, alpha: float, n_pre_exponential_samples: int = 10):
        self.alpha = alpha
        self.ema: Optional[float] = None
        self.n_pre_exponential_samples = n_pre_exponential_samples
        self.pre_exponential_samples: list[float] = []


    def update(self, value: float, weight: Optional[float] = None) -> float:
        if len(self.pre_exponential_samples) < self.n_pre_exponential_samples:
            if weight is None:
                weight = 1.0
            self.pre_exponential_samples.append(value * weight)

            self.ema = sum(self.pre_exponential_samples)
            return self.ema

        if weight is None:
            effective_alpha = self.alpha
        else:
            effective_alpha = 1 - (1 - self.alpha) ** weight

        self.ema = effective_alpha * value + (1 - effective_alpha) * self.ema
        return self.ema

    def get(self) -> Optional[float]:
        return self.ema
