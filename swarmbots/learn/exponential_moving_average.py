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

