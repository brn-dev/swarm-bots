from typing import Optional

class ExponentialMovingAverage:

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema: Optional[float] = None

    def update(self, value: float, weight: float = 1.0) -> float:
        if weight <= 0:
            return self.ema if self.ema is not None else value
        if self.ema is None:
            self.ema = value
            return self.ema
        effective_alpha = 1 - (1 - self.alpha) ** weight
        self.ema = effective_alpha * value + (1 - effective_alpha) * self.ema
        return self.ema

    def get(self) -> Optional[float]:
        return self.ema

