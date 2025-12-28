from typing import Optional

class ExponentialMovingAverage:

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema: Optional[float] = None

    def update(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        return self.ema

    def get(self) -> Optional[float]:
        return self.ema

