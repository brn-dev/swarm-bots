from collections import defaultdict
from typing import TypeVar, Generic

T = TypeVar('T')

class MetricsLists(Generic[T]):

    def __init__(self):
        self.lists: dict[str, list[T]] = defaultdict(list)

    def add(self, metrics: dict[str, T]):
        for k, v in metrics.items():
            self.lists[k].append(v)

    def get(self):
        return self.lists
