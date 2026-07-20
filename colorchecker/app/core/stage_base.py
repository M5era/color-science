"""The parametric stage contract (base class only, no dependencies —
lives apart so stage modules can import it without cycles).

- ALL state is a flat float vector `params` with box `bounds()`
- `apply(x, params)` is pure and vectorized: no hidden state, no side
  effects; swap numpy for torch and the architecture holds
- `identity()` is the do-nothing parameter vector (also the
  regularization anchor, so overlapping stages don't fight)
"""

from abc import ABC, abstractmethod

import numpy as np


class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def identity(self) -> np.ndarray: ...

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def apply(self, x: np.ndarray, params: np.ndarray) -> np.ndarray: ...

    def describe(self, params: np.ndarray) -> str:
        return f"{self.name}: {np.round(params, 4).tolist()}"
