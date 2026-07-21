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
    # identity-regularization multiplier: >1 means the solver treats
    # deviating from identity as expensive — used for prep stages that
    # should only move when it makes the fit a LOT easier
    reg_scale: float = 1.0

    @abstractmethod
    def identity(self) -> np.ndarray: ...

    @abstractmethod
    def bounds(self) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def apply(self, x: np.ndarray, params: np.ndarray) -> np.ndarray: ...

    def describe(self, params: np.ndarray) -> str:
        return f"{self.name}: {np.round(params, 4).tolist()}"

    def label(self, params: np.ndarray) -> str:
        """Short human name for what this fitted stage is doing
        ("skew dark greens", "cool lows") — overridden per stage."""
        return self.name

    def short_label(self, params: np.ndarray) -> str:
        """<= 9 characters, for Resolve node labels (the node graph
        truncates around there). Derived from label()."""
        return shorten_label(self.label(params))


_PHRASES = [("white balance", "WB"), ("exposure trim", "Exp"),
            ("gamma trim", "Gam"), ("global twist", "GlbTwst"),
            ("add contrast", "Con+"), ("flatten contrast", "Con-")]

_WORDS = {
    "skew": "Skw", "squash": "Sqsh", "spread": "Sprd", "boost": "Bst",
    "desat": "Dst", "brighten": "Brt", "darken": "Drk", "bleach": "Blch",
    "tilt": "Tlt", "toward": ">", "warm": "Wrm", "cool": "Cool",
    "highs": "Hi", "lows": "Lo", "highlights": "Hi", "shadows": "Lo",
    "colors": "Col", "zone": "Zn", "shift": "Shf", "prep": "Prep",
    "dark": "Dk", "bright": "Br", "saturation": "Sat", "sector": "Sec",
    "contrast": "Con", "crosstalk": "XT", "tint": "Tnt",
    "reduce": "Rdc", "brilliance": "Brill",
    "red": "Red", "reds": "Red", "orange": "Org", "oranges": "Org",
    "yellow": "Yel", "yellows": "Yel", "lime": "Lim", "limes": "Lim",
    "green": "Grn", "greens": "Grn", "teal": "Teal", "teals": "Teal",
    "cyan": "Cyn", "cyans": "Cyn", "azure": "Azr", "azures": "Azr",
    "blue": "Blu", "blues": "Blu", "purple": "Ppl", "purples": "Ppl",
    "magenta": "Mag", "magentas": "Mag", "pink": "Pnk", "pinks": "Pnk",
}


def shorten_label(label: str) -> str:
    """Compress a grading-note label to <= 9 chars for node names."""
    text = label
    if "(idle)" in text:
        return "idle"
    for phrase, short in _PHRASES:
        text = text.replace(phrase, short)
    # drop parenthetical qualifiers ("(spare blues)")
    while "(" in text:
        a = text.index("(")
        b = text.find(")", a)
        text = text[:a] + (text[b + 1:] if b >= 0 else "")
    out = ""
    for token in text.replace("+", " + ").split():
        piece = _WORDS.get(token.lower(), token[:4].capitalize()
                           if token not in ("+", ">", "-") else token)
        if len(out) + len(piece) > 9:
            break
        out += piece
    out = out.rstrip(">+-")
    return out or label[:9]
