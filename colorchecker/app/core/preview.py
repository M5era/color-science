"""Display-only conversion of raw float buffers to 8-bit preview arrays.

Strictly one-way: the sampler and exports never read anything produced
here. Values are shown linearly — a log/flat image looks dim on screen
by design, because the app does not manage color.
"""

import numpy as np


def to_display_u8(pixels: np.ndarray) -> np.ndarray:
    """Map raw floats to uint8 for the screen: clamp [0, 1], scale to 255.

    Returns a new array; the input buffer is never modified.
    """
    clamped = np.clip(pixels, 0.0, 1.0)
    return (clamped * 255.0 + 0.5).astype(np.uint8)
