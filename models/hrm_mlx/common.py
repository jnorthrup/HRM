import math
import numpy as np
from scipy.special import erfinv

import mlx.core as mx


def trunc_normal_init(shape: tuple, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> mx.array:
    """
    Truncated normal initialization, ported from the PyTorch version.
    """
    if std == 0:
        return mx.zeros(shape)

    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)

    # Uniformly fill tensor with values from [a, b]
    unif = np.random.uniform(a, b, size=shape)

    # Apply inverse error function
    tensor_np = sqrt2 * erfinv(unif)

    # The standard deviation of the truncated normal distribution
    z = (b - a) / 2
    c = (2 * math.pi) ** -0.5
    pdf_u = c * math.exp(-0.5 * lower ** 2)
    pdf_l = c * math.exp(-0.5 * upper ** 2)
    comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

    # Scale and clip
    tensor_np *= comp_std
    tensor_np = np.clip(tensor_np, lower * comp_std, upper * comp_std)

    return mx.array(tensor_np, dtype=mx.float32)
