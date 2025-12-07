from __future__ import annotations

import torch


def add_gaussian_noise(t: torch.Tensor, sigma: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if sigma <= 0:
        return t
    noise = torch.randn_like(t, generator=generator) * sigma
    return t + noise


def add_laplace_noise(t: torch.Tensor, b: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if b <= 0:
        return t
    # Torch has Laplace in distributions; sample using inverse CDF for generator control
    # U ~ Uniform(-0.5, 0.5); Lap(b) = -b * sgn(U) * ln(1 - 2|U|)
    if generator is None:
        u = torch.rand_like(t) - 0.5
    else:
        u = torch.rand(t.shape, dtype=t.dtype, device=t.device, generator=generator) - 0.5
    sign = torch.sign(u)
    u_abs = torch.abs(u)
    # Clamp to avoid log(0)
    eps = torch.finfo(t.dtype).eps if t.dtype.is_floating_point else 1e-12
    val = -b * sign * torch.log(torch.clamp(1 - 2 * u_abs, min=eps))
    return t + val


def add_logit_noise(logits: torch.Tensor, kind: str, scale: float, generator: torch.Generator | None = None) -> torch.Tensor:
    kind = (kind or "none").lower()
    if kind in ("none", ""):  # no noise
        return logits
    if kind in ("gaussian", "normal"):
        return add_gaussian_noise(logits, sigma=scale, generator=generator)
    if kind in ("laplace", "laplacian"):
        return add_laplace_noise(logits, b=scale, generator=generator)
    raise ValueError(f"Unknown noise kind: {kind}")
