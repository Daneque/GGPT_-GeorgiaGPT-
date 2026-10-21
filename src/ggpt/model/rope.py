"""Adjacent-pair rotary embeddings with an FP32, nonpersistent trig cache."""

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()

        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even")

        self.head_dim = head_dim
        self.theta = theta

        self.register_buffer(
            "_cos_cached",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_sin_cached",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        device = torch.device(device)

        needs_rebuild = (
            self._cos_cached.device != device
            or self._cos_cached.dtype != torch.float32
            or self._cos_cached.numel() == 0
            or self._cos_cached.shape[-2] < seq_len
            or self._cos_cached.is_inference()
            or self._sin_cached.is_inference()
        )

        if needs_rebuild:
            with (
                torch.inference_mode(False),
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    enabled=False,
                ),
            ):
                exponent = torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=device,
                    dtype=torch.float32,
                )

                frequencies = self.theta ** (
                    -exponent / self.head_dim
                )

                positions = torch.arange(
                    seq_len,
                    device=device,
                    dtype=torch.float32,
                )

                angles = (
                    positions[:, None]
                    * frequencies[None, :]
                )

                self._cos_cached = angles.cos()[
                    None, None, :, :
                ]
                self._sin_cached = angles.sin()[
                    None, None, :, :
                ]

        return (
            self._cos_cached[..., :seq_len, :],
            self._sin_cached[..., :seq_len, :],
        )


def apply_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Rotate [B, H, T, D] in FP32 and restore its original dtype."""

    with torch.autocast(
        device_type=x.device.type,
        enabled=False,
    ):
        even = x.float()[..., 0::2]
        odd = x.float()[..., 1::2]

        rotated = torch.stack(
            (
                even * cos - odd * sin,
                even * sin + odd * cos,
            ),
            dim=-1,
        )

        return rotated.flatten(-2).to(dtype=x.dtype)