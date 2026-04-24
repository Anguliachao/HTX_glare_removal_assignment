"""Configurable lightweight U-Net baseline for grayscale glare removal."""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

DECODER_MODES = ("deconv", "deconv_oneconv", "bilinear_oneconv")


def _resolve_encoder_channels(level: int, channel_ratio: float) -> List[int]:
    """Resolve encoder channels from base config and channel ratio."""
    base_channels = [32, 64, 128, 256][:level]
    # assert channel_ratio  in [0.125, 0.25, 0.5, 1.0], f"channel_ratio: {channel_ratio} ineligible,  must be one of the following option in [0.125, 0.25, 0.5, 1.0]"
    return [max(4, int(round(ch * channel_ratio))) for ch in base_channels]


class ConvStage(nn.Module):
    """Single conv stage (standard or depthwise-separable) with BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, use_depthwise: bool) -> None:
        super().__init__()
        if use_depthwise:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    """Stack of conv stages (default two stages, optionally one for lighter decoder)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_stages: int = 2,
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_in = in_channels
        for _ in range(num_stages):
            layers.append(ConvStage(current_in, out_channels, use_depthwise))
            current_in = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Configurable decoder block with skip connection fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        decoder_mode: str,
        use_depthwise: bool,
    ) -> None:
        super().__init__()

        if decoder_mode in {"deconv", "deconv_oneconv"}:
            # Baseline upsampling path: learnable transposed convolution.
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2, bias=False
            )
        elif decoder_mode == "bilinear_oneconv":
            # Lightweight path: bilinear upsample + 1x1 channel projection.
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(
                f"Unsupported decoder_mode '{decoder_mode}'. "
                f"Expected one of {DECODER_MODES}."
            )

        refine_stages = 2 if decoder_mode == "deconv" else 1
        self.refine = ConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            num_stages=refine_stages,
            use_depthwise=use_depthwise,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)


class DeglarUNet(nn.Module):
    """U-Net autoencoder for grayscale de-glaring with configurable efficiency knobs."""

    def __init__(
        self,
        level: int = 4,
        channel_ratio: float = 1.0,
        decoder_mode: str = "deconv",
        use_depthwise: bool = False,
    ) -> None:
        super().__init__()
        self._validate_config(level, channel_ratio, decoder_mode)

        self.level = level
        self.channel_ratio = channel_ratio
        self.decoder_mode = decoder_mode
        self.use_depthwise = use_depthwise

        self.encoder_channels = _resolve_encoder_channels(level, channel_ratio)
        self.bottleneck_channels = self.encoder_channels[-1] * 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: builds up feature channels while downsampling between blocks.
        self.encoders = nn.ModuleList()
        in_channels = 1
        for out_channels in self.encoder_channels:
            self.encoders.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_stages=2,
                    use_depthwise=use_depthwise,
                )
            )
            in_channels = out_channels

        # Bottleneck: deepest feature processing at lowest spatial resolution.
        self.bottleneck = ConvBlock(
            in_channels=self.encoder_channels[-1],
            out_channels=self.bottleneck_channels,
            num_stages=2,
            use_depthwise=use_depthwise,
        )

        # Decoder: mirror the encoder and fuse matching skip features.
        self.decoders = nn.ModuleList()
        current_channels = self.bottleneck_channels
        for skip_channels in reversed(self.encoder_channels):
            out_channels = skip_channels
            self.decoders.append(
                DecoderBlock(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    decoder_mode=decoder_mode,
                    use_depthwise=use_depthwise,
                )
            )
            current_channels = out_channels

        # Output projection to grayscale reconstruction in [0, 1].
        self.head = nn.Conv2d(current_channels, 1, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    @staticmethod
    def _validate_config(level: int, channel_ratio: float, decoder_mode: str) -> None:
        """Validate public constructor arguments."""
        if not isinstance(level, int) or isinstance(level, bool) or not (2 <= level <= 4):
            raise ValueError(f"level must be an integer in [2, 4], got {level!r}.")

        if not isinstance(channel_ratio, (float, int)) or not (0 < float(channel_ratio) <= 1):
            raise ValueError(
                f"channel_ratio must be a float in (0, 1], got {channel_ratio!r}."
            )

        if decoder_mode not in DECODER_MODES:
            raise ValueError(
                f"decoder_mode must be one of {DECODER_MODES}, got {decoder_mode!r}."
            )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected input shape [B,1,H,W], got {tuple(x.shape)} instead."
            )

        down_factor = 2 ** self.level
        if x.shape[2] % down_factor != 0 or x.shape[3] % down_factor != 0:
            raise ValueError(
                f"Input H/W must be divisible by 2**level ({down_factor}) for level={self.level}. "
                f"Got H={x.shape[2]}, W={x.shape[3]}."
            )

        skips: List[Tensor] = []
        out = x

        for encoder in self.encoders:
            out = encoder(out)
            skips.append(out)
            out = self.pool(out)

        out = self.bottleneck(out)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            out = decoder(out, skip)

        out = self.head(out)
        return self.output_activation(out)


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_gflops(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> float:
    """Estimate GFLOPs for Conv2d/ConvTranspose2d from one forward pass."""
    total_flops = 0
    hooks = []

    def _conv_hook(module: nn.Module, _: Tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total_flops
        if not isinstance(output, Tensor):
            return

        batch_size, out_channels, out_h, out_w = output.shape
        in_channels = module.in_channels
        k_h, k_w = module.kernel_size
        groups = module.groups

        flops = (
            batch_size
            * out_channels
            * out_h
            * out_w
            * (in_channels // groups)
            * k_h
            * k_w
            * 2
        )
        total_flops += flops

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(_conv_hook))

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape)
        _ = model(dummy)

    for hook in hooks:
        hook.remove()
    if model_was_training:
        model.train()

    return total_flops / 1e9


def run_forward_self_test(
    level: int,
    channel_ratio: float,
    decoder_mode: str,
    use_depthwise: bool,
    batch_size: int,
    input_size: int,
) -> None:
    """Run one configurable forward self-test and print efficiency metrics."""
    model = DeglarUNet(
        level=level,
        channel_ratio=channel_ratio,
        decoder_mode=decoder_mode,
        use_depthwise=use_depthwise,
    )
    dummy_input = torch.randn(batch_size, 1, input_size, input_size)

    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(
        "Config -> "
        f"level={level}, channel_ratio={channel_ratio}, decoder_mode={decoder_mode}, "
        f"use_depthwise={use_depthwise}"
    )
    print(f"Resolved encoder channels: {model.encoder_channels}")
    print(f"Resolved bottleneck channels: {model.bottleneck_channels}")
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(dummy_output.shape)}")

    total_params = count_parameters(model)
    gflops = estimate_gflops(model, input_shape=tuple(dummy_input.shape))
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Estimated forward GFLOPs (Conv/Deconv only): {gflops:.3f}")

    temp_path = os.path.join(tempfile.gettempdir(), "deglar_unet_temp_state_dict.pth")
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    print(f"Temporary state_dict path: {temp_path}")
    print(f"state_dict size: {size_mb:.3f} MB")


def run_shape_matrix_check(input_size: int = 512) -> None:
    """Run a matrix of shape checks for levels, decoder modes, and depthwise options."""
    print("Running shape matrix checks...")
    for level in (2, 3, 4):
        for decoder_mode in DECODER_MODES:
            for use_depthwise in (False, True):
                model = DeglarUNet(
                    level=level,
                    channel_ratio=1.0,
                    decoder_mode=decoder_mode,
                    use_depthwise=use_depthwise,
                )
                x = torch.randn(1, 1, input_size, input_size)
                with torch.no_grad():
                    y = model(x)
                expected_shape = (1, 1, input_size, input_size)
                if tuple(y.shape) != expected_shape:
                    raise RuntimeError(
                        f"Shape check failed for level={level}, decoder_mode={decoder_mode}, "
                        f"use_depthwise={use_depthwise}. Got {tuple(y.shape)}."
                    )
                print(
                    f"[OK] level={level}, decoder_mode={decoder_mode}, "
                    f"use_depthwise={use_depthwise}, output={tuple(y.shape)}"
                )
    print("All shape matrix checks passed.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for model self-test and quick architecture sweeps."""
    parser = argparse.ArgumentParser(description="DeglarUNet configurable self-test")
    parser.add_argument("--level", type=int, default=4, help="Number of U-Net levels (2..4)")
    parser.add_argument(
        "--channel_ratio",
        type=float,
        default=1.0,
        help="Channel scaling ratio in (0, 1], e.g. 0.5 or 0.25",
    )
    parser.add_argument(
        "--decoder_mode",
        type=str,
        default="deconv",
        choices=list(DECODER_MODES),
        help="Decoder strategy",
    )
    parser.add_argument(
        "--use_depthwise",
        action="store_true",
        help="Use depthwise-separable conv composition",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Dummy batch size")
    parser.add_argument("--input_size", type=int, default=512, help="Dummy H/W size")
    parser.add_argument(
        "--run_matrix_check",
        action="store_true",
        help="Run shape checks for level/decoder/depthwise combinations",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_forward_self_test(
        level=args.level,
        channel_ratio=args.channel_ratio,
        decoder_mode=args.decoder_mode,
        use_depthwise=args.use_depthwise,
        batch_size=args.batch_size,
        input_size=args.input_size,
    )

    if args.run_matrix_check:
        run_shape_matrix_check(input_size=args.input_size)
