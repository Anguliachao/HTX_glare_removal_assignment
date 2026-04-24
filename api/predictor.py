"""Model loading and inference utilities for de-glaring."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from notebooks.model import DeglarUNet

TARGET_SIZE = 512

if hasattr(Image, "Resampling"):
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
else:
    RESAMPLE_BILINEAR = Image.BILINEAR


class DeglarePredictor:
    """Loads the trained model and performs preprocessing/inference/postprocessing."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_checkpoint = os.path.join(repo_root, "checkpoints", "best_model.pth")
        self.checkpoint_path = checkpoint_path or os.getenv(
            "DEGLARE_CHECKPOINT_PATH", default_checkpoint
        )
        self.device = torch.device("cpu")
        self.model: DeglarUNet | None = None

    def load_model(self) -> None:
        """Load model weights from checkpoint into CPU inference mode."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at '{self.checkpoint_path}'. "
                "Set DEGLARE_CHECKPOINT_PATH or mount checkpoints directory."
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict, model_config = self._extract_checkpoint_parts(checkpoint)

        model = DeglarUNet(**model_config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def predict_base64_png(self, image_bytes: bytes) -> str:
        """Predict de-glared output image and return PNG bytes as base64 string."""
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        input_tensor = self._preprocess(image_bytes)
        with torch.no_grad():
            output_tensor = self.model(input_tensor.to(self.device))
        return self._postprocess(output_tensor)

    def _extract_checkpoint_parts(
        self, checkpoint: Any
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Handle both training-checkpoint dict and raw state_dict formats."""
        if not isinstance(checkpoint, dict) or not checkpoint:
            raise ValueError("Checkpoint format is invalid.")

        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
            model_config = checkpoint.get("model_config", {})
            if not isinstance(model_state_dict, dict):
                raise ValueError("model_state_dict is invalid in checkpoint.")
            if not isinstance(model_config, dict):
                model_config = {}
            return model_state_dict, model_config

        # Fallback: treat checkpoint itself as a raw state_dict.
        return checkpoint, {}

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Convert uploaded image bytes to normalized model input tensor."""
        if not image_bytes:
            raise ValueError("Input image bytes are empty.")

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert("L")
                image = image.resize((TARGET_SIZE, TARGET_SIZE), RESAMPLE_BILINEAR)
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)

    def _postprocess(self, output_tensor: torch.Tensor) -> str:
        """Convert model output tensor to base64-encoded grayscale PNG."""
        if output_tensor.ndim != 4:
            raise ValueError(f"Unexpected output shape: {tuple(output_tensor.shape)}")

        output_np = (
            output_tensor[0, 0].detach().cpu().clamp(0.0, 1.0).numpy() * 255.0
        ).round().astype(np.uint8)
        output_image = Image.fromarray(output_np, mode="L")

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
