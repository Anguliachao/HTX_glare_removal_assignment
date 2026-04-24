"""Simple endpoint checks for the deglare FastAPI service."""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import requests
from PIL import Image

API_BASE_URL = os.getenv("DEGLARE_API_URL", "http://127.0.0.1:4000")
TIMEOUT_SECONDS = float(os.getenv("DEGLARE_API_TIMEOUT", "30"))


def run_ping_test() -> bool:
    """Validate GET /ping response."""
    try:
        response = requests.get(f"{API_BASE_URL}/ping", timeout=TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"[FAIL] /ping status={response.status_code}")
            return False
        if response.json() != {"message": "pong"}:
            print(f"[FAIL] /ping payload={response.text}")
            return False
    except Exception as exc:
        print(f"[FAIL] /ping exception: {exc}")
        return False

    print("[PASS] /ping")
    return True


def run_infer_test() -> bool:
    """Validate POST /infer with an in-memory image."""
    # Create an RGB test image in memory to verify color input handling.
    width, height = 640, 480
    x_grad = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    y_grad = np.tile(
        np.linspace(255, 0, height, dtype=np.uint8).reshape(height, 1), (1, width)
    )
    rgb = np.stack([x_grad, y_grad, np.full_like(x_grad, 128)], axis=2)

    #simulate uploading a PNG file to the API without creating an actual file on disk. 
    # The test image is entirely in memory
    # the files dict is passed to requests.post(..., files=files) to send it as a multipart request.
    image = Image.fromarray(rgb, mode="RGB")
    input_buffer = io.BytesIO() #Creates an in-memory buffer to store binary data without writing to disk
    image.save(input_buffer, format="PNG")
    files = {"image": ("sample.png", input_buffer.getvalue(), "image/png")} #Creates a dictionary mimicking a multipart form upload

    try:
        response = requests.post(
            f"{API_BASE_URL}/infer", files=files, timeout=TIMEOUT_SECONDS
        )
        if response.status_code != 200:
            print(f"[FAIL] /infer status={response.status_code}, body={response.text}")
            return False

        payload = response.json()
        if "image" not in payload or not isinstance(payload["image"], str):
            print("[FAIL] /infer missing base64 image field.")
            return False

        output_bytes = base64.b64decode(payload["image"])
        with Image.open(io.BytesIO(output_bytes)) as output_image:
            if output_image.mode != "L":
                print(f"[FAIL] /infer output mode={output_image.mode}, expected L")
                return False
            if output_image.size != (512, 512):
                print(f"[FAIL] /infer output size={output_image.size}, expected (512, 512)")
                return False
    except Exception as exc:
        print(f"[FAIL] /infer exception: {exc}")
        return False

    print("[PASS] /infer")
    return True


def main() -> int:
    """Run both endpoint checks and return process exit code."""
    print(f"Testing API at: {API_BASE_URL}")
    ping_ok = run_ping_test()
    infer_ok = run_infer_test()

    if ping_ok and infer_ok:
        print("All checks passed.")
        return 0

    print("One or more checks failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
