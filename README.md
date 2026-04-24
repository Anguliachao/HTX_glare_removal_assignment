# HTX Glare Removal

## 1) Project Overview
This repository is a take-home assignment submission for grayscale image glare removal.  
The project trains a lightweight PyTorch U-Net style model on SD1 paired data and serves inference through FastAPI on port `4000`.

Major components:
- Notebook-based training and evaluation workflow.
- FastAPI inference microservice.
- Standalone endpoint test client script.
- Dockerized API packaging and execution.

Primary deliverables:
```text
1) notebooks/deglare.ipynb (training/evaluation reproduction)
2) api/app.py + api/predictor.py (serving)
3) Dockerfile (containerized service)
4) tests/test_client.py (endpoint verification)
```

## 2) Repository Structure
```text
htx_glare_removal/
|-- README.md                                # Submission documentation
|-- requirements.txt                         # Local/API runtime Python dependencies
|-- Dockerfile                               # Container build and API startup definition
|-- checkpoints/
|   |-- best_model.pth                       # Trained model checkpoint (.pth)
|-- notebooks/
|   |-- dataset.py                           # SD1 dataset parsing + preprocessing + dataloaders
|   |-- model.py                             # DeglarUNet architecture and config knobs
|   |-- loss.py                              # Weighted L1/L2 loss helpers
|   |-- deglare_colab.ipynb                  # Primary end-to-end Colab training notebook
|-- api/
|   |-- app.py                               # FastAPI routes/lifespan; HTTP concerns only
|   |-- predictor.py                         # Model loading + preprocess/inference/postprocess
|-- tests/
    |-- test_client.py                       # Standalone endpoint test script 
```

## 3) Python Environment and Requirements

### 3.1 Overview
This section covers local setup for running the FastAPI service and `tests/test_client.py` outside Docker.  
This environment is separate from Colab; Colab notebooks install notebook-specific packages inside notebook cells.

### 3.2 Conda Setup + `requirements.txt` Installation
Create and activate a Python 3.10 environment:
```powershell
conda create -n deglare310 python=3.10 -y
conda activate deglare310
```

Install dependencies:
```powershell
pip install -r requirements.txt
```

Runtime note:
```text
The API service is designed to run inference on CPU.
Training is expected to use Google Colab GPU runtime.
```

## 4) Artefact #1 - Jupyter Notebook Training and Evaluation

### 4.1 Notebook File Descriptions
Path:
```text
notebooks/dataset.py
```
`SD1Dataset` reads SD1 PNG files and splits each `1536x512` image horizontally into three equal panels, using panel 0 (GT) and panel 1 (glare input), while ignoring mask panel 2. It converts both selected panels to grayscale and normalizes tensors to `[0,1]` with shape `[1,512,512]`. The loader includes paired geometric augmentation so glare/GT remain spatially aligned; this is implemented as synchronized transforms in the dataset pipeline. `get_dataloaders()` returns train/val `DataLoader` objects with configurable batch sizes and workers.

Path:
```text
notebooks/model.py
```
`DeglarUNet` is a lightweight grayscale encoder-decoder with skip connections and a sigmoid output head. The architecture exposes configuration knobs for size/latency trade-offs: `level` controls encoder/decoder depth, `channel_ratio` scales channel widths, `decoder_mode` selects upsampling strategy (`deconv`, `deconv_oneconv`, `bilinear_oneconv`), and `use_depthwise` toggles depthwise-separable convolution blocks. The module also includes self-test utilities (parameter count, GFLOPs estimate, shape checks) to validate candidate model variants.

Path:
```text
notebooks/loss.py
```
This module defines simple weighted loss wrappers used in notebook training, including `WeightedL1Loss` and `WeightedL2Loss`. A `build_loss()` helper creates the requested loss by name and enforces argument validation. The design keeps losses minimal and explicit for reproducibility while allowing future composite-loss extension.

Path:
```text
notebooks/deglare_colab.ipynb
```
This is the primary end-to-end notebook for Colab reproduction. It covers environment setup, Drive mount, package installation, data loading/visualization, model construction, and training with L1 objective plus PSNR/SSIM tracking. It saves checkpoints and experiment logs to Drive, plots training curves, performs final validation against the target (`val L1 < 0.06`), and demonstrates inference outputs.

### 4.2 How to Reproduce Training on Google Colab
1. **Colab runtime setup**
```text
Open a new Colab notebook.
Set Runtime -> Change runtime type -> T4 GPU before running cells.
```

2. **Upload/prepare notebook files**
```text
Use /content/notebooks/ and upload:
- notebooks/dataset.py
- notebooks/model.py
- notebooks/loss.py
Open and run notebooks/deglare.ipynb in Colab.
```

3. **Prepare SD1 dataset on Google Drive**
```text
/content/drive/MyDrive/SD1/train/
/content/drive/MyDrive/SD1/val/
```
```text
Each file should be a 1536x512 PNG (Ground Truth | Glare | Mask).
```

4. **Install required packages**
```text
Section 0 in the notebook installs missing packages automatically.
```
```text
opencv-python 
scikit-image
scipy
tqdm
matplotlib
numpy
```
```text
Note: torch and torchvision are preinstalled in standard Colab GPU runtimes.
```

5. **Configure paths in notebook setup cells**
```text
TRAIN_DIR
VAL_DIR
BEST_CHECKPOINT_PATH
EXPERIMENTS_ROOT
```

6. **Run training**
```text
Runtime -> Run all
```
```text
Training/validation metrics include L1, PSNR, and SSIM.
Checkpoints and experiment logs are saved per epoch.
```

7. **Retrieve trained model**
```text
/content/drive/MyDrive/deglare_checkpoints/best_model.pth
```
Download and Copy it to:
```text
checkpoints/best_model.pth
```

## 5) Artefact #2 - API Service Source Code

### 5.1 File Descriptions
Path:
```text
api/predictor.py
```
`DeglarePredictor` owns all ML concerns: checkpoint loading, image preprocessing, model forward pass, and output encoding. It imports `DeglarUNet` from `notebooks.model`, always runs inference on CPU, and resizes any input image to `512x512` grayscale before prediction. Checkpoint loading supports both formats: full checkpoint dict (`model_state_dict` + optional `model_config`) and legacy plain `state_dict`. The exposed inference entrypoint is `predict_base64_png(image_bytes)`, returning base64 PNG content for API responses.

Path:
```text
api/app.py
```
This module owns HTTP concerns only and does not embed model math. It defines a FastAPI app with lifespan startup that loads the predictor once and stores it in `app.state`. It implements `GET /ping` and `POST /infer` on port `4000`, including request validation and error mapping. `/infer` accepts multipart image upload and returns JSON with base64-encoded PNG output.

### 5.2 API Endpoints Reference
| Method | Endpoint | Description | Request | Response |
|---|---|---|---|---|
| GET | `/ping` | Health check | none | `{"message":"pong"}` |
| POST | `/infer` | Glare removal inference | multipart form-data field `image` | `{"image":"<base64 PNG>"}` |

Example response:
```json
{"message":"pong"}
```
```json
{"image":"iVBORw0KGgoAAAANSUhEUgAA..."}
```

### 5.3 Running the API Service Locally
1. **Activate environment**
```powershell
conda activate deglare310
```

2. **Confirm checkpoint**
```powershell
Test-Path .\checkpoints\best_model.pth
```
Optional checkpoint override:
```powershell
DEGLARE_CHECKPOINT_PATH="Your\absolute\path\to\best_model.pth"
```

3. **Start server**
```powershell
uvicorn api.app:app --host 0.0.0.0 --port 4000
```

4. **Verify health**
```powershell
curl http://127.0.0.1:4000/ping
```
Expected:
```json
{"message":"pong"}
```

5. **Manual inference via docs UI**
```text
http://127.0.0.1:4000/docs
```

## 6) Artefact #3 - Dockerfile (Containerized API)

### 6.1 Dockerfile Summary
Path:
```text
Dockerfile
```
The image uses `python:3.10-slim`, installs dependencies from `requirements.txt`, copies `api/` and `notebooks/`, creates `/app/checkpoints`, exposes port `4000`, and starts Uvicorn with `api.app:app`. The service runs inference on CPU inside the container and expects `best_model.pth` under `/app/checkpoints` at runtime.

### 6.2 Prerequisites
```text
1) Docker Desktop (or Docker Engine) is installed and running.
2) checkpoints/best_model.pth exists locally before running the container.
```

### 6.3 Build and Run
1. **Build image**
```powershell
docker build -t deglare-service .
```

2. **Run container**
```powershell
docker run -p 4000:4000 -v "${PWD}\checkpoints:/app/checkpoints" deglare-service
```
Volume flag explanation:
```text
-v mounts local checkpoints/ into /app/checkpoints so model weights are supplied at runtime and not baked into the image.
```

3. **Verify container service**
```powershell
curl http://127.0.0.1:4000/ping
```
Expected:
```json
{"message":"pong"}
```

## 7) Artefact #4 - Endpoint Test Script

### 7.1 Test File Description
Path:
```text
tests/test_client.py
```
This is a standalone script (no pytest required) that validates both API endpoints against a running service. It checks `/ping` status/payload and `/infer` output schema. For inference testing it builds an in-memory synthetic RGB image, uploads it as multipart, base64-decodes the response, and verifies output mode `L` with size `512x512`. The script prints per-endpoint PASS/FAIL lines and a final success/failure summary.

### 7.2 Prerequisites
```text
Start the API first, either:
- local Uvicorn run , or
- Docker container run .
```

### 7.3 Running the Tests
1. **Start service**
```text
uvicorn api.app:app --host 0.0.0.0 --port 4000
```

2. **Run against default URL**
```powershell
python tests/test_client.py
```
Default base URL:
```text
http://127.0.0.1:4000
```

3. **Expected terminal output**
Passing example:
```text
[PASS] /ping
[PASS] /infer
All checks passed.
```


## 8) Submission Notes
This repository is organized to satisfy the assessment deliverables with a clear split between training code (`notebooks/`) and serving code (`api/`).  
The checkpoint artifact (`.pth`) is produced by notebook training and consumed by both local FastAPI and Dockerized inference flows.
