# OilTrace AI service

The Python service is organized around API routes and replaceable data providers:

```text
app/
	main.py
	api/       detection, drift, and vessel routes
	services/  satellite, weather, ocean, and AIS provider interfaces
main.py      compatibility entry point
inference.py existing U-Net inference and explicit demo fallback
preprocessing.py shared grayscale tile preprocessing
prepare_dataset.py licensed dataset download and preparation
train.py    paired-tile U-Net training
unet.py     segmentation model
```

The first provider implementations are local/sample-only. No external satellite,
weather, ocean-current, or AIS credentials are required or contacted. The
versioned detection endpoint requires an uploaded image and runs the trained
U-Net when its checkpoint exists; until then it uses an explicitly labeled,
uncalibrated CV fallback. Area in km² is only calculated when the image ground
footprint is supplied as `scene_area_km2`.

The Vite/React application remains at the repository root and uses the existing
same-origin Express API. Express proxies detection, metocean hindcast, and AIS
requests to FastAPI; React does not call external providers. Detection history
and investigation incidents are persisted in the browser's local storage.

Run locally with:

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

To obtain the labeled dataset and train from the AI-service directory:

```powershell
python prepare_dataset.py
python train.py `
	--images data/processed/oilspill-segmentation/train/images `
	--masks data/processed/oilspill-segmentation/train/masks `
	--val-images data/processed/oilspill-segmentation/val/images `
	--val-masks data/processed/oilspill-segmentation/val/masks `
	--test-images data/processed/oilspill-segmentation/test/images `
	--test-masks data/processed/oilspill-segmentation/test/masks
```

The downloader verifies the Zenodo archive checksum and preserves the source
train/validation/test scene splits. Dataset: [Oil Spill Segmentation Dataset
(800x600)](https://doi.org/10.5281/zenodo.22214144), licensed CC BY 4.0. The
prepared image tiles and checkpoint are local artifacts and are excluded from
Git. Training saves the best validation-IoU checkpoint to
`model/oiltrace_unet.pth`; the held-out test metrics are printed after training.
For the full dashboard flow, keep Uvicorn running and start the frontend/API
from a second terminal in the repository root with `npm run dev`.

The compatibility route accepts a base64 image payload at `POST /analyze`.
Responses include `segmentation_mask_base64` (a binary PNG with white oil
pixels), mask dimensions, and `mask_area_ratio`.

Versioned API endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service health |
| `POST /api/v1/detect` | Predict mask, score, and optional geospatial output |
| `POST /api/v1/drift/hindcast` | Fetch sample wind/current vectors and calculate a hindcast seed |
| `POST /api/v1/vessels/analyze` | Fetch timestamped synthetic sample AIS tracks |

For geospatial output, send `latitude`, `longitude`, and
`ground_sample_distance_m` with the uploaded image. The image is assumed to be
north-up. The response contains mask GeoJSON, image-footprint GeoJSON, and area
in km². Without these georeferencing inputs, the API returns mask coverage only
and does not claim geographic coordinates or physical area.

After a georeferenced positive detection, the dashboard calls the drift and AIS
endpoints, runs the Lagrangian model and candidate scorer, displays their layers
on Leaflet, and exports JSON/CSV or print-to-PDF reports. The initial metocean
and AIS services intentionally return sample data; AIS tracks are synthetic and
candidate ranking is not evidence of responsibility. Replace these providers
with licensed, validated feeds before operational use.

Run these endpoints locally at `http://127.0.0.1:8001` after starting Uvicorn.

## Input contract

The first model is a single-channel U-Net trained and served on 256 x 256 grayscale tiles. Training and inference resize tiles identically. The included dataset preparation script converts the dataset's single-channel labels (`0 = sea`, `1 = slick`) into the trainer's binary mask format. Validation and held-out test reports include IoU, Dice, precision, recall, and pixel accuracy. The returned per-image model score is currently uncalibrated. For other images, inputs must be calibrated and normalized consistently before they are encoded as 8-bit grayscale. This service does not ingest or calibrate raw Sentinel-1 SAFE/GRD products, convert sigma-nought units, or apply orbit/thermal-noise/terrain corrections. Prepare the selected product with a documented SAR toolchain (for example SNAP), and use that exact recipe for both training data and inference. Do not mix VV and VH or different acquisition products without defining and training for those channels.

Masks must use the same tile footprint and filename stem as their image, with oil pixels white (value >= 128) and background black. Example layout:

```text
dataset/
	images/
		tile_001.png
	masks/
		tile_001.png
```

For custom paired tiles, train and save a checkpoint:

```bash
python train.py --images ../dataset/images --masks ../dataset/masks --epochs 30
```

The default checkpoint path is `ai-service/model/oiltrace_unet.pth`. Set `OILTRACE_MODEL_PATH` to use another checkpoint. Restart the service after training. The included dataset is a small research dataset from one region and acquisition period, so its checkpoint is a development milestone, not an operationally validated detector. Evaluate on independent scenes, including look-alikes and varied sea conditions, before operational use.

If the checkpoint is absent, the service returns a dark-pixel CV mask for demo compatibility. That heuristic is not a trained detector, and its score is explicitly uncalibrated.
