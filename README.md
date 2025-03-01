# EyePy

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![GitHub stars](https://img.shields.io/github/stars/ck-zhang/EyePy.svg?style=social)

![Demo](https://github.com/user-attachments/assets/70819837-c689-4516-8b95-0952500014ff)

EyePy is a Python library that provides **webcam-based eye tracking**.
Extract facial features, train eye tracking model and predict gaze with super easy to use interface.

The repo also includes a virtual camera script allowing integration with streaming software like OBS.

## Installation

Clone this project:
```shell
git clone https://github.com/ck-zhang/EyePy
```

### Using Pip
```shell
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Using uv
```shell
# Install uv https://github.com/astral-sh/uv/?tab=readme-ov-file#installation
pip install uv
uv sync
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

## Demo

To run the gaze estimation demo:

```bash
python demo.py [OPTIONS]
```

### Options

| Option            | Description                                      | Default             |
|-------------------|--------------------------------------------------|---------------------|
| `--filter`        | Filter method (`kalman`, `kde`, `none`)          | `none`              |
| `--camera`        | Index of the camera to use                       | `0`                 |
| `--calibration`   | Calibration method (`9p`, `5p`, `lissajous`)     | `9p`                |
| `--background`    | Path to background image                         | None                |
| `--confidence`    | Confidence interval for KDE contours (0 to 1)    | `0.5`               |

## Virtual Camera Script (only tested on linux)

```bash
python virtual_cam.py [OPTIONS]
```

### Virtual Camera Options

| Option            | Description                                      | Default             |
|-------------------|--------------------------------------------------|---------------------|
| `--filter`        | Filter method (`kalman`, `kde`, `none`)          | `none`              |
| `--camera`        | Index of the camera to use                       | `0`                 |
| `--calibration`   | Calibration method (`9p`, `5p`, `lissajous`)     | `9p`                |
| `--confidence`    | Confidence interval for KDE contours (0 to 1)    | `0.5`               |

### Virtual camera demo

https://github.com/user-attachments/assets/7337f28c-6ce6-4252-981a-db77db5509f6

## Usage as library

### Initialization
```python
from EyePy import GazeEstimator
gaze_estimator = GazeEstimator()
```

### Feature Extraction
```python
import cv2
image = cv2.imread('image.jpg')
features, blink_detected = gaze_estimator.extract_features(image)

if features is None:
    print("No face detected.")
elif blink_detected:
    print("Blink detected!")
else:
    print("Extracted features:", features)
```

### Training the Model
```python
X = [[...], [...], ...]  # Each element is a feature vector
y = [[x1, y1], [x2, y2], ...]  # Corresponding gaze coordinates
gaze_estimator.train(X, y)
```

### Predicting Gaze Location
```python
predicted_gaze = gaze_estimator.predict([features])
print("Predicted gaze coordinates:", predicted_gaze[0])
```

## Future Work

### TODO

- [x] Virtual camera script ~~Integrate with OBS~~
- [ ] Integrate with opentrack

Any suggestions for features and improvements are welcome.

If you enjoyed using EyePy, consider giving it a star.
