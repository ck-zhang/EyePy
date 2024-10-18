# EyePy

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![GitHub stars](https://img.shields.io/github/stars/ck-zhang/EyePy.svg?style=social)

This is a Python library that provides **webcam-based eye tracking**.
Extract facial features, train eye tracking model and predict gaze with super easy to use interface.

## Usage Showcase
![Demo](https://github.com/user-attachments/assets/08d7af7b-9a45-4c78-bfb5-93db1d0f45c4)

*(Note: The demo hasn't been updated yet, but the latest version now features smoother predictions and blink detection.)*

## Installation and Interactive Demo

Clone this project:
```shell
git clone https://github.com/ck-zhang/EyePy
```

### Using Pip
```shell
pip install -r requirements.txt
python demo.py
```

### Using uv
```shell
pip install uv
uv sync
uv run demo.py
```

## Usage

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
if blink_detected:
    print("Blink detected!")
else:
    print(features)
```

### Training the Model
```python
X = [...]  # Features
y = [...]  # Gaze coordinates
gaze_estimator.train(X, y)
```

### Predicting Gaze Location
```python
predicted_gaze = gaze_estimator.predict([features])
print(predicted_gaze)
```

## Future Work

### TODO

- [x] Improve demo calibration
- [x] Blink detection
- [x] Demo Kalman filter auto tuning
- [ ] Integrate with opentrack

Any suggestions for features and improvements are welcome.

If you enjoyed using EyePy, consider giving it a star.
