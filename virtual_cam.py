import argparse
import time
import cv2
import numpy as np
import tkinter as tk
import pyvirtualcam
from scipy.stats import gaussian_kde
from gaze_estimator import GazeEstimator
from calibration import (
    run_9_point_calibration,
    run_5_point_calibration,
    run_lissajous_calibration,
    fine_tune_kalman_filter,
)


def main():
    parser = argparse.ArgumentParser(
        description="Virtual Camera Gaze Overlay (v4l2loopback)"
    )
    parser.add_argument("--filter", choices=["kalman", "kde", "none"], default="none")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--calibration", choices=["9p", "5p", "lissajous"], default="9p"
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence

    gaze_estimator = GazeEstimator()
    if calibration_method == "9p":
        run_9_point_calibration(gaze_estimator, camera_index=camera_index)
    elif calibration_method == "5p":
        run_5_point_calibration(gaze_estimator, camera_index=camera_index)
    else:
        run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    kalman = None
    if filter_method == "kalman":
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 10
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        kalman.statePre = np.zeros((4, 1), np.float32)
        kalman.statePost = np.zeros((4, 1), np.float32)
        fine_tune_kalman_filter(gaze_estimator, kalman, camera_index=camera_index)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    green_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    green_bg[:] = (0, 255, 0)

    gaze_history = []
    time_window = 0.5
    prev_time = time.time()
    mask_prev = None
    mask_next = None
    blend_alpha = 1.0
    contours_cache = []
    last_kde_x_pred = None
    last_kde_y_pred = None
    frame_count = 0

    with pyvirtualcam.Camera(
        width=screen_width,
        height=screen_height,
        fps=cam_fps,
        fmt=pyvirtualcam.PixelFormat.BGR,
    ) as cam:
        print(f"Virtual camera started: {cam.device}")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            features, blink_detected = gaze_estimator.extract_features(frame)
            x_pred, y_pred = None, None

            if features is not None and not blink_detected:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                x, y = int(gaze_point[0]), int(gaze_point[1])

                if kalman and filter_method == "kalman":
                    prediction = kalman.predict()
                    x_pred = int(prediction[0][0])
                    y_pred = int(prediction[1][0])
                    x_pred = max(0, min(x_pred, screen_width - 1))
                    y_pred = max(0, min(y_pred, screen_height - 1))
                    measurement = np.array([[np.float32(x)], [np.float32(y)]])
                    if np.count_nonzero(kalman.statePre) == 0:
                        kalman.statePre[:2] = measurement
                        kalman.statePost[:2] = measurement
                    kalman.correct(measurement)

                elif filter_method == "kde":
                    current_time = time.time()
                    gaze_history.append((current_time, x, y))
                    gaze_history = [
                        (t, gx, gy)
                        for (t, gx, gy) in gaze_history
                        if current_time - t <= time_window
                    ]
                    if len(gaze_history) > 1 and frame_count % 5 == 0:
                        arr = np.array([[gx, gy] for (_, gx, gy) in gaze_history])
                        try:
                            kde = gaussian_kde(arr.T)
                            xi, yi = np.mgrid[0:screen_width:200j, 0:screen_height:120j]
                            coords = np.vstack([xi.ravel(), yi.ravel()])
                            zi = kde(coords).reshape(xi.shape).T
                            zi_flat = zi.flatten()
                            sort_idx = np.argsort(zi_flat)[::-1]
                            zi_sorted = zi_flat[sort_idx]
                            cumsum = np.cumsum(zi_sorted)
                            cumsum /= cumsum[-1]
                            idx = np.searchsorted(cumsum, confidence_level)
                            if idx >= len(zi_sorted):
                                idx = len(zi_sorted) - 1
                            threshold = zi_sorted[idx]
                            mask_new = np.where(zi >= threshold, 1, 0).astype(np.uint8)
                            mask_new = cv2.resize(
                                mask_new, (screen_width, screen_height)
                            )
                            kernel = np.ones((5, 5), np.uint8)
                            mask_new = cv2.morphologyEx(
                                mask_new, cv2.MORPH_OPEN, kernel
                            )
                            mask_new = cv2.morphologyEx(
                                mask_new, cv2.MORPH_CLOSE, kernel
                            )
                            mask_prev = mask_next if mask_next is not None else mask_new
                            mask_next = mask_new
                            last_kde_x_pred = int(np.mean(arr[:, 0]))
                            last_kde_y_pred = int(np.mean(arr[:, 1]))
                            blend_alpha = 0.0
                        except np.linalg.LinAlgError:
                            last_kde_x_pred = int(np.mean(arr[:, 0]))
                            last_kde_y_pred = int(np.mean(arr[:, 1]))
                    x_pred = last_kde_x_pred
                    y_pred = last_kde_y_pred

                else:
                    x_pred, y_pred = x, y

            output = green_bg.copy()

            if (
                filter_method == "kde"
                and mask_prev is not None
                and mask_next is not None
            ):
                blend_alpha = min(blend_alpha + 0.2, 1.0)
                blended_mask = cv2.addWeighted(
                    mask_prev.astype(np.float32),
                    1.0 - blend_alpha,
                    mask_next.astype(np.float32),
                    blend_alpha,
                    0,
                ).astype(np.uint8)
                kernel2 = np.ones((5, 5), np.uint8)
                blended_mask = cv2.morphologyEx(blended_mask, cv2.MORPH_OPEN, kernel2)
                blended_mask = cv2.morphologyEx(blended_mask, cv2.MORPH_CLOSE, kernel2)
                contours, _ = cv2.findContours(
                    blended_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
                )
                contours_cache = contours
                if x_pred is not None and y_pred is not None:
                    cv2.circle(output, (x_pred, y_pred), 8, (0, 0, 255), -1)

            if filter_method == "kde" and contours_cache:
                cv2.drawContours(output, contours_cache, -1, (0, 0, 255), 3)

            if filter_method != "kde" and x_pred is not None and y_pred is not None:
                cv2.circle(output, (x_pred, y_pred), 10, (0, 0, 255), -1)

            cam.send(output)
            cam.sleep_until_next_frame()
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
