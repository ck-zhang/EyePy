import cv2
import numpy as np
import tkinter as tk
import time
from gaze_estimator import GazeEstimator


def run_calibration(gaze_estimator, camera_index=0):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    A = screen_width * 0.4
    B = screen_height * 0.4
    a = 3
    b = 2
    delta = 0

    total_time = 5
    fps = 60
    total_frames = int(total_time * fps)

    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(camera_index)

    features_list = []
    targets_list = []

    def lissajous_curve(t, A, B, a, b, delta):
        x = A * np.sin(a * t + delta) + screen_width / 2
        y = B * np.sin(b * t) + screen_height / 2
        return x, y

    face_detected = False
    countdown_active = False
    face_detection_start_time = None
    countdown_duration = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        features, blink_detected = gaze_estimator.extract_features(frame)
        if features is not None and not blink_detected:
            face_detected = True
        else:
            face_detected = False

        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        current_time = time.time()

        if face_detected:
            if not countdown_active:
                face_detection_start_time = current_time
                countdown_active = True
            elapsed_time = current_time - face_detection_start_time
            if elapsed_time >= countdown_duration:
                countdown_active = False
                break
            else:
                t = elapsed_time / countdown_duration
                eased_t = t * t * (3 - 2 * t)
                angle = 360 * (1 - eased_t)
                center = (screen_width // 2, screen_height // 2)
                radius = 50
                axes = (radius, radius)
                start_angle = -90
                end_angle = start_angle + angle
                color = (0, 255, 0)
                thickness = -1
                cv2.ellipse(
                    canvas, center, axes, 0, start_angle, end_angle, color, thickness
                )
        else:
            countdown_active = False
            face_detection_start_time = None
            text = "Face not detected"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            color = (0, 0, 255)
            thickness = 3
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (screen_width - text_size[0]) // 2
            text_y = (screen_height + text_size[1]) // 2
            cv2.putText(
                canvas, text, (text_x, text_y), font, font_scale, color, thickness
            )

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyWindow("Calibration")
            return

    start_time = time.time()
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        t = (time.time() - start_time) * (2 * np.pi / total_time)
        x, y = lissajous_curve(t, A, B, a, b, delta)
        x, y = int(x), int(y)

        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.circle(canvas, (x, y), 20, (0, 255, 0), -1)

        cv2.imshow("Calibration", canvas)
        cv2.waitKey(1)

        features, blink_detected = gaze_estimator.extract_features(frame)
        if features is not None and not blink_detected:
            features_list.append(features)
            targets_list.append([x, y])

    cap.release()
    cv2.destroyWindow("Calibration")

    X = np.array(features_list)
    y = np.array(targets_list)

    gaze_estimator.train(X, y)


def fine_tune_kalman_filter(gaze_estimator, kalman, camera_index=0):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    initial_points = [
        {
            "position": (screen_width // 2, screen_height // 4),
            "start_time": None,
            "data_collection_started": False,
            "collection_start_time": None,
            "collected_gaze": [],
        },
        {
            "position": (screen_width // 4, 3 * screen_height // 4),
            "start_time": None,
            "data_collection_started": False,
            "collection_start_time": None,
            "collected_gaze": [],
        },
        {
            "position": (3 * screen_width // 4, 3 * screen_height // 4),
            "start_time": None,
            "data_collection_started": False,
            "collection_start_time": None,
            "collected_gaze": [],
        },
    ]

    points = initial_points.copy()

    proximity_threshold = screen_width / 5
    initial_delay = 0.5
    data_collection_duration = 0.5

    cv2.namedWindow("Fine Tuning", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Fine Tuning", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(camera_index)

    gaze_positions = []

    while len(points) > 0:
        ret, frame = cap.read()
        if not ret:
            continue

        features, blink_detected = gaze_estimator.extract_features(frame)
        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        for point in points:
            cv2.circle(canvas, point["position"], 20, (0, 255, 0), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (255, 255, 255)
        thickness = 2
        text = "Look at the points until they disappear"
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (screen_width - text_size[0]) // 2
        text_y = screen_height - 50
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color, thickness)

        current_time = time.time()

        if features is not None and not blink_detected:
            X = np.array([features])
            gaze_point = gaze_estimator.predict(X)[0]
            gaze_x, gaze_y = int(gaze_point[0]), int(gaze_point[1])

            cv2.circle(canvas, (gaze_x, gaze_y), 10, (255, 0, 0), -1)

            for point in points[:]:
                dx = gaze_x - point["position"][0]
                dy = gaze_y - point["position"][1]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance <= proximity_threshold:
                    if point["start_time"] is None:
                        point["start_time"] = current_time
                        point["data_collection_started"] = False
                        point["collection_start_time"] = None
                        point["collected_gaze"] = []
                    elapsed_time = current_time - point["start_time"]

                    if (
                        not point["data_collection_started"]
                        and elapsed_time >= initial_delay
                    ):
                        point["data_collection_started"] = True
                        point["collection_start_time"] = current_time
                        point["collected_gaze"] = []

                    if point["data_collection_started"]:
                        data_collection_elapsed = (
                            current_time - point["collection_start_time"]
                        )
                        point["collected_gaze"].append([gaze_x, gaze_y])

                        shake_amplitude = int(
                            5
                            + (data_collection_elapsed / data_collection_duration) * 20
                        )
                        shake_x = int(
                            np.random.uniform(-shake_amplitude, shake_amplitude)
                        )
                        shake_y = int(
                            np.random.uniform(-shake_amplitude, shake_amplitude)
                        )
                        shaken_position = (
                            int(point["position"][0] + shake_x),
                            int(point["position"][1] + shake_y),
                        )
                        cv2.circle(canvas, shaken_position, 20, (0, 255, 0), -1)

                        if data_collection_elapsed >= data_collection_duration:
                            gaze_positions.extend(point["collected_gaze"])
                            points.remove(point)
                    else:
                        cv2.circle(canvas, point["position"], 25, (0, 255, 255), 2)
                else:
                    point["start_time"] = None
                    point["data_collection_started"] = False
                    point["collection_start_time"] = None
                    point["collected_gaze"] = []
        else:
            for point in points:
                point["start_time"] = None
                point["data_collection_started"] = False
                point["collection_start_time"] = None
                point["collected_gaze"] = []

        cv2.imshow("Fine Tuning", canvas)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyWindow("Fine Tuning")
            return

    cap.release()
    cv2.destroyWindow("Fine Tuning")

    gaze_positions = np.array(gaze_positions)
    if gaze_positions.shape[0] < 2:
        return

    gaze_variance = np.var(gaze_positions, axis=0)
    gaze_variance[gaze_variance == 0] = 1e-4

    kalman.measurementNoiseCov = np.array(
        [[gaze_variance[0], 0], [0, gaze_variance[1]]], dtype=np.float32
    )
