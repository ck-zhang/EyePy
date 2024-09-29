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

    points = [
        (screen_width / 2, screen_height / 2),  # Middle
        (50, 50),  # Top left
        (screen_width - 50, 50),  # Top right
        (50, screen_height - 50),  # Bottom left
        (screen_width - 50, screen_height - 50),  # Bottom right
        (50, 50),  # Top left
        (50, screen_height - 50),  # Bottom left
        (screen_width - 50, 50),  # Top right
        (screen_width - 50, screen_height - 50),  # Bottom right
        (screen_width / 2, screen_height / 2),  # Middle
    ]

    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(camera_index)

    features_list = []
    targets_list = []

    N = 30  # Frames per movement

    def ease_in_out_quad(t):
        return t * t * (3 - 2 * t)

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

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]

        for frame_idx in range(N):
            ret, frame = cap.read()
            if not ret:
                continue

            t = frame_idx / (N - 1)
            eased_t = ease_in_out_quad(t)

            x = int(p0[0] + (p1[0] - p0[0]) * eased_t)
            y = int(p0[1] + (p1[1] - p0[1]) * eased_t)

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


def main():
    camera_index = 1

    gaze_estimator = GazeEstimator()

    run_calibration(gaze_estimator, camera_index=camera_index)

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    cam_width, cam_height = 480, 360

    cv2.namedWindow("Gaze Estimation", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Gaze Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    cap = cv2.VideoCapture(camera_index)
    prev_time = time.time()

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
    kalman.statePre = np.zeros((4, 1), np.float32)
    kalman.statePost = np.zeros((4, 1), np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        features, blink_detected = gaze_estimator.extract_features(frame)
        if features is not None and not blink_detected:
            X = np.array([features])
            gaze_point = gaze_estimator.predict(X)[0]
            x, y = int(gaze_point[0]), int(gaze_point[1])

            prediction = kalman.predict()
            x_pred, y_pred = int(prediction[0]), int(prediction[1])

            measurement = np.array([[np.float32(x)], [np.float32(y)]])
            if np.count_nonzero(kalman.statePre) == 0:
                kalman.statePre[:2] = measurement
                kalman.statePost[:2] = measurement
            kalman.correct(measurement)
        else:
            x_pred, y_pred = None, None
            blink_detected = True

        small_frame = cv2.resize(frame, (cam_width, cam_height))

        canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        canvas[:cam_height, :cam_width] = small_frame

        if x_pred is not None and y_pred is not None:
            cv2.circle(canvas, (x_pred, y_pred), 20, (0, 0, 255), -1)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            canvas,
            f"FPS: {int(fps)}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        blink_text = "Blinking" if blink_detected else "Not Blinking"
        cv2.putText(
            canvas,
            blink_text,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if not blink_detected else (0, 0, 255),
            2,
        )

        cv2.imshow("Gaze Estimation", canvas)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
