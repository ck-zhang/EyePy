import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import tkinter as tk
import time

## Mouse code. this could be all as a seperate file i guess

import numpy as np
from pynput.mouse import Controller
import cv2

class GazeMouseController:
    def __init__(self, gaze_estimator, smooth_factor=0.1):
        self.gaze_estimator = gaze_estimator
        self.smooth_factor = smooth_factor
        self.mouse = Controller()

        # Initialize previous gaze point for smoothing
        self.prev_gaze_point = None

    def smooth_gaze(self, new_gaze_point):
        """
        Smooth the new gaze point with the previous one using a simple linear interpolation.
        """
        if self.prev_gaze_point is None:
            self.prev_gaze_point = new_gaze_point
            return new_gaze_point

        smoothed_gaze = (
            self.prev_gaze_point * (1 - self.smooth_factor)
            + new_gaze_point * self.smooth_factor
        )

        self.prev_gaze_point = smoothed_gaze
        return smoothed_gaze

    def move_mouse(self, frame):
        """
        Predict the gaze point and move the mouse smoothly.
        """
        features = self.gaze_estimator.extract_features(frame)
        if features is not None:
            # Predict the gaze location
            X = np.array([features])
            gaze_point = self.gaze_estimator.predict(X)[0]

            # Smooth the gaze point
            smoothed_gaze = self.smooth_gaze(gaze_point)

            # Move the mouse using pynput
            x, y = int(smoothed_gaze[0]), int(smoothed_gaze[1])
            self.mouse.position = (x, y)

            return (x, y)  # Return the position for feedback if needed
        return None
        
class GazeEstimator:
    def __init__(self, use_separate_models=False):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.use_separate_models = use_separate_models
        self.variable_scaling = None

        if self.use_separate_models:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            self.model_x = None
            self.model_y = None
        else:
            self.model = None
            self.scaler = StandardScaler()

    def extract_features(self, image):
        """
        Takes in image and returns features needed for gaze estimation
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        left_pupil = np.array([landmarks[468].x, landmarks[468].y])
        right_pupil = np.array([landmarks[473].x, landmarks[473].y])

        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_pupil_rel = self._calculate_relative_position(
            left_pupil, left_eye_inner, left_eye_outer, left_eye_top, left_eye_bottom
        )
        right_pupil_rel = self._calculate_relative_position(
            right_pupil,
            right_eye_inner,
            right_eye_outer,
            right_eye_top,
            right_eye_bottom,
        )

        yaw, pitch = self._calculate_head_orientation(landmarks)

        features = np.hstack([left_pupil_rel, right_pupil_rel, [yaw, pitch]])
        return features

    def _calculate_relative_position(
        self, pupil, inner_corner, outer_corner, top_point, bottom_point
    ):
        """
        Calculates relative pupil position within the eye
        """
        eye_width = np.linalg.norm(outer_corner - inner_corner)
        horizontal_pos = np.dot(pupil - inner_corner, outer_corner - inner_corner) / (
            eye_width**2
        )

        eye_height = np.linalg.norm(top_point - bottom_point)
        vertical_pos = np.dot(pupil - bottom_point, top_point - bottom_point) / (
            eye_height**2
        )

        return np.array([horizontal_pos, vertical_pos])

    def _calculate_head_orientation(self, landmarks):
        """
        Calculates head orientation
        """
        nose_tip = np.array([landmarks[1].x, landmarks[1].y])

        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        eye_center = (left_eye_outer + right_eye_outer) / 2

        yaw = nose_tip[0] - eye_center[0]
        pitch = nose_tip[1] - eye_center[1]

        return yaw, pitch

    def train(self, X, y, alpha=1.0, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self.variable_scaling = variable_scaling

        if self.use_separate_models:
            X_x = X[:, [0, 2, 4]]  # horizontal ratios and yaw
            X_y = X[:, [1, 3, 5]]  # vertical ratios and pitch

            X_x_scaled = self.scaler_x.fit_transform(X_x)
            X_y_scaled = self.scaler_y.fit_transform(X_y)

            if self.variable_scaling is not None:
                X_x_scaled *= self.variable_scaling
                X_y_scaled *= self.variable_scaling

            self.model_x = Ridge(alpha=alpha)
            self.model_y = Ridge(alpha=alpha)
            self.model_x.fit(X_x_scaled, y[:, 0])
            self.model_y.fit(X_y_scaled, y[:, 1])
        else:
            X_scaled = self.scaler.fit_transform(X)

            if self.variable_scaling is not None:
                X_scaled *= self.variable_scaling

            self.model = Ridge(alpha=alpha)
            self.model.fit(X_scaled, y)

    def predict(self, X):
        """
        Predicts gaze location
        """
        if self.use_separate_models:
            if self.model_x is None or self.model_y is None:
                raise Exception("Models are not trained yet.")

            X_x = X[:, [0, 2, 4]]  # horizontal ratios and yaw
            X_y = X[:, [1, 3, 5]]  # vertical ratios and pitch

            X_x_scaled = self.scaler_x.transform(X_x)
            X_y_scaled = self.scaler_y.transform(X_y)

            if self.variable_scaling is not None:
                X_x_scaled *= self.variable_scaling
                X_y_scaled *= self.variable_scaling

            x_pred = self.model_x.predict(X_x_scaled)
            y_pred = self.model_y.predict(X_y_scaled)
            return np.vstack((x_pred, y_pred)).T
        else:
            if self.model is None:
                raise Exception("Model is not trained yet.")

            X_scaled = self.scaler.transform(X)

            if self.variable_scaling is not None:
                X_scaled *= self.variable_scaling

            return self.model.predict(X_scaled)


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
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]

        # Loop until a valid face is detected
        valid_face_detected = False
        while not valid_face_detected:
            ret, frame = cap.read()
            if not ret:
                continue

            # Extract features from the current frame
            features = gaze_estimator.extract_features(frame)
            if features is not None:
                valid_face_detected = True
                print(f"Face detected for calibration point {p0}")

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

            # Extract features from the current frame
            features = gaze_estimator.extract_features(frame)
            if features is not None:
                features_list.append(features)
                targets_list.append([x, y])

    cap.release()
    cv2.destroyWindow("Calibration")

    X = np.array(features_list)
    y = np.array(targets_list)

    gaze_estimator.train(X, y)


def main():
    camera_index = 1

    # Initialize gaze estimator and gaze mouse controller
    gaze_estimator = GazeEstimator()
    gaze_mouse_controller = GazeMouseController(gaze_estimator, smooth_factor=0.2)

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

        # Use the gaze mouse controller to move the mouse smoothly
        gaze_position = gaze_mouse_controller.move_mouse(frame)

        if gaze_position is not None:
            x, y = gaze_position
            prediction = kalman.predict()
            x_pred, y_pred = int(prediction[0]), int(prediction[1])

            small_frame = cv2.resize(frame, (cam_width, cam_height))
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            canvas[:cam_height, :cam_width] = small_frame

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

            cv2.imshow("Gaze Estimation", canvas)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
