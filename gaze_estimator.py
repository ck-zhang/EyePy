import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


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
        Takes in image and returns features
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None

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

        yaw, pitch, roll = self._calculate_head_orientation(landmarks)

        features = np.hstack([left_pupil_rel, right_pupil_rel, [yaw, pitch, roll]])

        # Blink detection
        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / left_eye_width

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / right_eye_width

        EAR = (left_EAR + right_EAR) / 2

        blink_threshold = 0.2

        if EAR < blink_threshold:
            blink_detected = True
        else:
            blink_detected = False

        return features, blink_detected

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

        eye_line_vector = right_eye_outer - left_eye_outer
        roll = np.arctan2(eye_line_vector[1], eye_line_vector[0])

        return yaw, pitch, roll

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
