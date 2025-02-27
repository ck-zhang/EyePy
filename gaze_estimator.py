import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class GazeEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.model = None
        self.scaler = StandardScaler()
        self.variable_scaling = None

    def extract_features(self, image):
        """
        Takes in image and returns landmarks around the eye region
        Normalization with nose tip as anchor
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # fmt: off
        left_eye_indices = [
            107, 66, 105, 63, 70,
            55, 65, 52, 53, 46,
            468, 469, 470, 471, 472,
            133, 33,
            173, 157, 158, 159, 160, 161, 246,
            155, 154, 153, 145, 144, 163, 7,
            243, 190, 56, 28, 27, 29, 30, 247,
            130, 25, 110, 24, 23, 22, 26, 112,
            244, 189, 221, 222, 223, 224, 225, 113,
            226, 31, 228, 229, 230, 231, 232, 233,
            193, 245, 128, 121, 120, 119, 118, 117,
            111, 35, 124, 143, 156,
        ]

        right_eye_indices = [
            336, 296, 334, 293, 300,
            285, 295, 282, 283, 276,
            473, 476, 475, 474, 477,
            362, 263,
            398, 384, 385, 386, 387, 388, 466,
            382, 381, 380, 374, 373, 390, 249,
            463, 414, 286, 258, 257, 259, 260, 467,
            359, 255, 339, 254, 253, 252, 256, 341,
            464, 413, 441, 442, 443, 444, 445, 342,
            446, 261, 448, 449, 450, 451, 452, 453,
            417, 465, 357, 350, 349, 348, 347, 346,
            340, 265, 353, 372, 383,
        ]

        mutual_indices = [
            4,  # Nose
            10,  # Very top
            151,  # Forehead
            9,  # Between brow
            152,  # Chin
            234,  # Very left
            454,  # Very right
            58,  # Left jaw
            288,  # Right jaw
        ]
        # fmt: on

        all_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        )
        nose_anchor = all_points[4]
        left_corner = all_points[33]
        right_corner = all_points[263]
        top_of_head = all_points[10]

        shifted_points = all_points - nose_anchor
        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_approx = top_of_head - nose_anchor
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T

        left_corner_rot = R.T @ (left_corner - nose_anchor)
        right_corner_rot = R.T @ (right_corner - nose_anchor)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist

        subset_indices = left_eye_indices + right_eye_indices + mutual_indices
        eye_landmarks = rotated_points[subset_indices]
        features = eye_landmarks.flatten()

        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        features = np.concatenate([features, [yaw, pitch, roll]])

        # Blink detection
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        EAR = (left_EAR + right_EAR) / 2
        blink_detected = EAR < 0.2

        return features, blink_detected

    def train(self, X, y, alpha=1.0, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self.variable_scaling = variable_scaling

        X_scaled = self.scaler.fit_transform(X)
        if self.variable_scaling is not None:
            X_scaled *= self.variable_scaling

        self.model = Ridge(alpha=alpha)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        """
        Predicts gaze location
        """
        if self.model is None:
            raise Exception("Model is not trained yet.")

        X_scaled = self.scaler.transform(X)
        if self.variable_scaling is not None:
            X_scaled *= self.variable_scaling

        return self.model.predict(X_scaled)
