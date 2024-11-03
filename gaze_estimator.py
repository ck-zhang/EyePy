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
        Takes in image and returns landmarks around the eye region.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        left_eye_indices = [
            # Upper brow
            107,
            66,
            105,
            63,
            70,
            # Lower brow
            55,
            65,
            52,
            53,
            46,
            # Pupil center and around
            468,
            469,
            470,
            471,
            472,
            # Corners of the eye
            133,  # Inner eye corner
            33,  # Outer eye corner
            # Eye upper
            173,
            157,
            158,
            159,
            160,
            161,
            246,
            # Eye lower
            155,
            154,
            153,
            145,
            144,
            163,
            7,
            # First layer around eye
            243,
            190,
            56,
            28,
            27,
            29,
            30,
            247,
            130,
            25,
            110,
            24,
            23,
            22,
            26,
            112,
            # Second layer around eye
            244,
            189,
            221,
            222,
            223,
            224,
            225,
            113,
            226,
            31,
            228,
            229,
            230,
            231,
            232,
            233,
            # Third layer around eye
            193,
            245,
            128,
            121,
            120,
            119,
            118,
            117,
            111,
            35,
            124,
            143,
            156,
        ]

        right_eye_indices = [
            # Upper brow
            336,
            296,
            334,
            293,
            300,
            # Lower brow
            285,
            295,
            282,
            283,
            276,
            # Pupil center and around
            473,
            476,
            475,
            474,
            477,
            # Corners of the eye
            362,  # Inner eye corner
            263,  # Outer eye corner
            # Eye upper
            398,
            384,
            385,
            386,
            387,
            388,
            466,
            # Eye lower
            382,
            381,
            380,
            374,
            373,
            390,
            249,
            # First layer around eye
            463,
            414,
            286,
            258,
            257,
            259,
            260,
            467,
            359,
            255,
            339,
            254,
            253,
            252,
            256,
            341,
            # Second layer around eye
            464,
            413,
            441,
            442,
            443,
            444,
            445,
            342,
            446,
            261,
            448,
            449,
            450,
            451,
            452,
            453,
            # Third layer around eye
            417,
            465,
            357,
            350,
            349,
            348,
            347,
            346,
            340,
            265,
            353,
            372,
            383,
        ]

        mutual_indices = [4, 10, 151, 9, 152, 234, 454, 288, 58]

        eye_landmarks = [
            landmarks[i] for i in left_eye_indices + right_eye_indices + mutual_indices
        ]
        features = np.array([(lm.x, lm.y, lm.z) for lm in eye_landmarks]).flatten()

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

    def train(self, X, y, alpha=1.0, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self.variable_scaling = variable_scaling

        if self.use_separate_models:
            X_x = X
            X_y = X

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

            X_x_scaled = self.scaler_x.transform(X)
            X_y_scaled = self.scaler_y.transform(X)

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
