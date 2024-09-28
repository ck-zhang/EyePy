import numpy as np
import cv2


def calculate_head_pose(shape):
    image_points = np.array(
        [
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),  # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye left corner
            (shape.part(45).x, shape.part(45).y),  # Right eye right corner
            (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
            (shape.part(54).x, shape.part(54).y),  # Right mouth corner
        ],
        dtype="double",
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ]
    )

    camera_matrix = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    pitch = (np.degrees(x) + 360) % 360
    yaw = np.degrees(y)

    return pitch, yaw
