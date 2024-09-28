import cv2
import dlib
import math
from .gaze_tracking.gaze_tracking import GazeTracking
from .tilt_detection import calculate_head_pose


gaze = GazeTracking()


def initialize_face_processing():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor


def process_frame_for_face_data(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if faces:
        face = faces[0]
        landmarks = predictor(gray, face)
        pitch, yaw = calculate_head_pose(landmarks)
        gaze.refresh(frame, landmarks)
        horizontal_ratio = gaze.horizontal_ratio()
        vertical_ratio = gaze.vertical_ratio()
        try:
            return {
                "yaw": yaw,
                "pitch": pitch,
                "horizontal_ratio": 1 - horizontal_ratio,
                "vertical_ratio": 1 - vertical_ratio,
            }
        except:
            pass

    return None
