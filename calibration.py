import cv2
import numpy as np
import tkinter as tk
import time
from gaze_estimator import GazeEstimator


def wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, dur=2):
    """
    Waits for a face to be detected (not blinking), then does a countdown ellipse.
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fd_start = None
    countdown = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        f, blink = gaze_estimator.extract_features(frame)
        face = f is not None and not blink
        c = np.zeros((sh, sw, 3), dtype=np.uint8)
        now = time.time()
        if face:
            if not countdown:
                fd_start = now
                countdown = True
            elapsed = now - fd_start
            if elapsed >= dur:
                return True
            t = elapsed / dur
            e = t * t * (3 - 2 * t)
            ang = 360 * (1 - e)
            cv2.ellipse(
                c, (sw // 2, sh // 2), (50, 50), 0, -90, -90 + ang, (0, 255, 0), -1
            )
        else:
            countdown = False
            fd_start = None
            txt = "Face not detected"
            fs = 2
            thick = 3
            size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            tx = (sw - size[0]) // 2
            ty = (sh + size[1]) // 2
            cv2.putText(
                c, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), thick
            )
        cv2.imshow("Calibration", c)
        if cv2.waitKey(1) == 27:
            return False


def run_9_point_calibration(gaze_estimator, camera_index=0):
    """
    Standard 9-point calibration
    """
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return
    mx, my = int(sw * 0.1), int(sh * 0.1)
    gw, gh = sw - 2 * mx, sh - 2 * my
    order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (0, 1), (2, 1), (1, 2)]
    pts = [(mx + int(c * (gw / 2)), my + int(r * (gh / 2))) for (r, c) in order]
    feats, targs = [], []
    pulse_d, cd_d = 1.0, 1.0
    for cycle in range(1):
        for x, y in pts:
            ps = time.time()
            final_radius = 20
            while True:
                e = time.time() - ps
                if e > pulse_d:
                    break
                r, f = cap.read()
                if not r:
                    continue
                c = np.zeros((sh, sw, 3), dtype=np.uint8)
                radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
                final_radius = radius
                cv2.circle(c, (x, y), radius, (0, 255, 0), -1)
                cv2.imshow("Calibration", c)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            cs = time.time()
            while True:
                e = time.time() - cs
                if e > cd_d:
                    break
                r, f = cap.read()
                if not r:
                    continue
                c = np.zeros((sh, sw, 3), dtype=np.uint8)
                cv2.circle(c, (x, y), final_radius, (0, 255, 0), -1)
                t = e / cd_d
                ease = t * t * (3 - 2 * t)
                ang = 360 * (1 - ease)
                cv2.ellipse(c, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
                cv2.imshow("Calibration", c)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                ft, blink = gaze_estimator.extract_features(f)
                if ft is not None and not blink:
                    feats.append(ft)
                    targs.append([x, y])
    cap.release()
    cv2.destroyAllWindows()
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))


def run_5_point_calibration(gaze_estimator, camera_index=0):
    """
    Simpler 5-point calibration
    """
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return
    m = 100
    # center, top-left, top-right, bottom-left, bottom-right
    order = [(1, 1), (0, 0), (2, 0), (0, 2), (2, 2)]
    pts = []
    for r, c in order:
        x = m if c == 0 else (sw - m if c == 2 else sw // 2)
        y = m if r == 0 else (sh - m if r == 2 else sh // 2)
        pts.append((x, y))
    feats, targs = [], []
    pd, cd = 1.0, 1.0
    for cycle in range(1):
        for x, y in pts:
            ps = time.time()
            final_radius = 20
            while True:
                e = time.time() - ps
                if e > pd:
                    break
                r, f = cap.read()
                if not r:
                    continue
                c = np.zeros((sh, sw, 3), dtype=np.uint8)
                radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
                final_radius = radius
                cv2.circle(c, (x, y), radius, (0, 255, 0), -1)
                cv2.imshow("Calibration", c)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            cs = time.time()
            while True:
                e = time.time() - cs
                if e > cd:
                    break
                r, f = cap.read()
                if not r:
                    continue
                c = np.zeros((sh, sw, 3), dtype=np.uint8)
                cv2.circle(c, (x, y), final_radius, (0, 255, 0), -1)
                t = e / cd
                ease = t * t * (3 - 2 * t)
                ang = 360 * (1 - ease)
                cv2.ellipse(c, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
                cv2.imshow("Calibration", c)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                ft, blink = gaze_estimator.extract_features(f)
                if ft is not None and not blink:
                    feats.append(ft)
                    targs.append([x, y])
    cap.release()
    cv2.destroyAllWindows()
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))


def run_lissajous_calibration(gaze_estimator, camera_index=0):
    """
    Moves a calibration point in a Lissajous curve
    """
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return
    A, B, a, b, d = sw * 0.4, sh * 0.4, 3, 2, 0

    def curve(t):
        return (A * np.sin(a * t + d) + sw / 2, B * np.sin(b * t) + sh / 2)

    tt = 5.0
    fps = 60
    frames = int(tt * fps)
    feats, targs = [], []
    vals = []
    acc = 0

    # Generate a time scale that speeds up / slows down sinusoidally
    for i in range(frames):
        frac = i / (frames - 1)
        spd = 0.3 + 0.7 * np.sin(np.pi * frac)
        acc += spd / fps
    end = acc
    if end < 1e-6:
        end = 1e-6
    acc = 0

    for i in range(frames):
        frac = i / (frames - 1)
        spd = 0.3 + 0.7 * np.sin(np.pi * frac)
        acc += spd / fps
        t = (acc / end) * (2 * np.pi)
        ret, f = cap.read()
        if not ret:
            continue
        x, y = curve(t)
        c = np.zeros((sh, sw, 3), dtype=np.uint8)
        cv2.circle(c, (int(x), int(y)), 20, (0, 255, 0), -1)
        cv2.imshow("Calibration", c)
        if cv2.waitKey(1) == 27:
            break
        ft, blink = gaze_estimator.extract_features(f)
        if ft is not None and not blink:
            feats.append(ft)
            targs.append([x, y])

    cap.release()
    cv2.destroyAllWindows()
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))


def fine_tune_kalman_filter(gaze_estimator, kalman, camera_index=0):
    """
    Quick fine-tuning pass to adjust Kalman filter's measurementNoiseCov.
    """
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
                            point["position"][0] + shake_x,
                            point["position"][1] + shake_y,
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
