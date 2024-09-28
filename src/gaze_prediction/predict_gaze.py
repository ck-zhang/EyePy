import cv2
import pygame
import numpy as np
from joblib import load
import pandas as pd
from ..data_processing.process_faces import (
    initialize_face_processing,
    process_frame_for_face_data,
)
import os
import time
from scipy.stats import gaussian_kde
from skimage.measure import find_contours
import random
import matplotlib.pyplot as plt

WINDOW_LENGTH = 0.5
CONFIDENCE_LEVEL = 0.60

GRID_SIZE = 5
NUM_TRIALS = 20
TRIAL_INTERVAL = 1.0
ADJUST_TIME = 2.0
MEASUREMENT_TIME = 1.0


class KalmanFilter2D:
    def __init__(self):
        self.dt = 1.0

        self.x = np.matrix([[0], [0], [0], [0]])

        self.A = np.matrix(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.B = np.matrix([[0], [0], [0], [0]])

        self.H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.P = np.eye(self.A.shape[1]) * 1000

        self.Q = np.eye(self.A.shape[1])

        self.R = np.eye(self.H.shape[0]) * 10

    def predict(self):
        self.x = self.A * self.x + self.B

        self.P = self.A * self.P * self.A.T + self.Q

        return self.x

    def update(self, z):
        S = self.H * self.P * self.H.T + self.R
        K = self.P * self.H.T * np.linalg.inv(S)

        y = z - self.H * self.x
        self.x = self.x + K * y

        I = np.eye(self.A.shape[1])
        self.P = (I - K * self.H) * self.P

        return self.x


def predict_gaze(
    do_kde=True,
    do_accuracy_test=False,
    use_kalman_filter=False,
    center_neon_circle=False,
    feature_scales=None,
):
    if feature_scales is None:
        feature_scales = {}

    dir_path = os.path.dirname(os.path.realpath(__file__))

    model_x_path = os.path.join(
        dir_path, "..", "..", "data", "models", "ridge_regression_model_x.joblib"
    )
    model_y_path = os.path.join(
        dir_path, "..", "..", "data", "models", "ridge_regression_model_y.joblib"
    )
    scaler_x_path = os.path.join(
        dir_path, "..", "..", "data", "models", "scaler_x.joblib"
    )
    scaler_y_path = os.path.join(
        dir_path, "..", "..", "data", "models", "scaler_y.joblib"
    )

    model_x = load(model_x_path)
    model_y = load(model_y_path)
    scaler_x = load(scaler_x_path)
    scaler_y = load(scaler_y_path)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    detector, predictor = initialize_face_processing()
    pygame.init()
    infoObject = pygame.display.Info()
    screen_width = infoObject.current_w
    screen_height = infoObject.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Real-Time Gaze Prediction")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    gaze_data = []

    prediction_count = 0
    fps = 0.0
    fps_timer = 0.0

    if use_kalman_filter:
        kf = KalmanFilter2D()
        kalman_initialized = False

    if do_accuracy_test:
        trial_timer = 0.0
        trial_state = None
        trial_state_timer = 0.0
        trial_count = 0

        rect_width = screen_width / GRID_SIZE
        rect_height = screen_height / GRID_SIZE

        results = []
    else:
        trial_state = None

    running = True
    while running:
        delta_time = clock.tick(60) / 1000.0
        fps_timer += delta_time

        if do_accuracy_test:
            trial_timer += delta_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        face_data = process_frame_for_face_data(frame, detector, predictor)
        if face_data:
            prediction_count += 1
            features = {
                "yaw": [face_data["yaw"]],
                "horizontal_ratio": [face_data["horizontal_ratio"]],
                "pitch": [face_data["pitch"]],
                "vertical_ratio": [face_data["vertical_ratio"]],
            }

            for feature in features:
                features[feature][0] *= feature_scales.get(feature, 1.0)

            features_df_x = pd.DataFrame(
                {
                    "yaw": features["yaw"],
                    "horizontal_ratio": features["horizontal_ratio"],
                }
            )
            features_df_y = pd.DataFrame(
                {
                    "pitch": features["pitch"],
                    "vertical_ratio": features["vertical_ratio"],
                }
            )

            X_x_scaled = scaler_x.transform(features_df_x)
            X_y_scaled = scaler_y.transform(features_df_y)

            x_pred = model_x.predict(X_x_scaled)[0]
            y_pred = model_y.predict(X_y_scaled)[0]

            if use_kalman_filter:
                z = np.matrix([[x_pred], [y_pred]])
                if not kalman_initialized:
                    kf.x[0, 0] = x_pred
                    kf.x[1, 0] = y_pred
                    kf.x[2, 0] = 0
                    kf.x[3, 0] = 0
                    kalman_initialized = True
                else:
                    kf.predict()
                    kf.update(z)

                x_display = kf.x[0, 0]
                y_display = kf.x[1, 0]
            else:
                x_display, y_display = x_pred, y_pred

            current_time = time.time()
            gaze_data.append((current_time, x_display, y_display))

            gaze_data = [
                (t, x, y)
                for (t, x, y) in gaze_data
                if current_time - t <= WINDOW_LENGTH
            ]

            if do_kde and len(gaze_data) >= 10:
                data = np.array([[x, y] for (t, x, y) in gaze_data]).T

                kde = gaussian_kde(data, bw_method=1)

                padding = 50
                x_min, y_min = data.min(axis=1) - padding
                x_max, y_max = data.max(axis=1) + padding

                xgrid = np.linspace(x_min, x_max, 300)
                ygrid = np.linspace(y_min, y_max, 300)
                Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
                positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()])
                Z = np.reshape(kde(positions).T, Xgrid.shape)

                Z_flat = Z.ravel()
                Z_sorted = np.sort(Z_flat)[::-1]
                cumulative_sum = np.cumsum(Z_sorted)
                cumulative_sum /= cumulative_sum[-1]

                idx = np.searchsorted(cumulative_sum, CONFIDENCE_LEVEL)
                density_level = Z_sorted[idx]

                contours = find_contours(Z, density_level)

                contour_points_list = []
                for contour in contours:
                    x_contour = xgrid[contour[:, 1].astype(int)]
                    y_contour = ygrid[contour[:, 0].astype(int)]

                    points = [(int(x), int(y)) for x, y in zip(x_contour, y_contour)]

                    if len(points) > 2:
                        contour_points_list.append(points)
            else:
                contour_points_list = []
        else:
            x_display, y_display = None, None
            contour_points_list = []

        if fps_timer >= 1.0:
            fps = prediction_count / fps_timer
            fps_timer = 0.0
            prediction_count = 0

        if do_accuracy_test:
            if trial_state is None and trial_timer >= TRIAL_INTERVAL:
                if center_neon_circle:
                    circle_x = screen_width / 2
                    circle_y = screen_height / 2
                    circle_radius = min(screen_width, screen_height) * 0.05
                else:
                    selected_row = random.randint(0, GRID_SIZE - 1)
                    selected_col = random.randint(0, GRID_SIZE - 1)
                    rect_x = selected_col * rect_width
                    rect_y = selected_row * rect_height

                trial_state = "adjust"
                trial_state_timer = ADJUST_TIME
                trial_timer = 0.0
                if center_neon_circle:
                    print(
                        f"Trial {trial_count + 1}: Neon circle at center. Adjusting..."
                    )
                else:
                    print(
                        f"Trial {trial_count + 1}: Rectangle at ({selected_col}, {selected_row}) lights up. Adjusting..."
                    )

            elif trial_state == "adjust":
                trial_state_timer -= delta_time
                if trial_state_timer <= 0:
                    trial_state = "measure"
                    trial_state_timer = MEASUREMENT_TIME
                    gaze_positions = []
                    print("Measuring gaze points...")

            elif trial_state == "measure":
                trial_state_timer -= delta_time
                if x_display is not None and y_display is not None:
                    gaze_positions.append((x_display, y_display))

                if trial_state_timer <= 0:
                    if gaze_positions:
                        x_positions = [pos[0] for pos in gaze_positions]
                        y_positions = [pos[1] for pos in gaze_positions]
                        mean_x = np.mean(x_positions)
                        mean_y = np.mean(y_positions)
                        if center_neon_circle:
                            distance = np.sqrt(
                                (mean_x - circle_x) ** 2 + (mean_y - circle_y) ** 2
                            )
                            in_target = distance <= circle_radius
                            result = "inside" if in_target else "outside"
                            print(
                                f"Trial {trial_count + 1} completed. Mean gaze position is {result} the circle."
                            )
                            results.append(in_target)
                        else:
                            in_rectangle = (
                                rect_x <= mean_x < rect_x + rect_width
                                and rect_y <= mean_y < rect_y + rect_height
                            )
                            result = "inside" if in_rectangle else "outside"
                            print(
                                f"Trial {trial_count + 1} completed. Mean gaze position is {result} the rectangle."
                            )
                            results.append(in_rectangle)
                    else:
                        print(
                            f"Trial {trial_count + 1} completed. No gaze data collected."
                        )
                        results.append(False)

                    x_positions = [pos[0] for pos in gaze_positions]
                    y_positions = [pos[1] for pos in gaze_positions]

                    std_x = np.std(x_positions)
                    std_y = np.std(y_positions)
                    mad_x = np.median(np.abs(x_positions - np.median(x_positions)))
                    mad_y = np.median(np.abs(y_positions - np.median(y_positions)))

                    cov_matrix = np.cov(x_positions, y_positions)
                    sigma_x = np.sqrt(cov_matrix[0, 0])
                    sigma_y = np.sqrt(cov_matrix[1, 1])
                    rho = cov_matrix[0, 1] / (sigma_x * sigma_y)
                    bcea = 2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)

                    SNR_x = (
                        20 * np.log10(np.abs(mean_x) / std_x) if std_x != 0 else np.inf
                    )
                    SNR_y = (
                        20 * np.log10(np.abs(mean_y) / std_y) if std_y != 0 else np.inf
                    )

                    plt.figure(figsize=(10, 6))
                    plt.hist2d(
                        x_positions,
                        y_positions,
                        bins=[100, 100],
                        range=[[0, screen_width], [0, screen_height]],
                        cmap="inferno",
                    )
                    plt.colorbar(label="Number of Gaze Points")
                    plt.gca().invert_yaxis()
                    plt.xlim(0, screen_width)
                    plt.ylim(0, screen_height)

                    if center_neon_circle:
                        circle = plt.Circle(
                            (circle_x, circle_y),
                            circle_radius,
                            linewidth=2,
                            edgecolor="cyan",
                            facecolor="none",
                        )
                        plt.gca().add_patch(circle)
                    else:
                        rect = plt.Rectangle(
                            (rect_x, rect_y),
                            rect_width,
                            rect_height,
                            linewidth=2,
                            edgecolor="green",
                            facecolor="none",
                        )
                        plt.gca().add_patch(rect)

                    textstr = "\n".join(
                        (
                            f"STD X: {std_x:.2f}",
                            f"STD Y: {std_y:.2f}",
                            f"MAD X: {mad_x:.2f}",
                            f"MAD Y: {mad_y:.2f}",
                            f"BCEA: {bcea:.2f}",
                            f"SNR X: {SNR_x:.2f} dB",
                            f"SNR Y: {SNR_y:.2f} dB",
                        )
                    )

                    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
                    plt.text(
                        0.05,
                        0.95,
                        textstr,
                        transform=plt.gca().transAxes,
                        fontsize=12,
                        verticalalignment="top",
                        bbox=props,
                    )

                    plt.title(f"Gaze Heatmap for Trial {trial_count + 1}")
                    plt.xlabel("X Position")
                    plt.ylabel("Y Position")

                    heatmap_filename = f"heatmap_trial_{trial_count + 1}.png"
                    plt.savefig(heatmap_filename)
                    plt.close()
                    print(f"Heatmap saved as {heatmap_filename}")

                    trial_count += 1
                    trial_state = None
                    trial_timer = 0.0

                    if trial_count >= NUM_TRIALS:
                        total_inside = sum(results)
                        print("All trials completed.")
                        target_name = "circle" if center_neon_circle else "rectangle"
                        print(
                            f"Mean gaze position was inside the {target_name} in {total_inside} out of {NUM_TRIALS} trials."
                        )
                        running = False

        screen.fill((0, 0, 0))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        pygame_frame = pygame.surfarray.make_surface(frame_rgb)
        screen.blit(pygame_frame, (0, 0))

        if do_accuracy_test:
            if trial_state in ["adjust", "measure"]:
                if center_neon_circle:
                    pygame.draw.circle(
                        screen,
                        (0, 255, 255),
                        (int(circle_x), int(circle_y)),
                        int(circle_radius),
                        width=5,
                    )
                else:
                    pygame.draw.rect(
                        screen,
                        (0, 255, 0),
                        (rect_x, rect_y, rect_width, rect_height),
                        5,
                    )

        if x_display is not None and y_display is not None:
            pygame.draw.circle(
                screen, (255, 0, 0), (int(x_display), int(y_display)), 10
            )

        if do_kde:
            if contour_points_list:
                for points in contour_points_list:
                    pygame.draw.polygon(screen, (255, 255, 0), points, width=2)

        fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
        fps_rect = fps_text.get_rect()
        fps_rect.topright = (screen_width - 10, 10)
        screen.blit(fps_text, fps_rect)

        pygame.display.flip()

    cap.release()
    pygame.quit()
