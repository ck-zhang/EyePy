import pygame
import cv2
import numpy as np
import csv
import os
from .process_faces import initialize_face_processing, process_frame_for_face_data


def collect_data(camera_index=0):

    csv_directory = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(csv_directory, exist_ok=True)
    csv_file_path = os.path.join(csv_directory, "face_data.csv")
    csv_file = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Timestamp", "Data", "Click X", "Click Y"])

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    pygame.init()
    infoObject = pygame.display.Info()
    screen = pygame.display.set_mode(
        (infoObject.current_w, infoObject.current_h), pygame.FULLSCREEN
    )

    detector, predictor = initialize_face_processing()

    running = True
    waiting_for_face = False
    click_x, click_y = None, None

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting ...")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        pygame_frame = pygame.surfarray.make_surface(frame_rgb)
        screen.blit(pygame_frame, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click_x, click_y = event.pos
                waiting_for_face = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False

        if waiting_for_face and click_x is not None and click_y is not None:
            face_data = process_frame_for_face_data(frame, detector, predictor)
            if face_data:
                print(
                    f"Face Data: {face_data} Click Coordinates: ({click_x}, {click_y})"
                )
                csv_writer.writerow(
                    [pygame.time.get_ticks(), face_data, click_x, click_y]
                )
                waiting_for_face = False
                click_x, click_y = (
                    None,
                    None,
                )
            else:
                print("Trying to detect face...")

    csv_file.close()
    cap.release()
    pygame.quit()
