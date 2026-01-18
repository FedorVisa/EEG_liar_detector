import cv2

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt


POSE_CONNECTIONS = frozenset([
    # Лицо
    (0, 1), (1, 2), (2, 3), (3, 7),  # левая сторона лица
    (0, 4), (4, 5), (5, 6), (6, 8),  # правая сторона лица
    (9, 10),                          # рот

    # Плечи и туловище
    (11, 12),                         # плечи между собой
    (11, 13), (13, 15),               # левая рука
    (12, 14), (14, 16),               # правая рука
    (11, 23), (12, 24),               # плечи → бёдра
    (23, 24),                         # бёдра между собой

    # Ноги
    (23, 25), (25, 27), (27, 29), (27, 31),  # левая нога
    (24, 26), (26, 28), (28, 30), (28, 32),  # правая нога

    # Кисти (опционально — можно отключить для простоты)
    (15, 17), (17, 19), (19, 21),    # левая кисть
    (16, 18), (18, 20), (20, 22),    # правая кисть
])

def draw_pose_landmarks(image: mp.Image, detection_result) -> np.ndarray:
    img = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    if not detection_result.pose_landmarks:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for landmarks in detection_result.pose_landmarks:
        VISIBILITY_THRESHOLD = 0.5

        for lm in landmarks:
            if lm.visibility < VISIBILITY_THRESHOLD:
                continue
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            cv2.circle(img, (x_px, y_px), 5, (255, 0, 0), -1)

        # Draw connections
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            start_x, start_y = int(start_lm.x * w), int(start_lm.y * h)
            end_x, end_y = int(end_lm.x * w), int(end_lm.y * h)
            cv2.line(img, (start_x, start_y), (end_x, end_y), color=(0, 255, 0), thickness=2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

MODEL_PATH = r'C:\Users\New\PycharmProjects\geltek-research\source\models\external\pose_landmarker_lite.task'
IMAGE_PATH = r'C:\Users\New\PycharmProjects\geltek-research\source\img\test_itmo.jpg'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.IMAGE,
#     num_poses=1,
#     min_pose_detection_confidence=0.7,
#     min_pose_presence_confidence=0.7,
#     min_tracking_confidence=0.7,
#     output_segmentation_masks=False
# )

detector = vision.PoseLandmarker.create_from_options(options)
image = mp.Image.create_from_file(IMAGE_PATH)
result = detector.detect(image)

annotated = draw_pose_landmarks(image, result)
plt.figure(figsize=(12, 8))
plt.imshow(annotated)
plt.axis('off')
plt.show()

from PIL import Image
Image.fromarray(annotated).save("runs/output_test_itmo.jpg")