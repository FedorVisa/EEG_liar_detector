import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt

# --- Настройки модели ---
MODEL_PATH = r'C:\Users\New\PycharmProjects\geltek-research\source\models\external\pose_landmarker_lite.task'

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- Connections для 21 точки (lite модель) ---
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

# --- Функция визуализации ---
def draw_pose_landmarks(image: mp.Image, detection_result) -> np.ndarray:
    img = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    if not detection_result.pose_landmarks:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for landmarks in detection_result.pose_landmarks:
        VISIBILITY_THRESHOLD = 0.5

        # Рисуем точки
        for lm in landmarks:
            if lm.visibility < VISIBILITY_THRESHOLD:
                continue
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            cv2.circle(img, (x_px, y_px), 5, (255, 0, 0), -1)

        # Рисуем линии
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            if start_lm.visibility < VISIBILITY_THRESHOLD or end_lm.visibility < VISIBILITY_THRESHOLD:
                continue
            start_x, start_y = int(start_lm.x * w), int(start_lm.y * h)
            end_x, end_y = int(end_lm.x * w), int(end_lm.y * h)
            cv2.line(img, (start_x, start_y), (end_x, end_y), color=(0, 255, 0), thickness=2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Захват видео с камеры ---
cap = cv2.VideoCapture(0)  # 0 — первая камера

if not cap.isOpened():
    print("❌ Не удалось открыть камеру!")
    exit()

print("✅ Камера запущена. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Не удалось получить кадр.")
        break

    # Конвертация в RGB и создание mp.Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Детекция позы
    result = detector.detect(mp_image)

    # Визуализация
    annotated = draw_pose_landmarks(mp_image, result)

    # Показываем результат
    cv2.imshow('MediaPipe Pose Detection', cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()