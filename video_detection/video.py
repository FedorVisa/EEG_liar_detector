import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

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

# --- Загрузка видеофайла ---
VIDEO_PATH = r"C:\Users\New\Downloads\Telegram Desktop\M2U00557.MPG"  # ← УКАЖИТЕ СВОЙ ПУТЬ К ВИДЕО

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Не удалось открыть видео: {VIDEO_PATH}")
    exit()

# Получаем параметры входного видео
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Видео загружено: {VIDEO_PATH}")
print(f"   Разрешение: {width}x{height}, FPS: {fps}")

# --- Создание VideoWriter для записи результата ---
OUTPUT_PATH = r"C:\Users\New\PycharmProjects\geltek-research\source\img\output_M2U00557_first_5min.mp4"

# Выбираем кодек (на Windows обычно работает 'mp4v' или 'avc1')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'XVID', 'avc1'
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"📌 Результат будет сохранён в: {OUTPUT_PATH}")
print("⏳ Обработка первых 5 минут...")

# --- Таймер: 5 минут = 300 секунд ---
start_time = time.time()
max_duration = 300  # 5 минут в секундах

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Видео закончилось раньше 5 минут.")
        break

    # Проверяем, не прошло ли 5 минут
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_duration:
        print(f"⏱️ Достигнуто ограничение: {max_duration} секунд.")
        break

    # Конвертация в RGB и создание mp.Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Детекция позы
    result = detector.detect(mp_image)

    # Визуализация
    annotated = draw_pose_landmarks(mp_image, result)

    # Конвертируем обратно в BGR для OpenCV
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # Показываем результат
    cv2.imshow('MediaPipe Pose Detection', annotated_bgr)

    # Записываем кадр в выходное видео
    out.write(annotated_bgr)

    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Видео успешно сохранено: {OUTPUT_PATH}")
print(f"   Обработано кадров: {frame_count}")
print(f"   Продолжительность: {int(elapsed_time)} секунд")