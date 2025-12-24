# BioRadio Lie Detector

Простой десктоп-интерфейс на PySide6 + pyqtgraph для оценки вероятности лжи по физиологическим сигналам BioRadio. В проекте есть обучение модели, инференс и GUI для просмотра результатов и метрик.

## Быстрый старт (GUI)

1. Установите зависимости (Python 3.10+):
	```bash
	pip install -r requirements.txt
	```
2. Запустите приложение:
	```bash
	python -m src.app
	```
3. В окне выберите файл с сигналом (`.csv` или `.bcrx`), при необходимости измените длину окна и шаг, нажмите «Запустить».
4. На графике появятся столбцы P(ложь). Клик по столбцу выделяет соответствующие строки в таблицах вероятностей и характеристик.

## Структура

- `src/data_io.py` — загрузка сигналов и разметки, нарезка окон.
- `src/features.py` — извлечение ECG/PPG/resp признаков.
- `src/train.py` — обучение модели, обработка дисбаланса классов, сохранение артефактов.
- `src/infer.py` — инференс файла, выдача вероятностей и метрик по окнам.
- `src/app.py` — GUI: график вероятностей, таблицы интервалов и физиологических метрик, выбор метрик и описаний.
- `src/artifacts/` — сохраненные `model.joblib`, `scaler.joblib`, `feature_names.json`, `metrics.json`.
- `data/` — примеры сигналов и разметки (BioRadio csv/bcrx, видео FRONT метки).

## Формат входных данных

- CSV BioRadio: столбцы `ECG`, `PPG`, `SpO2`, `Heart Rate`, `Elapsed Time` (500 Гц). Видео-разметка FRONT: `Start`, `End`, `Label` (в секундах).
- BCRX: обрабатывается при наличии BioRadio Reader, затем конвертируется в аналогичный CSV.

## Обучение модели

Для переобучения на своих данных:
```bash
python -m src.run_training \
  --data_dir data/BioRadio-20251223T133036Z-3-001/BioRadio \
  --labels_dir data/video
```
Артефакты будут сохранены в `src/artifacts/`.

## Инференс из кода

```python
from src.infer import predict_file

result = predict_file(
	 "data/BioRadio/Denis.csv",
	 model_dir="src/artifacts",
	 fs=500.0,
	 window_sec=10.0,
	 step_sec=2.0,
	 prob_threshold=0.5,
)
print(result["proba_per_window"])
print(result["lie_intervals"])  # [(start, end, proba), ...]
```

## Возможности GUI

- Фиксированная ось Y (0–1), подпись времени без SI-префиксов.
- Таблицы вероятностей и метрик только для чтения, одиночный выбор строк.
- Чекбокс-список метрик, кнопка показа описаний.
- Клик по столбцу графика выделяет соответствующее окно в таблицах.

## Требования

- Python 3.10+.
- PySide6, pyqtgraph, numpy, pandas, scikit-learn, joblib (см. requirements).

## TODO

- Тест end-to-end на эталонном файле с видео-разметкой.
- Упаковка инференс-утилиты для bcrx.