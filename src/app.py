import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.infer import predict_file


class LieDetectorApp(QtWidgets.QMainWindow):
    def __init__(self, model_dir: str):
        super().__init__()
        self.setWindowTitle("BioRadio Lie Detector")
        self.model_dir = model_dir
        self.fs = 500.0
        self.window_sec = 10.0
        self.step_sec = 2.0
        self.prob_threshold = 0.5
        self._setup_ui()

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Выбрать файл")
        browse_btn.clicked.connect(self.browse_file)
        controls.addWidget(self.path_edit)
        controls.addWidget(browse_btn)

        # Window/step controls
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(2.0, 60.0)
        self.window_spin.setValue(self.window_sec)
        self.window_spin.setSuffix(" s")
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setRange(0.5, 30.0)
        self.step_spin.setValue(self.step_sec)
        self.step_spin.setSuffix(" s")
        controls.addWidget(QtWidgets.QLabel("Окно"))
        controls.addWidget(self.window_spin)
        controls.addWidget(QtWidgets.QLabel("Шаг"))
        controls.addWidget(self.step_spin)

        run_btn = QtWidgets.QPushButton("Запустить")
        run_btn.clicked.connect(self.run_inference)
        controls.addWidget(run_btn)

        layout.addLayout(controls)

        # Probability plot with fixed SI prefixes (no Gs etc.)
        axis_bottom = pg.AxisItem(orientation="bottom")
        axis_bottom.enableAutoSIPrefix(False)
        self.plot = pg.PlotWidget(background="w", axisItems={"bottom": axis_bottom})
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "P(ложь)")
        self._orig_plot_mouse = self.plot.mousePressEvent
        self.plot.mousePressEvent = self._plot_mouse_press
        layout.addWidget(self.plot, stretch=3)

        # Results
        self.result_label = QtWidgets.QLabel("Результат не рассчитан")
        layout.addWidget(self.result_label)

        # Tabs: probabilities table / phys text
        self.tabs = QtWidgets.QTabWidget()

        prob_tab = QtWidgets.QWidget()
        prob_layout = QtWidgets.QVBoxLayout(prob_tab)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Начало (с)", "Конец (с)", "P(ложь)"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        prob_layout.addWidget(self.table)
        self.tabs.addTab(prob_tab, "Вероятности")

        phys_tab = QtWidgets.QWidget()
        phys_layout = QtWidgets.QVBoxLayout(phys_tab)

        # Metric selection list
        self.metric_list = QtWidgets.QListWidget()
        self.metric_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.metric_list.itemChanged.connect(self._on_metric_selection_changed)
        phys_layout.addWidget(QtWidgets.QLabel("Показать метрики:"))
        phys_layout.addWidget(self.metric_list, stretch=1)

        self.toggle_desc_btn = QtWidgets.QPushButton("Показать описания")
        self.toggle_desc_btn.setCheckable(True)
        self.toggle_desc_btn.toggled.connect(self._on_toggle_desc)
        phys_layout.addWidget(self.toggle_desc_btn)

        self.metric_desc = QtWidgets.QLabel("")
        self.metric_desc.setWordWrap(True)
        self.metric_desc.setVisible(False)
        phys_layout.addWidget(self.metric_desc)

        self.phys_table = QtWidgets.QTableWidget(0, 0)
        self.phys_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.phys_table.setMinimumHeight(240)
        self.phys_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.phys_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        phys_layout.addWidget(self.phys_table, stretch=2)
        self.tabs.addTab(phys_tab, "Характеристики")

        layout.addWidget(self.tabs, stretch=1)

        self.setCentralWidget(central)

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл", str(Path.cwd()), "*.bcrx *.csv")
        if path:
            self.path_edit.setText(path)

    def run_inference(self):
        path = self.path_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Нет файла", "Укажите путь к файлу с сигналом.")
            return
        self.window_sec = self.window_spin.value()
        self.step_sec = self.step_spin.value()
        try:
            result = predict_file(
                path,
                self.model_dir,
                fs=self.fs,
                window_sec=self.window_sec,
                step_sec=self.step_sec,
                prob_threshold=self.prob_threshold,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", str(e))
            return

        proba = result["proba_per_window"]
        intervals = result["lie_intervals"]
        self._phys_cache = result.get("phys_metrics", {}) or {}
        # Update label
        agg = result.get("mean_proba", 0.0)
        self.result_label.setText(f"Средняя вероятность лжи: {agg:.2f}; максимум: {result.get('max_proba', 0):.2f}")

        # Plot simple probability bar chart
        self.plot.clear()
        x = np.arange(len(proba)) * self.step_sec
        bar = pg.BarGraphItem(x=x, height=proba, width=self.step_sec * 0.8, brush="r")
        self.plot.addItem(bar)
        self.plot.setLabel("left", "P(ложь)")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self.plot.setYRange(0, 1)
        self._window_times = list(x)

        # Fill metric selector and table
        self._fill_metrics_ui()

        # Table
        self.table.setRowCount(len(intervals))
        for i, (s, e, p) in enumerate(intervals):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{s:.1f}"))
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{e:.1f}"))
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{p:.2f}"))

    def _plot_mouse_press(self, event):
        # If файл не выбран, клик по области графика открывает диалог выбора
        if not self.path_edit.text().strip():
            self.browse_file()
            return
        # If data loaded, map click to nearest window and select row
        if hasattr(self, "_window_times") and self._window_times:
            try:
                view_pos = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
                t = view_pos.x()
                times = np.array(self._window_times)
                idx = int(np.argmin(np.abs(times - t)))
                self._select_window(idx)
            except Exception:
                pass
        # иначе передаем событие стандартному обработчику
        if self._orig_plot_mouse:
            self._orig_plot_mouse(event)

    def _select_window(self, idx: int):
        if idx < 0:
            return
        # Select in prob table
        if self.table.rowCount() > idx:
            self.table.setCurrentCell(idx, 0)
            item = self.table.item(idx, 0)
            if item:
                self.table.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        # Select in phys table
        if self.phys_table.rowCount() > idx:
            self.phys_table.setCurrentCell(idx, 0)
            item = self.phys_table.item(idx, 0)
            if item:
                self.phys_table.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)

    def _fill_metrics_ui(self):
        phys = getattr(self, "_phys_cache", {}) or {}
        self.metric_list.blockSignals(True)
        self.metric_list.clear()
        metrics = list(phys.keys())
        for name in metrics:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.metric_list.addItem(item)
        self.metric_list.blockSignals(False)
        self._update_phys_table()
        self._update_metric_description()

    def _on_metric_selection_changed(self, item):
        self._update_phys_table()
        self._update_metric_description()

    def _on_toggle_desc(self, checked: bool):
        self.metric_desc.setVisible(checked)
        if checked:
            self._update_metric_description()

    def _selected_metrics(self):
        selected = []
        for i in range(self.metric_list.count()):
            it = self.metric_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                selected.append(it.text())
        return selected

    def _metric_description(self, name: str) -> str:
        desc = {
            "hr_mean": "Средняя ЧСС (уд/мин) по окну",
            "hr_std": "Ст. отклонение ЧСС",
            "rr_mean": "Средний RR-интервал (мс)",
            "rr_std": "Ст. отклонение RR (мс)",
            "rr_rmssd": "RMSSD вариабельности сердечного ритма",
            "ppg_hr_mean": "Средняя ЧСС по пульсовым пикам (уд/мин)",
            "ppg_hr_std": "Ст. отклонение ЧСС по пульсу",
            "ppg_interval_std": "Ст. отклонение интервалов между пульсовыми пиками (мс)",
            "ppg_amplitude_mean": "Средняя амплитуда пульсовой волны",
            "ppg_amplitude_std": "Ст. отклонение амплитуды пульсовой волны",
            "spo2_mean": "Средний SpO2 (%)",
            "spo2_std": "Ст. отклонение SpO2",
            "resp_rate_ecg": "Частота дыхания по ЭКГ (вдох/мин)",
            "resp_rate_ppg": "Частота дыхания по ППГ (вдох/мин)",
            "resp_rate_std_ecg": "Ст. отклонение частоты дыхания (ЭКГ)",
            "resp_rate_std_ppg": "Ст. отклонение частоты дыхания (ППГ)",
            "hrv_time_sdnn": "SDNN (вариабельность, мс)",
            "hrv_time_rmssd": "RMSSD (вариабельность, мс)",
            "hrv_freq_lf": "LF мощность HRV",
            "hrv_freq_hf": "HF мощность HRV",
        }
        return desc.get(name, "Описание недоступно")

    def _update_metric_description(self):
        selected = self._selected_metrics()
        if not selected:
            self.metric_desc.setText("Описание: выберите метрики выше")
            return
        lines = [f"{name}: {self._metric_description(name)}" for name in selected]
        self.metric_desc.setText("\n".join(lines))

    def _update_phys_table(self):
        phys = getattr(self, "_phys_cache", {}) or {}
        selected = self._selected_metrics()
        if not phys or not selected:
            self.phys_table.clear()
            self.phys_table.setRowCount(0)
            self.phys_table.setColumnCount(0)
            return
        n_rows = len(next(iter(phys.values()))) if phys else 0
        self.phys_table.setColumnCount(len(selected))
        self.phys_table.setRowCount(n_rows)
        self.phys_table.setHorizontalHeaderLabels(selected)
        for c, name in enumerate(selected):
            vals = phys.get(name, [])
            for r, v in enumerate(vals):
                try:
                    txt = f"{float(v):.2f}"
                except Exception:
                    txt = str(v)
                self.phys_table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
        self.phys_table.resizeColumnsToContents()
def main():
    app = QtWidgets.QApplication(sys.argv)
    model_dir = str(Path(__file__).resolve().parent / "artifacts")
    w = LieDetectorApp(model_dir=model_dir)
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
