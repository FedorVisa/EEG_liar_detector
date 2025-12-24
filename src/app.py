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
        browse_btn = QtWidgets.QPushButton("Выбрать bcrx")
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

        # Probability plot
        self.plot = pg.PlotWidget(background="w")
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
        prob_layout.addWidget(self.table)
        self.tabs.addTab(prob_tab, "Вероятности")

        phys_tab = QtWidgets.QWidget()
        phys_layout = QtWidgets.QVBoxLayout(phys_tab)
        self.phys_table = QtWidgets.QTableWidget(0, 0)
        self.phys_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        phys_layout.addWidget(self.phys_table)
        self.tabs.addTab(phys_tab, "Характеристики")

        layout.addWidget(self.tabs, stretch=1)

        self.setCentralWidget(central)

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите bcrx", str(Path.cwd()), "*.bcrx *.csv")
        if path:
            self.path_edit.setText(path)

    def run_inference(self):
        path = self.path_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Нет файла", "Укажите путь к bcrx")
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

        # Phys metrics text
        phys = result.get("phys_metrics", {}) or {}
        if phys:
            metrics = list(phys.keys())
            n_rows = len(next(iter(phys.values()))) if phys else 0
            self.phys_table.setColumnCount(len(metrics))
            self.phys_table.setRowCount(n_rows)
            self.phys_table.setHorizontalHeaderLabels(metrics)
            for c, name in enumerate(metrics):
                vals = phys.get(name, [])
                for r, v in enumerate(vals):
                    try:
                        txt = f"{float(v):.2f}"
                    except Exception:
                        txt = str(v)
                    self.phys_table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
            self.phys_table.resizeColumnsToContents()
        else:
            self.phys_table.clear()
            self.phys_table.setRowCount(0)
            self.phys_table.setColumnCount(0)

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
        # иначе передаем событие стандартному обработчику
        if self._orig_plot_mouse:
            self._orig_plot_mouse(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    model_dir = str(Path(__file__).resolve().parent / "artifacts")
    w = LieDetectorApp(model_dir=model_dir)
    w.resize(1000, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
