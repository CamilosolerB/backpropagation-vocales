"""Ventana principal: asistente guiado en 4 pasos numerados con cabecera
institucional (UDEC)."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.paths import asset_path
from ui.state import AppState, STUDENTS
from ui.step_architecture import StepArchitecture
from ui.step_data import StepData
from ui.step_results import StepResults
from ui.step_training import StepTraining

TITULO = "Laboratorio #3 — Redes Neuronales Artificiales · RNA Backpropagation"
SUBTITULO = "Clasificación de vocales en una grilla 5x7 con red neuronal backpropagation"


def _build_branding_header():
    header = QWidget()
    header.setStyleSheet("background-color: #1a5276; color: white; padding: 6px;")
    layout = QHBoxLayout(header)
    layout.setContentsMargins(16, 6, 16, 6)
    layout.setSpacing(14)

    logo_label = QLabel()
    logo_label.setFixedSize(84, 44)
    path = asset_path("logo_udec.png")
    try:
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaled(84, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            logo_label.setText("LOGO UDEC")
            logo_label.setStyleSheet("border:1px dashed #c5d8ec; font-size:8px;")
    except Exception:
        logo_label.setText("LOGO UDEC")
        logo_label.setStyleSheet("border:1px dashed #c5d8ec; font-size:8px;")
    layout.addWidget(logo_label)

    texts = QVBoxLayout()
    texts.setSpacing(1)
    title = QLabel(TITULO)
    title.setStyleSheet("font-size:16px; font-weight:bold; color:#ffffff;")
    title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    subtitle = QLabel(SUBTITULO)
    subtitle.setStyleSheet("font-size:11px; color:#d6eaf8;")
    students = QLabel(" · ".join(STUDENTS))
    students.setStyleSheet("font-size:11px; font-weight:bold; color:#f9e79f;")
    texts.addWidget(title)
    texts.addWidget(subtitle)
    texts.addWidget(students)
    layout.addLayout(texts, 1)
    return header


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.setWindowTitle("BACKPROP_VOCALES — Laboratorio #3 de Redes Neuronales")
        self.resize(1280, 860)

        self.step_data = StepData(self.state)
        self.step_architecture = StepArchitecture(self.state)
        self.step_training = StepTraining(self.state)
        self.step_results = StepResults(self.state)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.step_data, "1. DATOS")
        self.tabs.addTab(self.step_architecture, "2. ARQUITECTURA")
        self.tabs.addTab(self.step_training, "3. ENTRENAMIENTO")
        self.tabs.addTab(self.step_results, "4. RESULTADOS Y PRUEBA")
        self.tabs.setDocumentMode(True)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(_build_branding_header())
        central_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.state.data_changed.connect(self._on_data_changed)
        self.state.config_changed.connect(self.step_training.refresh_enabled)
        self.step_training.training_finished.connect(self._on_training_finished)
        self.tabs.setCurrentIndex(0)

    def _on_data_changed(self):
        self.step_training.refresh_enabled()
        if not self.state.trained():
            self.step_results.refresh()
            return
        answer = QMessageBox.question(
            self,
            "Datos cambiados",
            "El dataset cambió tras haber entrenado. ¿Desea descartar el "
            "entrenamiento anterior y empezar de nuevo?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self.state.result = None
            self.state.network = None
            self.step_results.refresh()

    def _on_training_finished(self, result):
        self.step_results.refresh()
        self.tabs.setCurrentWidget(self.step_results)

    def closeEvent(self, event):
        self.step_training.cancel_and_wait()
        event.accept()