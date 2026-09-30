"""PASO 3 DE 4 — ENTRENAMIENTO: corre en un hilo en segundo plano (nunca
congela la interfaz) con controles de contador, panel de predicciones por
vocal (✓/✗ automático por argmax) y modos continuo y paso a paso, pausa y
cancelación. Las gráficas (curva ET, arquitectura, matriz de confusión,
plantillas y evolución de pesos) se muestran en el paso 4 de resultados.
"""

import threading

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.network import BackpropagationNetwork
from core.trainer import BackpropTrainer, TrainingConfig
from data.dataset import INPUT_SIZE, N_CLASSES, VOWELS
from ui.widgets import BitmapView, HelpButton
from ui.state import HELP

_HEADER = (
    "PASO 3 DE 4 — ENTRENAMIENTO: pulse ENTRENAR y observe en vivo cómo cada "
    "vocal pasa de ✗ a ✓ y cómo bajan el error y el contador de época. "
    "Las gráficas de resultados se muestran en el paso 4."
)


class TrainingWorker(QThread):
    epoch_done = Signal(object)
    training_done = Signal(object)
    training_failed = Signal(str)

    def __init__(self, network, train_x, train_y, val_x, val_y, config,
                 cancel_event, step_event, pause_event, ui_every, step_mode,
                 parent=None):
        super().__init__(parent)
        self._network = network
        self._train_x = train_x
        self._train_y = train_y
        self._val_x = val_x
        self._val_y = val_y
        self._config = config
        self._cancel_event = cancel_event
        self._step_event = step_event
        self._pause_event = pause_event
        self._ui_every = max(1, ui_every)
        self._step_mode = step_mode
        self.result = None

    def run(self):
        trainer = BackpropTrainer(
            self._network, self._train_x, self._train_y,
            self._val_x, self._val_y,
        )
        emitted = 0

        def on_epoch(rec):
            nonlocal emitted
            if self._step_mode or (rec.epoch - emitted) >= self._ui_every:
                if rec.weights is None:
                    try:
                        rec.weights, rec.biases = self._network.copy_params()
                    except Exception:
                        pass
                self.epoch_done.emit(rec)
                emitted = rec.epoch

        try:
            result = trainer.train(
                self._config,
                on_epoch=on_epoch,
                cancel_event=self._cancel_event,
                step_event=self._step_event if self._step_mode else None,
                pause_event=self._pause_event,
            )
        except Exception as exc:
            self.training_failed.emit(str(exc))
            return
        if result.history:
            last = result.history[-1]
            if last is not None and last.epoch != emitted:
                on_epoch(last)
        self.result = result
        self.training_done.emit(result)


class StepTraining(QWidget):
    training_finished = Signal(object)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker = None
        self._cancel_event = None
        self._step_event = None
        self._pause_event = None
        self._network = None
        self._ui_history = []
        self._reps = {}  # clase -> (índice en entrenamiento, patrón)
        self._last_epoch = 0

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        content = QVBoxLayout(container)
        content.setContentsMargins(14, 12, 14, 14)
        content.setSpacing(8)

        header = QLabel(_HEADER)
        header.setWordWrap(True)
        header.setStyleSheet(
            "background:#eaf2fa; border:1px solid #c5d8ec; border-radius:6px;"
            "padding:8px; font-size:12px; font-weight:bold; color:#1a5276;"
        )
        content.addWidget(header)

        content.addWidget(self._build_session_box())
        content.addWidget(self._build_controls_box())
        content.addWidget(self._build_status_box())

        content.addWidget(self._build_live_patterns_box())

        content.addStretch(1)
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.refresh_enabled()

    # ------------------------------------------------------------------
    def _build_session_box(self):
        box = QGroupBox("SESIÓN")
        layout = QVBoxLayout(box)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color:#1a5276; font-weight:bold;")
        layout.addWidget(self.summary_label)
        self.partition_label = QLabel("")
        self.partition_label.setStyleSheet("color:#555;")
        layout.addWidget(self.partition_label)
        row = QHBoxLayout()
        row.addWidget(QLabel("Modo de entrenamiento:"))
        self.mode_continuous = QRadioButton("Continuo")
        self.mode_step = QRadioButton("Paso a paso (una época por clic)")
        self.mode_continuous.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_continuous)
        self.mode_group.addButton(self.mode_step)
        row.addWidget(self.mode_continuous)
        row.addWidget(self.mode_step)
        row.addWidget(HelpButton(HELP["epoch"]))
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _build_controls_box(self):
        box = QGroupBox("CONTROLES")
        layout = QHBoxLayout(box)

        self.train_btn = QPushButton(" ENTRENAR ")
        self.train_btn.setStyleSheet(
            "font-size:15px; font-weight:bold; background:#1e8449; color:white;"
            "padding:10px 22px; border-radius:6px;"
        )
        self.train_btn.clicked.connect(self._start_training)
        layout.addWidget(self.train_btn)

        self.step_btn = QPushButton(" Siguiente época ")
        self.step_btn.setStyleSheet(
            "font-size:14px; font-weight:bold; background:#b9770e; color:white;"
            "padding:10px 18px; border-radius:6px;"
        )
        self.step_btn.clicked.connect(self._next_epoch)
        layout.addWidget(self.step_btn)

        self.pause_btn = QPushButton(" Pausar ")
        self.pause_btn.setStyleSheet(
            "font-size:14px; font-weight:bold; background:#f39c12; color:white;"
            "padding:10px 18px; border-radius:6px;"
        )
        self.pause_btn.clicked.connect(self._toggle_pause)
        layout.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton(" CANCELAR ")
        self.cancel_btn.setStyleSheet(
            "font-size:14px; font-weight:bold; background:#cb4335; color:white;"
            "padding:10px 18px; border-radius:6px;"
        )
        self.cancel_btn.clicked.connect(self._cancel_training)
        layout.addWidget(self.cancel_btn)
        layout.addStretch(1)

        legend = QLabel(
            "<span style='color:#1e8449'>■ correcto</span>  "
            "<span style='color:#922b21'>■ incorrecto</span>  "
            "<span style='color:#b9770e'>■ en progreso</span>"
        )
        legend.setStyleSheet("font-size:11px;")
        layout.addWidget(legend)
        return box

    def _build_status_box(self):
        box = QGroupBox("PROGRESO (en vivo)")
        layout = QHBoxLayout(box)
        self.epoch_label = QLabel("Época: —")
        self.epoch_label.setStyleSheet("font-weight:bold;")
        layout.addWidget(self.epoch_label)
        self.et_label = QLabel("ET actual: —")
        self.et_label.setStyleSheet("font-weight:bold;")
        layout.addWidget(self.et_label)
        self.best_label = QLabel("Mejor ET: —")
        self.best_label.setStyleSheet("font-weight:bold;")
        layout.addWidget(self.best_label)
        self.acc_label = QLabel("Aciertos: —")
        self.acc_label.setStyleSheet("font-weight:bold;")
        layout.addWidget(self.acc_label)
        self.estado_label = QLabel("Estado: en espera")
        self.estado_label.setStyleSheet(
            "font-weight:bold; font-size:13px; color:#555;"
        )
        self.estado_label.setWordWrap(True)
        layout.addWidget(self.estado_label)
        layout.addStretch(1)
        return box

    def _build_live_patterns_box(self):
        box = QGroupBox("PREDICCIÓN DE LAS VOCALES EN VIVO (✓ = acierta, ✗ = falla)")
        layout = QHBoxLayout(box)
        layout.setSpacing(14)
        self._live_cells = []
        for _ in range(N_CLASSES):
            cell = QVBoxLayout()
            view = BitmapView(scale=17, parent=self)
            cell.addWidget(view, 0, Qt.AlignCenter)
            label = QLabel("á")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size:15px; font-weight:bold; color:#888;")
            cell.addWidget(label)
            layout.addLayout(cell)
            self._live_cells.append((view, label))
        layout.addStretch(1)
        return box

    # ------------------------------------------------------------------
    def refresh_enabled(self):
        data_ok = not self.state.dataset_errors()
        self.summary_label.setText(self.state.summary_line())
        train = len(self.state.train_indices)
        val = len(self.state.val_indices)
        self.partition_label.setText(
            f"{train} patrones de entrenamiento · {val} de validación"
        )
        idle = self._worker is None
        self.mode_continuous.setEnabled(idle)
        self.mode_step.setEnabled(idle)
        self.train_btn.setEnabled(idle and data_ok)
        self.cancel_btn.setEnabled(not idle)
        self.step_btn.setEnabled(self._step_active())
        self.pause_btn.setEnabled(self._pause_active())
        self.pause_btn.setText(" Pausar ")

        if not data_ok and idle:
            self.estado_label.setText(
                "Estado: faltan patrones de alguna vocal (paso 1)."
            )
            self.estado_label.setStyleSheet("font-weight:bold; color:#922b21;")

    def _step_active(self):
        return self._worker is not None and self.mode_step.isChecked()

    def _pause_active(self):
        return self._worker is not None and self.mode_continuous.isChecked()

    def _start_training(self):
        if self._worker is not None:
            return
        errors = self.state.dataset_errors()
        if errors:
            return
        step_mode = self.mode_step.isChecked()

        self._build_reps()
        self._ui_history = []
        self._last_epoch = 0

        config = TrainingConfig(
            alpha=self.state.alpha,
            beta=self.state.beta,
            desired_error=self.state.desired_error,
            max_epochs=self.state.max_epochs,
            timeout=self.state.timeout,
        )
        self._network = BackpropagationNetwork(
            INPUT_SIZE, list(self.state.hidden_layers), N_CLASSES,
            seed=self.state.seed,
        )
        train_x, train_y, val_x, val_y = self.state.training_arrays

        self._cancel_event = threading.Event()
        self._step_event = threading.Event()
        self._pause_event = threading.Event()

        self._worker = TrainingWorker(
            self._network, train_x, train_y, val_x, val_y, config,
            self._cancel_event, self._step_event, self._pause_event,
            self.state.ui_every, step_mode, parent=self,
        )
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.training_done.connect(self._on_training_done)
        self._worker.training_failed.connect(self._on_training_failed)

        self.estado_label.setText("Estado: ENTRENANDO…")
        self.estado_label.setStyleSheet("font-weight:bold; font-size:13px; color:#b9770e;")
        self.epoch_label.setText("Época: 0")
        self.et_label.setText("ET actual: —")
        self.best_label.setText("Mejor ET: —")
        self.acc_label.setText("Aciertos: —")
        self.refresh_enabled()
        self._worker.start()

    def cancel_and_wait(self, timeout_ms=2000):
        """Cancela un entrenamiento en curso y espera a que su hilo termine
        (usado al cerrar la ventana para no dejar hilos colgados)."""
        if self._worker is None:
            return
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._worker.isRunning():
            self._worker.wait(max(0, int(timeout_ms)))

    def _cancel_training(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.estado_label.setText("Estado: cancelando…")

    def _toggle_pause(self):
        if self._pause_event is None or self._worker is None:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.pause_btn.setText(" Pausar ")
            self.estado_label.setText("Estado: ENTRENANDO…")
            self.estado_label.setStyleSheet("font-weight:bold; color:#b9770e;")
        else:
            self._pause_event.set()
            self.pause_btn.setText(" Reanudar ")
            self.estado_label.setText("Estado: PAUSADO")
            self.estado_label.setStyleSheet("font-weight:bold; color:#1a5276;")

    def _next_epoch(self):
        if self._step_event is not None and self._worker is not None:
            self._step_event.set()

    def _build_reps(self):
        self._reps = {}
        train = self.state.train_patterns
        seen = set()
        for pos, p in enumerate(train):
            if p.label not in seen:
                seen.add(p.label)
                self._reps[p.label] = (pos, p)

    def _on_epoch(self, record):
        self._last_epoch = record.epoch
        self._ui_history.append(record)
        self.epoch_label.setText(f"Época: {record.epoch}")
        self.et_label.setText(f"ET actual: {record.et:.6f}")
        self.best_label.setText(f"Mejor ET: {record.best_et:.6f}")
        self.acc_label.setText(
            f"Aciertos entren.: {record.train_correct}/{record.train_total}"
        )
        self._update_live_predictions(record)

    def _update_live_predictions(self, record):
        for c in range(N_CLASSES):
            view, label = self._live_cells[c]
            if c not in self._reps:
                view.set_grid([0] * INPUT_SIZE)
                label.setText("—")
                continue
            idx, pattern = self._reps[c]
            view.set_grid(pattern.grid)
            pred = record.train_preds[idx] if idx < len(record.train_preds) else -1
            if pred == c:
                label.setText("✓ " + VOWELS[c])
                label.setStyleSheet("font-size:14px; font-weight:bold; color:#1e8449;")
            else:
                label.setText("✗ " + (VOWELS[pred] if 0 <= pred < N_CLASSES else "?"))
                label.setStyleSheet("font-size:14px; font-weight:bold; color:#922b21;")

    def _on_training_done(self, result):
        self._worker = None
        if result.history and result.history[-1].epoch != self._last_epoch:
            self._on_epoch(result.history[-1])
        self.state.result = result
        self.state.network = self._network
        self._set_final_status(result)
        self.refresh_enabled()
        self.training_finished.emit(result)

    def _set_final_status(self, result):
        from core.trainer import StopReason

        if result.reason == StopReason.CONVERGED:
            text = (
                f"Estado: CONVERGIÓ ✓ (época {result.epochs}) · ET final "
                f"{result.final_et:.6f} · val. {result.val_correct}/"
                f"{result.val_total} aciertos"
            )
            color = "#1e8449"
        elif result.reason == StopReason.CANCELLED:
            text = "Estado: CANCELADO por el usuario"
            color = "#a04000"
        else:
            text = (
                f"Estado: {state_reason(result.reason)} en la época "
                f"{result.epochs}"
            )
            color = "#922b21"
        self.estado_label.setText(text)
        self.estado_label.setStyleSheet(
            "font-weight:bold; font-size:13px; color:#%s;" % color.lstrip("#")
        )

    def _on_training_failed(self, message):
        self._worker = None
        self.estado_label.setText("Estado: error durante el entrenamiento")
        self.estado_label.setStyleSheet("font-weight:bold; color:#922b21;")
        self.refresh_enabled()


def state_reason(reason):
    from core.trainer import MESSAGES

    return MESSAGES.get(reason, reason)