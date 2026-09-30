"""PASO 2 DE 4 — ARQUITECTURA Y PARÁMETROS: capas ocultas, tasa de
aprendizaje (α), momento (β), error deseado, épocas, tiempo y presets.
Todo valor inválido se impide con controles de rango y mensajes claros."""

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.state import HELP
from ui.widgets import HelpButton

PRESETS = {
    "Rápido": {
        "alpha": 0.5, "beta": 0.5, "desired_error": 0.05,
        "max_epochs": 1500, "timeout": 30.0,
    },
    "Estándar": {
        "alpha": 0.3, "beta": 0.5, "desired_error": 0.01,
        "max_epochs": 5000, "timeout": 60.0,
    },
    "Preciso": {
        "alpha": 0.15, "beta": 0.5, "desired_error": 0.001,
        "max_epochs": 20000, "timeout": 120.0,
    },
}

_HEADER = (
    "PASO 2 DE 4 — ARQUITECTURA Y PARÁMETROS: elija cuántas capas y "
    "neuronas ocultas usar y con qué parámetros entrenar. Con un clic en "
    "un preset puede dejar todo listo."
)


class StepArchitecture(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        content = QVBoxLayout(container)
        content.setContentsMargins(14, 12, 14, 14)
        content.setSpacing(10)

        header = QLabel(_HEADER)
        header.setWordWrap(True)
        header.setStyleSheet(
            "background:#eaf2fa; border:1px solid #c5d8ec; border-radius:6px;"
            "padding:8px; font-size:12px; font-weight:bold; color:#1a5276;"
        )
        content.addWidget(header)

        content.addWidget(self._build_presets_box())
        content.addWidget(self._build_layers_box())
        content.addWidget(self._build_params_box())

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet(
            "font-size:13px; font-weight:bold; color:#1e8449;"
            "background:#eafaf1; border:1px solid #a9dfbf;"
            "border-radius:6px; padding:8px;"
        )
        content.addWidget(self.validation_label)

        restore = QPushButton(" Restaurar valores por defecto ")
        restore.setStyleSheet("font-weight:bold;")
        restore.clicked.connect(self._restore_defaults)
        content.addWidget(restore)
        content.addStretch(1)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._load_state_into_widgets()
        self._validate()

    def _build_presets_box(self):
        box = QGroupBox("PRESETS DE CONFIGURACIÓN")
        layout = QHBoxLayout(box)
        layout.addWidget(QLabel("Elija un preset y pulse ENTRENAR: "))
        for name in PRESETS:
            btn = QPushButton(f" {name} ")
            btn.setStyleSheet("font-weight:bold;")
            btn.clicked.connect(lambda _=False, n=name: self._apply_preset(n))
            layout.addWidget(btn)
        layout.addStretch(1)
        return box

    def _build_layers_box(self):
        box = QGroupBox("CAPAS OCULTAS (entre las 35 entradas y las 5 salidas)")
        layout = QVBoxLayout(box)
        self.layers_table = QTableWidget(1, 1)
        self.layers_table.setHorizontalHeaderLabels(["Neuronas en la capa"])
        self.layers_table.setMaximumHeight(140)
        header = self.layers_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.layers_table.itemChanged.connect(self._validate)
        layout.addWidget(self.layers_table)
        buttons = QHBoxLayout()
        self.add_layer_btn = QPushButton("+ Añadir capa oculta")
        self.add_layer_btn.clicked.connect(self._add_layer)
        self.remove_layer_btn = QPushButton("− Quitar última capa")
        self.remove_layer_btn.clicked.connect(self._remove_layer)
        buttons.addWidget(self.add_layer_btn)
        buttons.addWidget(self.remove_layer_btn)
        buttons.addWidget(HelpButton(
            "La red empieza con 35 entradas (píxeles de la vocal). Cada capa "
            "oculta transforma las activaciones de la capa anterior. "
            "Más neuronas = más capacidad, pero también más lentitud."
        ))
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return box

    def _build_params_box(self):
        box = QGroupBox("PARÁMETROS DE ENTRENAMIENTO")
        form = QFormLayout(box)

        row = QHBoxLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 5.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(3)
        row.addWidget(self.alpha_spin, 1)
        row.addWidget(HelpButton(HELP["alpha"]))
        form.addRow("Alfa (α) — tasa de aprendizaje:", row)

        row = QHBoxLayout()
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(0.0, 0.99)
        self.beta_spin.setSingleStep(0.05)
        self.beta_spin.setDecimals(2)
        row.addWidget(self.beta_spin, 1)
        row.addWidget(HelpButton(HELP["beta"]))
        form.addRow("Beta (β) — momento:", row)

        row = QHBoxLayout()
        self.error_spin = QDoubleSpinBox()
        self.error_spin.setRange(0.00001, 1.0)
        self.error_spin.setDecimals(5)
        self.error_spin.setSingleStep(0.005)
        row.addWidget(self.error_spin, 1)
        row.addWidget(HelpButton(HELP["et"]))
        form.addRow("Error deseado ET (parada por convergencia):", row)

        row = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 200000)
        self.epochs_spin.setSingleStep(100)
        row.addWidget(self.epochs_spin, 1)
        row.addWidget(HelpButton(HELP["epoch"]))
        form.addRow("Máximo de épocas:", row)

        row = QHBoxLayout()
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.5, 3600.0)
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setSingleStep(5.0)
        row.addWidget(self.timeout_spin, 1)
        row.addWidget(HelpButton(
            "Tiempo límite: si el entrenamiento supera estos segundos, se "
            "detiene solo para que la aplicación nunca quede congelada."
        ))
        form.addRow("Tiempo límite (segundos):", row)

        row = QHBoxLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(42)
        row.addWidget(self.seed_spin, 1)
        row.addWidget(HelpButton(
            "Semilla aleatoria: fija la inicialización de los pesos para que "
            "el entrenamiento sea reproducible."
        ))
        form.addRow("Semilla aleatoria:", row)

        row = QHBoxLayout()
        self.ui_spin = QSpinBox()
        self.ui_spin.setRange(1, 100)
        self.ui_spin.setValue(10)
        row.addWidget(self.ui_spin, 1)
        row.addWidget(HelpButton(
            "Cada cuántas épocas se actualizan las gráficas y el panel de "
            "predicciones durante el entrenamiento en modo continuo."
        ))
        form.addRow("Actualizar pantalla cada N épocas:", row)

        self.alpha_spin.valueChanged.connect(self._validate)
        self.beta_spin.valueChanged.connect(self._validate)
        self.error_spin.valueChanged.connect(self._validate)
        self.epochs_spin.valueChanged.connect(self._validate)
        self.timeout_spin.valueChanged.connect(self._validate)
        self.seed_spin.valueChanged.connect(self._validate)
        self.ui_spin.valueChanged.connect(self._validate)

        return box

    # ------------------------------------------------------------------
    @property
    def _hidden_layers(self):
        layers = []
        for r in range(self.layers_table.rowCount()):
            item = self.layers_table.item(r, 0)
            if item is None or not item.text().strip():
                layers.append(None)
                continue
            try:
                layers.append(int(item.text().strip()))
            except ValueError:
                layers.append(None)
        return layers

    def _apply_preset(self, name):
        p = PRESETS[name]
        for widget in (self.alpha_spin, self.beta_spin, self.error_spin,
                       self.epochs_spin, self.timeout_spin):
            widget.blockSignals(True)
        self.alpha_spin.setValue(p["alpha"])
        self.beta_spin.setValue(p["beta"])
        self.error_spin.setValue(p["desired_error"])
        self.epochs_spin.setValue(p["max_epochs"])
        self.timeout_spin.setValue(p["timeout"])
        for widget in (self.alpha_spin, self.beta_spin, self.error_spin,
                       self.epochs_spin, self.timeout_spin):
            widget.blockSignals(False)
        self._validate()

    def _add_layer(self):
        r = self.layers_table.rowCount()
        self.layers_table.setRowCount(r + 1)
        self.layers_table.setItem(r, 0, QTableWidgetItem("20"))
        self._validate()

    def _remove_layer(self):
        r = self.layers_table.rowCount()
        if r > 1:
            self.layers_table.setRowCount(r - 1)
        self._validate()

    def _restore_defaults(self):
        self.state.restore_defaults()
        self._load_state_into_widgets()
        self._validate()

    def _load_state_into_widgets(self):
        widgets = (self.alpha_spin, self.beta_spin, self.error_spin,
                   self.epochs_spin, self.timeout_spin, self.seed_spin,
                   self.ui_spin, self.layers_table)
        for widget in widgets:
            widget.blockSignals(True)
        self.alpha_spin.setValue(self.state.alpha)
        self.beta_spin.setValue(self.state.beta)
        self.error_spin.setValue(self.state.desired_error)
        self.epochs_spin.setValue(self.state.max_epochs)
        self.timeout_spin.setValue(self.state.timeout)
        self.seed_spin.setValue(self.state.seed)
        self.ui_spin.setValue(self.state.ui_every)
        self.layers_table.setRowCount(len(self.state.hidden_layers))
        for r, size in enumerate(self.state.hidden_layers):
            self.layers_table.setItem(r, 0, QTableWidgetItem(str(size)))
        for widget in widgets:
            widget.blockSignals(False)

    def _collect_state(self):
        layers = self._hidden_layers
        if all(x is not None for x in layers) and layers:
            self.state.hidden_layers = list(layers)
        self.state.alpha = self.alpha_spin.value()
        self.state.beta = self.beta_spin.value()
        self.state.desired_error = self.error_spin.value()
        self.state.max_epochs = self.epochs_spin.value()
        self.state.timeout = self.timeout_spin.value()
        self.state.seed = self.seed_spin.value()
        self.state.ui_every = self.ui_spin.value()

    def _validate(self):
        self._collect_state()
        from core.validator import validate_config

        errors = validate_config(
            self.state.alpha, self.state.beta, self.state.desired_error,
            self.state.max_epochs, self.state.timeout,
            self.state.hidden_layers,
        )
        if errors:
            self.validation_label.setText(" · ".join(errors))
            self.validation_label.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#922b21;"
                "background:#fdedec; border:1px solid #f5b7b1;"
                "border-radius:6px; padding:8px;"
            )
        else:
            self.validation_label.setText(
                "Configuración válida. " + self.state.summary_line()
            )
            self.validation_label.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#1e8449;"
                "background:#eafaf1; border:1px solid #a9dfbf;"
                "border-radius:6px; padding:8px;"
            )
        self.state.config_changed.emit()
        return not errors