"""PASO 4 DE 4 — RESULTADOS Y PRUEBA: comprobación visual de cualquier
patrón (dibujado o del dataset), matriz de confusión, métricas por clase,
curva final de error y exportación de resultados."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from data.dataset import INPUT_SIZE, N_CLASSES, VOWELS
from ui.widgets import HelpButton, VowelGridEditor
from visualization.confusion import build_confusion_figure
from visualization.plots import build_error_figure
from visualization.plots import build_weights_evolution_figure
from visualization.templates import build_templates_figure

_HEADER = (
    "PASO 4 DE 4 — RESULTADOS Y PRUEBA: dibuje una vocal o elija un patrón "
    "del dataset y la red la reconocerá al instante, con su vocal predicha "
    "en grande y barras de confianza. Nunca tendrá que comparar números."
)


class StepResults(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._known_label = None

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

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size:13px; font-weight:bold;")
        content.addWidget(self.status_label)

        row = QHBoxLayout()
        row.addWidget(self._build_checker_box(), 3)
        right = QVBoxLayout()
        right.addWidget(self._build_confusion_box())
        right.addWidget(self._build_metrics_box())
        row.addLayout(right, 4)
        content.addLayout(row)

        content.addWidget(self._build_curve_box())
        content.addWidget(self._build_weights_box())
        content.addWidget(self._build_template_box())
        content.addWidget(self._build_export_box())

        content.addStretch(1)
        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.editor.grid_changed.connect(self._evaluate)
        self.source_combo.currentIndexChanged.connect(self._on_source_change)
        self.refresh()

    # ------------------------------------------------------------------
    def _build_checker_box(self):
        box = QGroupBox("COMPROBACIÓN: la red predice esta vocal")
        layout = QVBoxLayout(box)

        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Probar con:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("◉  Dibujar un patrón nuevo")
        picker_row.addWidget(self.source_combo, 1)
        picker_row.addWidget(HelpButton(
            "Elija un patrón del dataset para probarlo, o dibuje el suyo "
            "haciendo clic sobre las celdas de la grilla."
        ))
        layout.addLayout(picker_row)

        self.editor = VowelGridEditor(parent=self)
        layout.addWidget(self.editor, 0, Qt.AlignHCenter)

        tools = QHBoxLayout()
        self.noise_spin = QSpinBox()
        self.noise_spin.setRange(1, 50)
        self.noise_spin.setSuffix(" %")
        self.noise_spin.setValue(10)
        tools.addWidget(self.noise_spin)
        self.noise_btn = QPushButton("Añadir ruido")
        self.noise_btn.clicked.connect(self._add_noise)
        tools.addWidget(self.noise_btn)
        self.clear_btn = QPushButton("Limpiar")
        self.clear_btn.clicked.connect(self._clear_draw)
        tools.addWidget(self.clear_btn)

        tools.addWidget(QLabel("Si conozco la vocal real:"))
        self.known_combo = QComboBox()
        self.known_combo.addItem("— no marcada —")
        for v in VOWELS:
            self.known_combo.addItem(v)
        self.known_combo.currentIndexChanged.connect(self._known_changed)
        tools.addWidget(self.known_combo)
        tools.addStretch(1)
        layout.addLayout(tools)

        result_row = QHBoxLayout()
        result_row.addWidget(QLabel("LA RED DICE:"))
        self.big_label = QLabel("?")
        self.big_label.setStyleSheet(
            "font-size:44px; font-weight:bold; color:#555;"
            "padding:6px 16px; border:3px solid #aaa; border-radius:10px;"
        )
        result_row.addWidget(self.big_label)
        self.check_label = QLabel("")
        self.check_label.setStyleSheet("font-size:20px; font-weight:bold;")
        self.check_label.setWordWrap(True)
        result_row.addWidget(self.check_label, 1)
        layout.addLayout(result_row)

        layout.addWidget(QLabel("Confianza por vocal (salida de la red):"))
        self.confidence_rows = []
        conf_grid = QGridLayout()
        for c in range(N_CLASSES):
            name = QLabel(VOWELS[c])
            name.setStyleSheet("font-weight:bold; font-size:13px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFixedHeight(20)
            bar.setStyleSheet(
                "QProgressBar { border:1px solid #bbb; border-radius:5px;"
                " background:#f0f0f0; text-align:center; }"
                "QProgressBar::chunk { background:#1e8449; border-radius:5px; }"
            )
            conf_grid.addWidget(name, c, 0)
            conf_grid.addWidget(bar, c, 1)
            self.confidence_rows.append(bar)
        layout.addLayout(conf_grid)
        return box

    def _build_confusion_box(self):
        box = QGroupBox("MATRIZ DE CONFUSIÓN (validación)")
        layout = QVBoxLayout(box)
        self._cm_placeholder = QWidget()
        self._cm_holder = QVBoxLayout(self._cm_placeholder)
        self._cm_holder.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._cm_placeholder)
        return box

    def _build_metrics_box(self):
        box = QGroupBox("MÉTRICAS POR CLASE (validación)")
        layout = QVBoxLayout(box)
        self.accuracy_label = QLabel("Exactitud global: —")
        self.accuracy_label.setStyleSheet(
            "font-weight:bold; font-size:14px; color:#1a5276;"
        )
        layout.addWidget(self.accuracy_label)
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Vocal", "Precisión", "Sensibilidad (recall)", "F1"]
        )
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.metrics_table)
        return box

    def _build_curve_box(self):
        box = QGroupBox("CURVA FINAL DE ERROR (★ = convergencia)")
        layout = QVBoxLayout(box)
        self._curve_placeholder = QWidget()
        self._curve_holder = QVBoxLayout(self._curve_placeholder)
        self._curve_holder.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._curve_placeholder)
        return box

    def _build_export_box(self):
        box = QGroupBox("EXPORTAR RESULTADOS")
        layout = QHBoxLayout(box)
        self.export_all_btn = QPushButton(" Exportar todo (TXT + CSV + PNG) ")
        self.export_all_btn.setStyleSheet("font-weight:bold;")
        self.export_all_btn.clicked.connect(self._export_all)
        layout.addWidget(self.export_all_btn)
        self.export_png_btn = QPushButton("Solo gráficas PNG")
        self.export_png_btn.clicked.connect(self._export_png)
        layout.addWidget(self.export_png_btn)
        layout.addStretch(1)
        return box

    def _build_weights_box(self):
        box = QGroupBox("PESOS FINALES OBTENIDOS EN EL ENTRENAMIENTO")
        layout = QVBoxLayout(box)
        info = QLabel(
            "Estas son las matrices de pesos y sesgos con las que quedó la red "
            "al terminar el entrenamiento. Verde = peso positivo (estimula la "
            "neurona), rojo = negativo (la inhibe). W1 conecta las 35 entradas "
            "con la capa oculta; la última matriz W conecta la capa oculta con "
            "las 5 salidas (A, E, I, O, U)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size:11px; color:#444;")
        layout.addWidget(info)
        self._w_tabs = QTabWidget()
        self._w_tables = []
        layout.addWidget(self._w_tabs)
        self._bias_table = QTableWidget(0, 3)
        self._bias_table.setHorizontalHeaderLabels(["Capa", "Neurona", "θ"])
        header = self._bias_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self._w_tabs.addTab(self._bias_table, "Sesgos θ")
        return box

    def _build_template_box(self):
        box = QGroupBox(
            "PRUEBA DE APRENDIZAJE: los pesos realmente aprendieron "
            "(resultado esperado)"
        )
        layout = QVBoxLayout(box)
        info = QLabel(
            "Esta figura demuestra que el entrenamiento no dejó pesos "
            "aleatorios. Cada neurona de salida (una por vocal) aprende la "
            "plantilla de píxeles de entrada que la activa. Si la red aprendió, "
            "la plantilla aprendida (arriba, verde = excita / rojo = inhibe) se "
            "parece a la vocal ideal de su columna (abajo) y su semejanza es la "
            "mayor de la fila."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size:11px; color:#444;")
        layout.addWidget(info)

        layout.addWidget(QLabel(
            "A) Plantillas aprendidas por cada salida vs vocal ideal:"
        ))
        self._tpl_placeholder = QWidget()
        self._tpl_holder = QVBoxLayout(self._tpl_placeholder)
        self._tpl_holder.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tpl_placeholder)

        layout.addWidget(QLabel(
            "B) Evolución de la magnitud de los pesos por capa durante el "
            "entrenamiento (cambiaron desde el valor aleatorio inicial):"
        ))
        self._evo_placeholder = QWidget()
        self._evo_holder = QVBoxLayout(self._evo_placeholder)
        self._evo_holder.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._evo_placeholder)
        return box

    # ------------------------------------------------------------------
    def refresh(self, result=None):
        self._populate_source_combo()
        if not self.state.trained():
            self.status_label.setText(
                "Aún no hay resultados: entrene la red en el paso 3."
            )
            self.status_label.setStyleSheet(
                "font-weight:bold; color:#a04000;"
            )
            self.editor.setEnabled(False)
            self.export_png_btn.setEnabled(False)
            self.export_all_btn.setEnabled(False)
            return
        self.status_label.setText(
            self.state.reason_label(self.state.result.reason)
        )
        self.status_label.setStyleSheet(
            "font-weight:bold; color:#1e8449;"
            if self.state.result.converged
            else "font-weight:bold; color:#922b21;"
        )
        self.editor.setEnabled(True)
        self.export_png_btn.setEnabled(True)
        self.export_all_btn.setEnabled(True)
        self._draw_confusion()
        self._draw_metrics()
        self._draw_curve()
        self._draw_weights()
        self._draw_templates()
        self._draw_evolution()
        self._evaluate()

    def trained(self):
        return self.state.trained()

    def _populate_source_combo(self):
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("◉  Dibujar un patrón nuevo")
        for idx, p in enumerate(self.state.patterns):
            self.source_combo.addItem(f"{p.vowel} · patrón {idx + 1}")
        self.source_combo.blockSignals(False)

    def _on_source_change(self, idx):
        if idx <= 0:
            self._known_label = None
            self.known_combo.setCurrentIndex(0)
            self.editor.set_grid([0] * INPUT_SIZE)
            self._evaluate()
            return
        p = self.state.patterns[idx - 1]
        self._known_label = p.label
        self.known_combo.setCurrentIndex(p.label + 1)
        self.editor.set_grid(p.grid)
        self._evaluate()

    def _known_changed(self, idx):
        if idx == 0:
            self._known_label = None
        else:
            self._known_label = idx - 1
        self._evaluate()

    def _add_noise(self):
        import random
        from data.noise import add_noise

        percent = self.noise_spin.value() / 100.0
        rng = random.Random(1)
        noisy = add_noise(self.editor.grid(), percent, rng)
        self.editor.set_grid(noisy)

    def _clear_draw(self):
        self.editor.clear()
        self._evaluate()

    def _evaluate(self):
        if not self.state.trained():
            return
        grid = self.editor.grid()
        probs = self.state.network.predict_proba(grid)
        pred = probs.index(max(probs))
        self.big_label.setText(VOWELS[pred])
        if self._known_label is None:
            self.big_label.setStyleSheet(
                "font-size:44px; font-weight:bold; color:#1a5276;"
                "padding:6px 16px; border:3px solid #1a5276; border-radius:10px;"
            )
            self.check_label.setText("Vocal real no marcada.")
            self.check_label.setStyleSheet("font-size:16px; color:#555;")
        elif pred == self._known_label:
            self.big_label.setStyleSheet(
                "font-size:44px; font-weight:bold; color:#1e8449;"
                "padding:6px 16px; border:3px solid #1e8449; border-radius:10px;"
            )
            self.check_label.setText(
                f"ACIERTO ✓  (la vocal real era {VOWELS[self._known_label]})"
            )
            self.check_label.setStyleSheet(
                "font-size:18px; font-weight:bold; color:#1e8449;"
            )
        else:
            self.big_label.setStyleSheet(
                "font-size:44px; font-weight:bold; color:#922b21;"
                "padding:6px 16px; border:3px solid #922b21; border-radius:10px;"
            )
            self.check_label.setText(
                f"ERROR ✗  (la vocal real era {VOWELS[self._known_label]})"
            )
            self.check_label.setStyleSheet(
                "font-size:18px; font-weight:bold; color:#922b21;"
            )
        for c, bar in enumerate(self.confidence_rows):
            value = max(0.0, min(1.0, probs[c]))
            bar.setValue(int(round(value * 100)))

    # ------------------------------------------------------------------
    def _clear_layout(self, holder):
        while holder.count():
            item = holder.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _draw_confusion(self):
        self._clear_layout(self._cm_holder)
        cm = self.state.val_confusion_matrix()
        if cm is None:
            self._cm_holder.addWidget(QLabel("Sin datos de validación."))
            return
        fig = build_confusion_figure(cm, VOWELS)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(280)
        self._cm_holder.addWidget(canvas)
        canvas.draw()

    def _draw_metrics(self):
        metrics = self.state.val_metrics()
        self.metrics_table.setRowCount(N_CLASSES)
        for c in range(N_CLASSES):
            m = metrics["per_class"][c]
            self.metrics_table.setItem(c, 0, QTableWidgetItem(VOWELS[c]))
            for col, key in enumerate(("precision", "recall", "f1"), start=1):
                self.metrics_table.setItem(
                    c, col, QTableWidgetItem(f"{m[key]:.3f}")
                )
        self.accuracy_label.setText(
            f"Exactitud global: {metrics['accuracy']:.3f}"
        )

    def _draw_curve(self):
        history = self.state.result.history
        converged_at = self.state.result.converged_at
        self._clear_layout(self._curve_holder)
        fig = build_error_figure(
            history, self.state.desired_error, converged_at, figsize=(9.0, 2.9)
        )
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(250)
        self._curve_holder.addWidget(canvas)
        canvas.draw()

    def _weight_item(self, value):
        item = QTableWidgetItem(f"{value:.5f}")
        if value > 1e-9:
            item.setForeground(QColor("#1e8449"))
            item.setBackground(QColor("#eafaf1"))
        elif value < -1e-9:
            item.setForeground(QColor("#922b21"))
            item.setBackground(QColor("#fdedec"))
        else:
            item.setForeground(QColor("#666"))
        return item

    def _draw_weights(self):
        w, b = self.state.network.copy_params()
        n_layers = len(w)
        layer_sizes = self.state.network.layer_sizes

        while len(self._w_tables) < n_layers:
            table = QTableWidget()
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setDefaultSectionSize(64)
            self._w_tables.append(table)
            self._w_tabs.insertTab(
                self._w_tabs.count() - 1, table, "W"
            )

        while len(self._w_tables) > n_layers:
            idx = self._w_tabs.count() - 1
            tab = self._w_tabs.widget(idx)
            self._w_tabs.removeTab(idx)
            tab.deleteLater()
            self._w_tables.pop()

        for l in range(n_layers):
            table = self._w_tables[l]
            n_in = len(w[l])
            n_out = len(w[l][0])
            table.setRowCount(n_in)
            table.setColumnCount(n_out)
            if l == 0:
                vlab = [f"píxel {i}" for i in range(n_in)]
            else:
                vlab = [f"oculta {i}" for i in range(n_in)]
            if l == n_layers - 1:
                hlab = list(VOWELS[:n_out])
            else:
                hlab = [f"oculta {j}" for j in range(n_out)]
            table.setVerticalHeaderLabels(vlab)
            table.setHorizontalHeaderLabels(hlab)
            for i in range(n_in):
                for j in range(n_out):
                    table.setItem(i, j, self._weight_item(w[l][i][j]))
            if l == 0:
                tab_name = "W1 · Entrada→Oculta"
            elif l == n_layers - 1:
                tab_name = f"W{l + 1} · Oculta→Salida"
            else:
                tab_name = f"W{l + 1} · Oculta→Oculta"
            self._w_tabs.setTabText(l, tab_name)

        self._bias_table.setRowCount(sum(len(vec) for vec in b))
        row = 0
        for l, vec in enumerate(b):
            for j, value in enumerate(vec):
                self._bias_table.setItem(row, 0, QTableWidgetItem(f"capa {l + 1}"))
                self._bias_table.setItem(row, 1, QTableWidgetItem(f"neurona {j}"))
                self._bias_table.setItem(row, 2, self._weight_item(value))
                row += 1

    def _draw_templates(self):
        self._clear_layout(self._tpl_holder)
        templates = self.state.network.effective_input_weights()
        fig = build_templates_figure(templates, VOWELS)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(360)
        self._tpl_holder.addWidget(canvas)
        canvas.draw()

    def _draw_evolution(self):
        self._clear_layout(self._evo_holder)
        fig = build_weights_evolution_figure(self.state.result.history)
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(230)
        self._evo_holder.addWidget(canvas)
        canvas.draw()

    # ------------------------------------------------------------------
    def _export_png(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta para guardar las gráficas"
        )
        if not folder:
            return
        self._write_pngs(folder)
        QMessageBox.information(
            self, "Exportado",
            f"Gráficas guardadas en:\n{folder}",
        )

    def _export_all(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta para guardar resultados"
        )
        if not folder:
            return
        self._write_pngs(folder)
        build_export_txt(self.state, folder)
        build_export_csv(self.state, folder)
        QMessageBox.information(
            self, "Exportado",
            f"Resultados guardados en:\n{folder}\n"
            f"  - resultados_backprop.txt\n  - historial_epocas.csv\n"
            f"  - error_vs_epocas.png\n  - matriz_confusion.png\n"
            f"  - arquitectura.png\n  - plantillas_aprendidas.png\n"
            f"  - evolucion_pesos.png",
        )

    def _write_pngs(self, folder):
        from visualization.architecture import save_architecture_png
        from visualization.confusion import save_confusion_png
        from visualization.plots import save_error_png
        from visualization.plots import save_weights_evolution_png
        from visualization.templates import build_templates_png

        import os

        state = self.state
        result = state.result
        history = result.history
        save_error_png(
            history, state.desired_error, result.converged_at,
            os.path.join(folder, "error_vs_epocas.png"),
        )
        cm = state.val_confusion_matrix()
        if cm is not None:
            save_confusion_png(
                cm, VOWELS, os.path.join(folder, "matriz_confusion.png")
            )
        layer_sizes = (
            [state.network.input_size]
            + state.network.hidden_sizes
            + [state.network.output_size]
        )
        w, b = state.network.copy_params()
        save_architecture_png(
            layer_sizes, w, b, os.path.join(folder, "arquitectura.png")
        )
        templates = state.network.effective_input_weights()
        build_templates_png(
            templates, os.path.join(folder, "plantillas_aprendidas.png"),
            VOWELS,
        )
        save_weights_evolution_png(
            history, os.path.join(folder, "evolucion_pesos.png")
        )


def _students_block(state):
    if not state.students:
        return "  - (sin nombres)"
    return "\n".join(f"  - {name}" for name in state.students)


def _cm_block(cm):
    lines = ["", "MATRIZ DE CONFUSIÓN (validación)", "        " + "   ".join(VOWELS)]
    for r, vowel in enumerate(VOWELS):
        lines.append(f"  {vowel}  | " + "   ".join(f"{cm[r][c]:3d}" for c in range(len(cm))))
    return "\n".join(lines)


def build_export_txt(state, folder):
    import os

    result = state.result
    metrics = state.val_metrics()
    layer_sizes = (
        [state.network.input_size] + state.network.hidden_sizes
        + [state.network.output_size]
    )
    w, b = state.network.copy_params()
    lines = [
        "=" * 68,
        "BACKPROPAGATION VOCALES — LABORATORIO #3 DE REDES NEURONALES",
        "UNIVERSIDAD DE CUNDINAMARCA (UDEC)",
        "=" * 68,
        "",
        "ESTUDIANTES",
        _students_block(state),
        "",
        "ARQUITECTURA",
        "  Entradas: 35 (grilla 5x7) -> ocultas: " +
        " -> ".join(str(s) for s in state.hidden_layers) + " -> salidas: 5",
        "  Activación: sigmoide net = Σ(w·x) + θ, y = 1/(1+e^-net)",
        "  alpha (α) = " + str(state.alpha),
        "  beta (β / momento) = " + str(state.beta),
        "  error deseado = " + str(state.desired_error),
        "  máxima de épocas = " + str(state.max_epochs),
        "  tiempo límite = " + str(state.timeout) + " s",
        "  semilla = " + str(state.seed),
        "",
        "RESULTADO",
        "  Estado: " + result.status_message,
        "  Épocas utilizadas: " + str(result.epochs),
        "  ET final: " + (f"{result.final_et:.8f}" if result.final_et is not None else "—"),
        "  Mejor ET alcanzado: " + f"{result.best_et:.8f}",
        "  Aciertos entrenamiento: " + f"{result.train_correct}/{result.train_total}",
        "  Aciertos validación: " + f"{result.val_correct}/{result.val_total}",
        "  Exactitud global (validación): " + f"{metrics['accuracy']:.4f}",
    ]
    lines.append(_cm_block(metrics["cm"]))
    lines.append("")
    lines.append("MÉTRICAS POR CLASE (validación)")
    lines.append("Vocal | Precisión | Sensibilidad | F1")
    for c, vowel in enumerate(VOWELS):
        m = metrics["per_class"][c]
        lines.append(
            f"{vowel} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f}"
        )
    lines.append("")
    lines.append("PESOS FINALES")
    for l in range(len(w)):
        for i in range(len(w[l])):
            for j in range(len(w[l][i])):
                lines.append(f"  W[{l}][{i}][{j}] = {w[l][i][j]:.8f}")
    for l in range(len(b)):
        for j in range(len(b[l])):
            lines.append(f"  θ[{l}][{j}] = {b[l][j]:.8f}")
    path = os.path.join(folder, "resultados_backprop.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def build_export_csv(state, folder):
    import os

    result = state.result
    headers = ["epoca", "et", "mejor_et", "aciertos_entren", "total_entren",
               "aciertos_val", "total_val"]
    path = os.path.join(folder, "historial_epocas.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(headers) + "\n")
        for rec in result.history:
            fh.write(
                f"{rec.epoch},{rec.et:.10f},{rec.best_et:.10f},"
                f"{rec.train_correct},{rec.train_total},"
                f"{rec.val_correct},{rec.val_total}\n"
            )
    return path