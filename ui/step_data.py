"""PASO 1 DE 4 — DATOS: editor visual de la grilla 5x7, generador de ruido,
carga/exportación TXT y partición 70/30 estratificada con validación."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from data.dataset import (
    CLEAN_GRIDS,
    VOWELS,
    VowelPattern,
    grid_from_rows,
)
from data.io import LoaderError, load_dataset, save_dataset
from data.noise import build_noisy_dataset
from ui.widgets import HelpButton

_HEADER = (
    "PASO 1 DE 4 — DATOS DE LAS VOCALES: revise, pinte o genere patrones. "
    "Cada vocal debe tener al menos 1 patrón para poder entrenar."
)


class StepData(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._selected = None

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

        toolbar = QHBoxLayout()
        self.default_btn = QPushButton(" Restaurar dataset por defecto")
        self.default_btn.setStyleSheet("font-weight:bold;")
        self.default_btn.clicked.connect(self._restore_default)
        toolbar.addWidget(self.default_btn)

        toolbar.addWidget(QLabel("Ruido:"))
        self.noise_spin = QSpinBox()
        self.noise_spin.setRange(1, 50)
        self.noise_spin.setSuffix(" %")
        self.noise_spin.setValue(int(round(10)))
        toolbar.addWidget(self.noise_spin)
        self.noise_spin.setToolTip(
            "Porcentaje de píxeles que se invierten al azar en cada variante."
        )

        toolbar.addWidget(QLabel("Variantes por vocal:"))
        self.variants_spin = QSpinBox()
        self.variants_spin.setRange(2, 10)
        self.variants_spin.setValue(4)
        toolbar.addWidget(self.variants_spin)

        self.gen_btn = QPushButton(" Generar dataset con ruido ")
        self.gen_btn.clicked.connect(self._generate_noise)
        toolbar.addWidget(self.gen_btn)

        self.load_btn = QPushButton("Cargar TXT")
        self.load_btn.clicked.connect(self._load_txt)
        toolbar.addWidget(self.load_btn)
        self.export_btn = QPushButton("Exportar TXT")
        self.export_btn.clicked.connect(self._export_txt)
        toolbar.addWidget(self.export_btn)
        toolbar.addStretch(1)
        content.addLayout(toolbar)

        middle = QHBoxLayout()
        middle.addWidget(self._build_list_box(), 2)
        middle.addWidget(self._build_editor_box(), 3)
        content.addLayout(middle, 1)

        content.addWidget(self._build_split_box())

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("font-size:13px; font-weight:bold;")
        content.addWidget(self.validation_label)

        content.addStretch(1)
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._refresh_list()
        self._refresh_split()
        self._update_validation()

    def _build_list_box(self):
        box = QGroupBox("PATRONES DEL DATASET")
        layout = QVBoxLayout(box)
        self.pattern_list = QListWidget()
        self.pattern_list.setMinimumHeight(300)
        self.pattern_list.currentRowChanged.connect(self._on_select)
        layout.addWidget(self.pattern_list)
        self.count_label = QLabel("")
        layout.addWidget(self.count_label)
        self.btn_delete = QPushButton("Eliminar patrón seleccionado")
        self.btn_delete.clicked.connect(self._delete_selected)
        layout.addWidget(self.btn_delete)
        hint = QLabel(
            "Seleccione un patrón para verlo en la grilla y poder editarlo "
            "pintando celdas con clic."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555; font-size:11px;")
        layout.addWidget(hint)
        return box

    def _build_editor_box(self):
        box = QGroupBox("EDITOR DE LA GRILLA 5x7 (clic = alternar negro/blanco)")
        layout = QHBoxLayout(box)

        from ui.widgets import VowelGridEditor

        self.editor = VowelGridEditor(parent=self)
        layout.addWidget(self.editor)

        side = QVBoxLayout()
        self.label_vowel = QLabel("Vocal: —")
        self.label_vowel.setStyleSheet("font-size:16px; font-weight:bold; color:#1a5276;")
        side.addWidget(self.label_vowel)
        side.addWidget(HelpButton(
            "Cada cuadro es un píxel de la vocal (1 = negro, 0 = blanco). "
            "35 píxeles = 5 columnas x 7 filas = las entradas de la red."
        ))

        self.apply_btn = QPushButton("Aplicar cambios al patrón")
        self.apply_btn.clicked.connect(self._apply_changes)
        side.addWidget(self.apply_btn)

        self.clear_btn = QPushButton("Restaurar vocal base")
        self.clear_btn.setToolTip("Vuelve a la forma original (sin ruido) de "
                                  "esta vocal.")
        self.clear_btn.clicked.connect(self._restore_base)
        side.addWidget(self.clear_btn)

        note = QLabel(
            "Los cambios aplicados sustituyen el patrón seleccionado y "
            "reparten de nuevo los datos en 70% entrenamiento / 30% "
            "validación."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555; font-size:11px;")
        side.addWidget(note)
        side.addStretch(1)
        layout.addLayout(side, 1)
        return box

    def _build_split_box(self):
        box = QGroupBox("PARTICIÓN AUTOMÁTICA 70% ENTRENAMIENTO / 30% VALIDACIÓN")
        layout = QGridLayout(box)
        self.split_title = QLabel("")
        self.split_title.setStyleSheet("font-size:12px; font-weight:bold;")
        layout.addWidget(self.split_title, 0, 0, 1, 3)
        self.split_detail = QLabel("")
        self.split_detail.setWordWrap(True)
        layout.addWidget(self.split_detail, 1, 0, 1, 3)
        note = QLabel(
            "La partición respeta la proporción por vocal (estratificada)."
        )
        note.setStyleSheet("color:#555; font-size:11px;")
        layout.addWidget(note, 2, 0, 1, 3)
        return box

    # ------------------------------------------------------------------
    def _refresh_list(self):
        self.pattern_list.blockSignals(True)
        self.pattern_list.clear()
        for idx, p in enumerate(self.state.patterns):
            item = f"{p.vowel} · patrón {idx + 1}"
            self.pattern_list.addItem(item)
        self.pattern_list.blockSignals(False)
        counts = self.state.label_counts()
        self.count_label.setText(
            "Total: {} patrones | Por vocal: {}".format(
                len(self.state.patterns),
                " · ".join(f"{VOWELS[c]}={counts[c]}" for c in range(5)),
            )
        )
        if self.pattern_list.count():
            self.pattern_list.setCurrentRow(0)
        else:
            self._on_select(-1)

    def _on_select(self, row):
        self._selected = row
        if row < 0 or row >= len(self.state.patterns):
            self.editor.clear()
            self.label_vowel.setText("Vocal: —")
            self.apply_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.btn_delete.setEnabled(False)
            return
        p = self.state.patterns[row]
        self.editor.set_grid(p.grid)
        self.label_vowel.setText(
            f"Vocal: {p.vowel}   (patrón {row + 1})"
        )
        self.apply_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.btn_delete.setEnabled(len(self.state.patterns) > 1)

    def _apply_changes(self):
        if self._selected is None:
            return
        grid = tuple(self.editor.grid())
        if 1 not in grid:
            QMessageBox.warning(
                self, "Patrón vacío",
                "El patrón quedó completamente blanco. Márquelo antes de "
                "aplicarlo.",
            )
            return
        old = self.state.patterns[self._selected]
        self.state.patterns[self._selected] = VowelPattern(grid, old.label)
        self.state.recompute_split()
        self._notify_changed()

    def _restore_base(self):
        if self._selected is None:
            return
        p = self.state.patterns[self._selected]
        clean = grid_from_rows(CLEAN_GRIDS[p.vowel])
        self.state.patterns[self._selected] = VowelPattern(clean, p.label)
        self.state.recompute_split()
        self._notify_changed()

    def _delete_selected(self):
        if self._selected is None or len(self.state.patterns) <= 1:
            return
        del self.state.patterns[self._selected]
        self.state.recompute_split()
        self._notify_changed()

    def _restore_default(self):
        self.state.restore_defaults()
        self.noise_spin.setValue(10)
        self.variants_spin.setValue(4)
        self._notify_changed()

    def _generate_noise(self):
        percent = self.noise_spin.value() / 100.0
        variants = self.variants_spin.value()
        self.state.set_dataset(
            build_noisy_dataset(n_variants=variants, percent=percent,
                                seed=self.state.seed)
        )
        self._notify_changed()

    def _load_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar patrones de vocales", "", "Archivos TXT (*.txt)"
        )
        if not path:
            return
        try:
            patterns = load_dataset(path)
        except LoaderError as exc:
            QMessageBox.critical(self, "Error al cargar", str(exc))
            return
        self.state.set_dataset(patterns)
        self._notify_changed()

    def _export_txt(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar patrones", "vocales_dataset.txt", "Archivos TXT (*.txt)"
        )
        if not path:
            return
        try:
            save_dataset(path, self.state.patterns)
        except Exception as exc:
            QMessageBox.critical(self, "Error al exportar", str(exc))
            return
        QMessageBox.information(self, "Exportado", f"Patrones guardados en:\n{path}")

    def _notify_changed(self):
        self._refresh_list()
        self._refresh_split()
        self._update_validation()
        self.state.data_changed.emit()

    def _refresh_split(self):
        self.state.recompute_split()
        train = len(self.state.train_indices)
        val = len(self.state.val_indices)
        total = train + val
        self.split_title.setText(
            f"{train} de entrenamiento · {val} de validación "
            f"(de {total} patrones)"
        )
        lines = []
        for c in range(5):
            t = sum(1 for i in self.state.train_indices
                    if self.state.patterns[i].label == c)
            v = sum(1 for i in self.state.val_indices
                    if self.state.patterns[i].label == c)
            lines.append(f"{VOWELS[c]}: {t} entrenar / {v} validar")
        self.split_detail.setText("   ".join(lines))

    def _update_validation(self):
        errors = self.state.dataset_errors()
        if not errors:
            self.validation_label.setText(
                "LISTO PARA ENTRENAR: hay al menos un patrón por vocal y la "
                "partición 70/30 está lista. Pase al paso 3 y pulse "
                "ENTRENAR."
            )
            self.validation_label.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#1e8449;"
                "background:#eafaf1; border:1px solid #a9dfbf;"
                "border-radius:6px; padding:8px;"
            )
        else:
            self.validation_label.setText(" · ".join(errors))
            self.validation_label.setStyleSheet(
                "font-size:13px; font-weight:bold; color:#922b21;"
                "background:#fdedec; border:1px solid #f5b7b1;"
                "border-radius:6px; padding:8px;"
            )