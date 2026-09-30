"""Widgets reutilizables: editor visual de la grilla 5x7, vista en miniatura
de un patrón y botón de ayuda contextual ("?")."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QCursor
from PySide6.QtWidgets import QToolButton, QToolTip, QWidget

from data.dataset import COLS, ROWS

CELL = 40
GRID_MARGIN = 2


class VowelGridEditor(QWidget):
    """Grilla 5x7 dibujable: clic en una celda alterna negro/blanco."""

    grid_changed = Signal()

    def __init__(self, rows=ROWS, cols=COLS, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self._grid = [0] * (rows * cols)
        self.setFixedSize(
            cols * CELL + 2 * GRID_MARGIN,
            rows * CELL + 2 * GRID_MARGIN,
        )
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def grid(self):
        return list(self._grid)

    def set_grid(self, grid):
        flat = list(grid)
        if len(flat) != len(self._grid):
            raise ValueError("La grilla debe tener 35 píxeles.")
        self._grid = flat
        self.update()

    def clear(self):
        self._grid = [0] * len(self._grid)
        self.update()

    def _cell_at(self, pos):
        x = pos.x() - GRID_MARGIN
        y = pos.y() - GRID_MARGIN
        if x < 0 or y < 0:
            return None
        col = x // CELL
        row = y // CELL
        if col >= self.cols or row >= self.rows:
            return None
        return row, col

    def mousePressEvent(self, event):
        hit = self._cell_at(event.position().toPoint())
        if hit is None:
            return
        row, col = hit
        idx = row * self.cols + col
        self._grid[idx] = 1 - self._grid[idx]
        self.update()
        self.grid_changed.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ececec"))
        for i in range(len(self._grid)):
            row = i // self.cols
            col = i % self.cols
            rect_edge = int(GRID_MARGIN + col * CELL)
            rect_top = int(GRID_MARGIN + row * CELL)
            if self._grid[i]:
                painter.fillRect(
                    rect_edge, rect_top, CELL, CELL, QColor("#1d1d1d")
                )
            else:
                painter.fillRect(
                    rect_edge, rect_top, CELL, CELL, QColor("#ffffff")
                )
                painter.setPen(QColor("#c9c9c9"))
                painter.drawRect(rect_edge, rect_top, CELL, CELL)
        painter.setPen(QColor("#1a1a1a"))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class BitmapView(QWidget):
    """Miniatura de un patrón binario (para paneles de predicción)."""

    def __init__(self, scale=13, rows=ROWS, cols=COLS, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.scale = scale
        self._grid = [0] * (rows * cols)
        self.setFixedSize(cols * scale + 2, rows * scale + 2)
        self._bg = QColor("#ffffff")

    def set_grid(self, grid):
        self._grid = list(grid)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg)
        for i in range(len(self._grid)):
            row = i // self.cols
            col = i % self.cols
            x = 1 + col * self.scale
            y = 1 + row * self.scale
            if self._grid[i]:
                painter.fillRect(x, y, self.scale, self.scale, QColor("#1d1d1d"))
            else:
                painter.setPen(QColor("#d7d7d7"))
                painter.drawRect(x, y, self.scale, self.scale)


class HelpButton(QToolButton):
    """Botón "?" que muestra ayuda contextual junto a un concepto técnico."""

    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setText("?")
        self.setFixedSize(22, 22)
        self.setCursor(QCursor(Qt.WhatsThisCursor))
        self.setStyleSheet(
            "QToolButton { background:#1f77b4; color:white; border-radius:11px;"
            " font-weight:bold; font-size:11px; }"
            "QToolButton:hover { background:#2e8cc7; }"
        )
        self._help = text
        self.setToolTip(text)
        self.clicked.connect(self._show_help)

    def _show_help(self):
        QToolTip.showText(self.mapToGlobal(self.rect().center()), self._help, self)