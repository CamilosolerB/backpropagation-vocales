"""Dataset de vocales: grillas 5x7 (5 columnas x 7 filas = 35 píxeles
binarios, 1 = negro, 0 = blanco) y división estratificada 70/30."""

from dataclasses import dataclass
from random import Random

VOWELS = ["A", "E", "I", "O", "U"]
ROWS = 7
COLS = 5
INPUT_SIZE = ROWS * COLS
N_CLASSES = len(VOWELS)

CLEAN_GRIDS = {
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "I": [
        "01110",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "O": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "U": [
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
}

DEFAULT_VARIANTS = 4
DEFAULT_NOISE = 0.10
DEFAULT_SEED = 42
TRAIN_FRACTION = 0.70


@dataclass
class VowelPattern:
    grid: tuple
    label: int

    @property
    def vowel(self):
        return VOWELS[self.label]


def grid_from_rows(rows):
    """Convierte 7 filas de 5 caracteres '0'/'1' en una tupla de 35 enteros."""
    flat = []
    for row in rows:
        if len(row) != COLS:
            raise ValueError(
                f"Cada fila debe tener {COLS} columnas: se recibió '{row}'."
            )
        for ch in row:
            if ch not in ("0", "1"):
                raise ValueError(
                    f"Solo se permiten '0' y '1': carácter inesperado '{ch}'."
                )
            flat.append(1 if ch == "1" else 0)
    return tuple(flat)


def rows_from_grid(grid):
    """Convierte una tupla de 35 enteros en 7 filas de 5 caracteres."""
    if len(grid) != INPUT_SIZE:
        raise ValueError(f"Se esperan {INPUT_SIZE} píxeles.")
    return ["".join("1" if grid[r * COLS + c] else "0" for c in range(COLS))
            for r in range(ROWS)]


def clean_patterns():
    """Los 5 patrones base sin ruido (uno por vocal)."""
    return [
        VowelPattern(grid_from_rows(CLEAN_GRIDS[v]), idx)
        for idx, v in enumerate(VOWELS)
    ]


def pattern_as_text(p):
    return "\n".join(rows_from_grid(p.grid))


def stratified_split(patterns, train_fraction=TRAIN_FRACTION, seed=DEFAULT_SEED,
                     n_classes=N_CLASSES):
    """División 70/30 estratificada: respeta la proporción por clase.

    Devuelve (train_indices, val_indices). Siempre deja al menos un patrón
    en cada conjunto por clase siempre que sea posible.
    """
    rng = Random(int(seed))
    groups = {c: [] for c in range(n_classes)}
    for idx, p in enumerate(patterns):
        groups[p.label].append(idx)
    train_idx = []
    val_idx = []
    for c in range(n_classes):
        indices = list(groups[c])
        rng.shuffle(indices)
        n = len(indices)
        if n == 0:
            continue
        n_train = max(1, int(round(n * train_fraction)))
        if n_train >= n:
            n_train = n - 1
        n_train = max(0, n_train)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:])
    return train_idx, val_idx