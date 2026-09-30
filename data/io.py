"""Carga y exportación de patrones de vocales en formato TXT.

Formato (consistente con laboratorios anteriores):

    # comentario opcional
    A
    01110
    10001
    ...
    10001

    E
    11111
    ...

Cada patrón es: una línea con la etiqueta (A/E/I/O/U) seguida de 7 filas
de 5 caracteres '0'/'1'. Los patrones se separan con líneas vacías.
"""

from pathlib import Path

from data.dataset import (
    COLS,
    ROWS,
    VOWELS,
    VowelPattern,
    grid_from_rows,
    rows_from_grid,
)


class LoaderError(ValueError):
    pass


def _read_lines(path):
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"El archivo no existe: {path}")
    with open(p, "r", encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def load_dataset(path):
    lines = _read_lines(path)
    patterns = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        label_text = line.upper()
        if label_text not in VOWELS:
            raise LoaderError(
                f"Fila {i + 1}: se esperaba una etiqueta (A/E/I/O/U) y se "
                f"encontró '{line}'."
            )
        rows = []
        i += 1
        while i < len(lines) and len(rows) < ROWS:
            candidate = lines[i].strip()
            if not candidate or candidate.startswith("#"):
                i += 1
                continue
            rows.append(candidate)
            i += 1
        if len(rows) != ROWS:
            raise LoaderError(
                f"El patrón '{label_text}' necesita {ROWS} filas y solo se "
                f"encontraron {len(rows)}."
            )
        try:
            grid = grid_from_rows(rows)
        except ValueError as exc:
            raise LoaderError(f"Patrón '{label_text}': {exc}")
        patterns.append(VowelPattern(grid, VOWELS.index(label_text)))
    if not patterns:
        raise LoaderError("El archivo no contiene ningún patrón de vocal.")
    return patterns


def save_dataset(path, patterns):
    blocks = []
    for p in patterns:
        title = VOWELS[p.label]
        body = "\n".join(rows_from_grid(p.grid))
        blocks.append(f"{title}\n{body}")
    content = (
        "# Dataset de vocales 5x7 (1 = negro, 0 = blanco) "
        "| etiqueta + 7 filas de 5 columnas\n\n"
    ) + "\n\n".join(blocks)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(content)
    return p