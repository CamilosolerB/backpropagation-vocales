"""Visualización de la "prueba de aprendizaje": las plantillas efectivas
que la red aprende por vocal (peso efectivo de cada píxel de entrada sobre
cada neurona de salida) comparadas con la vocal ideal.

Fila superior: plantilla aprendida por cada neurona de salida (rojo =
inhibe, verde = excita) como mapa de calor 5x7.
Fila inferior: vocal ideal (1 = negro) de la misma columna.
Si la red aprendió, cada plantilla se parece a su vocal ideal, y la
semejanza (coseno) lo confirma numéricamente.
"""

from matplotlib.figure import Figure

from data.dataset import CLEAN_GRIDS, ROWS, COLS, VOWELS, grid_from_rows

CMAP = "RdYlGn"


def _to_matrix(vector):
    return [[vector[r * COLS + c] for c in range(COLS)] for r in range(ROWS)]


def cosine_sim(a, b):
    """Semejanza por coseno entre dos vectores de igual longitud."""
    dot = sum(x * y for x, y in zip(a, b))
    na = (sum(x * x for x in a)) ** 0.5
    nb = (sum(y * y for y in b)) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _ideal_grids(class_labels):
    return {v: grid_from_rows(CLEAN_GRIDS[v]) for v in class_labels}


def build_templates_figure(templates, class_labels=None,
                           figsize=(13.2, 5.0)):
    """Devuelve una figura con plantillas aprendidas vs vocales ideales.

    ``templates``: lista con un vector por clase (peso efectivo de cada
    entrada hacia cada salida, de ``network.effective_input_weights``).
    """
    if class_labels is None:
        class_labels = list(VOWELS)
    n = len(class_labels)
    ideals = _ideal_grids(class_labels)

    fig = Figure(figsize=figsize, tight_layout=True)
    vmax = max(
        max(abs(x) for x in row)
        for tem in templates
        for row in _to_matrix(tem)
    ) or 1.0

    for k, label in enumerate(class_labels):
        ax = fig.add_subplot(2, n, k + 1)
        ax.imshow(_to_matrix(templates[k]), cmap=CMAP,
                  vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        sim = cosine_sim(templates[k], ideals[label])
        ax.set_title(f"{label}  ·  semejanza {sim:.2f}", fontsize=11,
                     fontweight="bold")

        axb = fig.add_subplot(2, n, n + k + 1)
        grid = ideals[label]
        inverted = [1 - v for v in grid]
        axb.imshow(_to_matrix(inverted), cmap="gray", vmin=0, vmax=1)
        axb.set_xticks([])
        axb.set_yticks([])

    fig.suptitle(
        "PRUEBA DE APRENDIZAJE — plantilla aprendida por la red (arriba) "
        "vs vocal ideal (abajo). Si aprendió, cada plantilla se parece a "
        "su vocal y la semejanza es la mayor de la fila.",
        fontsize=12, fontweight="bold",
    )
    return fig


def build_templates_png(templates, path, class_labels=None):
    build_templates_figure(templates, class_labels).savefig(path)
    return path