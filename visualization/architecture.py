"""Diagrama de la arquitectura de la red: capa de entrada -> capas ocultas
-> capa de salida, con peso reflejado en color (verde = positivo, rojo =
negativo) y grosor de línea (proporcional a su magnitud)."""

from matplotlib.figure import Figure
from matplotlib.patches import Circle

COLOR_POS = "#1e8449"
COLOR_NEG = "#cb4335"
COLOR_NEURON = "#e67e22"
COLOR_INPUT = "#2471a3"
COLOR_OUTPUT = "#7d3c98"
COLOR_BG = "#fdfdfd"


def _centers(n, top, bottom):
    if n <= 1:
        return [(top + bottom) / 2.0]
    return [top - i * (top - bottom) / (n - 1) for i in range(n)]


def build_architecture_figure(layer_sizes, weights, biases, title=None,
                              figsize=(9.5, 5.8)):
    """Dibuja la red con una posición por capa y colores por peso."""
    layer_sizes = list(layer_sizes)
    n_layers = len(layer_sizes)
    fig = Figure(figsize=figsize, facecolor=COLOR_BG)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        title or "ARQUITECTURA DE LA RED (entrada - ocultas - salida)",
        fontsize=13,
        weight="bold",
        pad=10,
    )

    margin = 0.055
    usable = 0.80
    xs = [
        margin + i * usable / (n_layers - 1)
        for i in range(n_layers)
    ]

    max_w = 0.0
    for mat in weights:
        for row in mat:
            for w in row:
                max_w = max(max_w, abs(w))
    if max_w <= 0:
        max_w = 1.0

    positions = []
    for li, n in enumerate(layer_sizes):
        top = 0.90 if n > 1 else 0.5
        bottom = 0.10 if n > 1 else 0.5
        ys = _centers(n, top, bottom)
        positions.append([(xs[li], y) for y in ys])

    for li in range(n_layers - 1):
        color_style = "input-to-hidden" if li == 0 else "hidden-to-out"
        radius_b = min(0.02, max(0.008, 0.05 / layer_sizes[li + 1] ** 0.5))
        for i, (x1, y1) in enumerate(positions[li]):
            for j, (x2, y2) in enumerate(positions[li + 1]):
                w = weights[li][i][j]
                color = COLOR_POS if w >= 0 else COLOR_NEG
                lw = 0.4 + 2.6 * (abs(w) / max_w)
                ax.plot([x1, x2 - radius_b], [y1, y2], color=color,
                        lw=lw, alpha=0.75, solid_capstyle="round")

    for li, n in enumerate(layer_sizes):
        color = COLOR_INPUT if li == 0 else (
            COLOR_OUTPUT if li == n_layers - 1 else COLOR_NEURON
        )
        radius = 0.014 if li == 0 else 0.026
        for x, y in positions[li]:
            ax.add_patch(Circle(
                (x, y), radius, facecolor="#ffffff",
                edgecolor=color, lw=2.0, zorder=4
            ))
        label_y = 0.985 if n > 1 else 0.5
        ax.text(
            xs[li], label_y, f"{n} neuronas" if li > 0 else f"{n} entradas",
            ha="center", va="top", fontsize=11, weight="bold", color=color,
        )

    ax.text(
        0.01, 0.01, "verde = peso positivo · rojo = peso negativo · grosor = magnitud",
        transform=ax.transAxes, fontsize=9, color="#555555",
    )
    fig.tight_layout()
    return fig


def save_architecture_png(layer_sizes, weights, biases, path, title=None):
    fig = build_architecture_figure(layer_sizes, weights, biases, title=title)
    fig.savefig(str(path), dpi=150, facecolor=COLOR_BG)
    fig.clf()
    return path