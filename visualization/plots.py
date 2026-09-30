"""Gráficas: curva de error (ET) vs épocas con marcador de convergencia."""

from matplotlib.figure import Figure

COLOR_CURVE = "#1f77b4"
COLOR_TARGET = "#d62728"
COLOR_CONV = "#2ca02c"


def build_error_figure(history, desired_error=None, converged_at=None,
                       figsize=(7.8, 4.4)):
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if history:
        epochs = [r.epoch for r in history]
        et = [r.et for r in history]
        ax.plot(epochs, et, marker=".", markersize=3, linewidth=1.4,
                color=COLOR_CURVE, label="ET (error total por época)")
    if desired_error is not None:
        ax.axhline(desired_error, color=COLOR_TARGET, linestyle="--",
                   linewidth=1.5, label=f"Error deseado = {desired_error}")
    if converged_at is not None:
        ax.plot(
            converged_at,
            next((r.et for r in history if r.epoch == converged_at),
                 float("nan")),
            marker="*", markersize=16, color=COLOR_CONV, zorder=5,
            label="Convergencia",
        )
    ax.set_xlabel("Época")
    ax.set_ylabel("Error total ET")
    ax.set_title("Error total vs Épocas", fontsize=13, weight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    if desired_error is not None or converged_at is not None:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def save_error_png(history, desired_error, converged_at, path):
    fig = build_error_figure(history, desired_error, converged_at)
    fig.savefig(str(path), dpi=150)
    fig.clf()
    return path


def build_weights_evolution_figure(history, figsize=(9.0, 2.9)):
    """Evolución de la magnitud media de los pesos por capa durante el
    entrenamiento (solo épocas muestreadas). Muestra que los pesos partieron
    de valores aleatorios y cambiaron al aprender.
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sampled = [(r.epoch, r.weights) for r in history if r.weights is not None]
    if not sampled:
        ax.text(0.5, 0.5, "Sin evolución de pesos registrada.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
    epochs = [p[0] for p in sampled]
    layer_norms = []
    for _, weights in sampled:
        norms = []
        for layer in weights:
            flat = [v for row in layer for v in row]
            norms.append(sum(abs(v) for v in flat) / len(flat))
        layer_norms.append(norms)
    total_layers = len(layer_norms[0])
    for l in range(total_layers):
        if l == 0:
            lab = "Entrada→Oculta 1"
        elif l == total_layers - 1:
            lab = f"Oculta {l}→Salida"
        else:
            lab = f"Oculta {l}→Oculta {l + 1}"
        ax.plot(epochs, [row[l] for row in layer_norms], marker=".",
                markersize=3, linewidth=1.4, label=lab)
    ax.set_xlabel("Época")
    ax.set_ylabel("Media de |peso|")
    ax.set_title("Evolución de los pesos durante el entrenamiento",
                 fontsize=13, weight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def save_weights_evolution_png(history, path):
    fig = build_weights_evolution_figure(history)
    fig.savefig(str(path), dpi=150)
    fig.clf()
    return path