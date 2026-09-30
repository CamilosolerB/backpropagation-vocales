"""Matriz de confusión como mapa de calor."""

from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

_HOT_TABLEAU = LinearSegmentedColormap.from_list(
    "vocales_hot", ["#eafaf1", "#2ecc71", "#1e8449"]
)


def build_confusion_figure(cm, class_labels, title="Matriz de confusión "
                            "(conjunto de validación)", figsize=(5.6, 4.6)):
    n = len(cm)
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, cmap=_HOT_TABLEAU, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    fontsize=13, weight="bold",
                    color="#993d00" if cm[i][j] else "#7f8c8d")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Vocal predicha")
    ax.set_ylabel("Vocal real")
    ax.set_title(title, fontsize=12, weight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Nº patrones")
    fig.tight_layout()
    return fig


def save_confusion_png(cm, class_labels, path, title="Matriz de confusión "
                       "(conjunto de validación)"):
    fig = build_confusion_figure(cm, class_labels, title=title)
    fig.savefig(str(path), dpi=150)
    fig.clf()
    return path