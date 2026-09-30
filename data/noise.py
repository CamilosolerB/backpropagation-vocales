"""Generador de ruido: crea variantes de cada vocal invirtiendo píxeles
al azar (1 <-> 0) hasta completar el dataset (5 vocales x 4 variantes)."""

from random import Random

from data.dataset import (
    DEFAULT_NOISE,
    DEFAULT_VARIANTS,
    DEFAULT_SEED,
    VowelPattern,
    clean_patterns,
    grid_from_rows,
)


def flip_pixels(grid, count, rng):
    """Invierte `count` píxeles distintos al azar sobre una copia."""
    if count <= 0:
        return tuple(grid)
    if count > len(grid):
        count = len(grid)
    s = list(grid)
    for i in rng.sample(range(len(grid)), count):
        s[i] = 1 - s[i]
    return tuple(s)


def noise_count(percent, size=35):
    """Número de píxeles a invertir para un porcentaje dado (mínimo 1)."""
    if percent <= 0:
        return 0
    return max(1, int(round(percent * size)))


def add_noise(grid, percent, rng):
    return flip_pixels(grid, noise_count(percent, len(grid)), rng)


def generate_variants(clean_grid, n_variants, percent, rng):
    """Variantes de una vocal: la original limpia + (n_variants - 1) con
    ruido. Total = n_variants patrones por vocal."""
    variants = [tuple(clean_grid)]
    for _ in range(n_variants - 1):
        variants.append(add_noise(clean_grid, percent, rng))
    return variants


def build_noisy_dataset(n_variants=DEFAULT_VARIANTS, percent=DEFAULT_NOISE,
                        seed=DEFAULT_SEED):
    """Dataset por defecto de 20 imágenes: 5 vocales x (1 limpia + n-1
    con ruido). Determinista gracias a la semilla fija."""
    rng = Random(int(seed))
    patterns = []
    for label, clean in enumerate(clean_patterns()):
        for grid in generate_variants(clean.grid, n_variants, percent, rng):
            patterns.append(VowelPattern(grid, label))
    return patterns


def noisy_copy(dataset, percent=DEFAULT_NOISE, seed=DEFAULT_SEED):
    """Re-aplica ruido a cada patrón con su misma etiqueta."""
    rng = Random(int(seed))
    out = []
    for p in dataset:
        out.append(VowelPattern(add_noise(p.grid, percent, rng), p.label))
    return out