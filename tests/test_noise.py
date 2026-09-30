"""Pruebas del generador de ruido y del dataset por defecto."""

import unittest
from random import Random

from data.dataset import INPUT_SIZE, VOWELS, grid_from_rows
from data.noise import (
    add_noise,
    build_noisy_dataset,
    flip_pixels,
    generate_variants,
    noise_count,
)
from data.io import LoaderError, load_dataset, save_dataset


class TestRuido(unittest.TestCase):
    def test_cantidad_de_pixeles_invertidos(self):
        rng = Random(99)
        grid = tuple([0] * INPUT_SIZE)
        count = noise_count(0.10, INPUT_SIZE)
        flipped = flip_pixels(grid, count, rng)
        total = sum(1 for a, b in zip(grid, flipped) if a != b)
        self.assertEqual(total, count)

    def test_ruido_mantiene_binario(self):
        rng = Random(5)
        noisy = add_noise([1] * INPUT_SIZE, 0.10, rng)
        self.assertTrue(all(v in (0, 1) for v in noisy))

    def test_pixeles_invertidos_son_distintos(self):
        rng = Random(5)
        result = add_noise([0] * INPUT_SIZE, 0.5, rng)
        self.assertEqual(sum(result), min(noise_count(0.5, INPUT_SIZE), INPUT_SIZE))

    def test_porcentaje_cero_no_cambia_nada(self):
        rng = Random(5)
        grid = [0, 1, 0, 1] * 9
        self.assertEqual(add_noise(grid, 0.0, rng), tuple(grid))

    def test_semilla_determinista(self):
        a = build_noisy_dataset(seed=42)
        b = build_noisy_dataset(seed=42)
        self.assertEqual(
            [p.grid for p in a], [p.grid for p in b]
        )

    def test_semilla_distinta_cambia_el_dataset(self):
        a = build_noisy_dataset(seed=1)
        b = build_noisy_dataset(seed=2)
        self.assertNotEqual(
            [p.grid for p in a], [p.grid for p in b]
        )

    def test_generador_alcanza_20_patrones(self):
        dataset = build_noisy_dataset()
        self.assertEqual(len(dataset), 20)
        counts = {}
        for p in dataset:
            counts[p.label] = counts.get(p.label, 0) + 1
        self.assertEqual(set(counts), set(range(5)))
        self.assertTrue(all(v == 4 for v in counts.values()))

    def test_variantes_incluyen_original(self):
        rng = Random(3)
        variants = generate_variants(grid_from_rows(["01110"] * 7), 4, 0.10, rng)
        self.assertEqual(len(variants), 4)
        self.assertEqual(variants[0], tuple(grid_from_rows(["01110"] * 7)))


class TestArchivoTXT(unittest.TestCase):
    def _roundtrip(self):
        dataset = build_noisy_dataset()
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vocales.txt")
            save_dataset(path, dataset)
            loaded = load_dataset(path)
            self.assertEqual(len(loaded), len(dataset))
            for orig, new in zip(dataset, loaded):
                self.assertEqual(orig.grid, new.grid)
                self.assertEqual(orig.label, new.label)

    def test_carga_y_exportacion_redondeo(self):
        self._roundtrip()

    def test_archivo_invalido_lanza_error(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "malo.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("A\n010\n")
            with self.assertRaises(LoaderError):
                load_dataset(path)

    def test_etiqueta_desconocida_lanza_error(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "malo.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Z\n01110\n10001\n10001\n10001\n10001\n10001\n10001\n")
            with self.assertRaises(LoaderError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()