"""Pruebas de la partición 70/30 estratificada."""

import unittest

from data.dataset import N_CLASSES, VOWELS, stratified_split
from data.noise import build_noisy_dataset


class TestParticion(unittest.TestCase):
    def _counts(self, patterns, indices):
        counts = {c: 0 for c in range(N_CLASSES)}
        for i in indices:
            counts[patterns[i].label] += 1
        return counts

    def test_proporcion_70_30_con_dataset_por_defecto(self):
        dataset = build_noisy_dataset()
        train, val = stratified_split(dataset, seed=42)
        self.assertEqual(len(train) + len(val), len(dataset))
        self.assertEqual(len(train), 15)
        self.assertEqual(len(val), 5)
        self.assertAlmostEqual(len(train) / len(dataset), 0.70, delta=0.35)

    def test_estratificada_por_clase(self):
        dataset = build_noisy_dataset()
        train, val = stratified_split(dataset, seed=42)
        c_train = self._counts(dataset, train)
        c_val = self._counts(dataset, val)
        for c in range(N_CLASSES):
            self.assertGreater(c_train[c], 0)
            self.assertGreater(c_val[c], 0)
        self.assertTrue(all(t >= 2 for t in c_train.values()))

    def test_no_comparte_indices(self):
        dataset = build_noisy_dataset()
        train, val = stratified_split(dataset, seed=42)
        self.assertTrue(set(train).isdisjoint(val))

    def test_determinista_con_semilla(self):
        dataset = build_noisy_dataset()
        a = stratified_split(dataset, seed=42)
        b = stratified_split(dataset, seed=42)
        self.assertEqual(a, b)

    def test_semilla_distinta_altera_la_particion(self):
        dataset = build_noisy_dataset()
        a = stratified_split(dataset, seed=1)
        b = stratified_split(dataset, seed=2)
        self.assertNotEqual(a, b)

    def test_un_patron_por_clase(self):
        probs = [
            (["01110"] * 7, 0),
            (["11111"] * 7, 1),
            (["00100"] * 7, 2),
            (["01110"] * 7, 3),
            (["10001"] * 7, 4),
        ]
        from data.dataset import VowelPattern, grid_from_rows

        dataset = [VowelPattern(grid_from_rows(r), l) for r, l in probs]
        train, val = stratified_split(dataset, seed=1)
        self.assertEqual(len(train), len(dataset) - len(val))
        self.assertEqual(len(val), 5)
        for p in dataset:
            self.assertIn(id(p), [id(dataset[i]) for i in train] +
                          [id(dataset[i]) for i in val])

    def test_velocidad_de_generacion(self):
        dataset = build_noisy_dataset()
        for _ in range(5):
            stratified_split(dataset, seed=42)


if __name__ == "__main__":
    unittest.main()