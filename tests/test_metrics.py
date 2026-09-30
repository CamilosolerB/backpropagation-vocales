"""Pruebas de métricas: matriz de confusión, exactitud, precisión,
sensibilidad y F1 (incluyendo divisiones por cero)."""

import unittest

from core.metrics import (
    accuracy,
    build_confusion_matrix,
    per_class_metrics,
)


class TestMetricas(unittest.TestCase):
    def test_matriz_de_confusion_conteos(self):
        y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        y_pred = [0, 0, 1, 2, 2, 2, 3, 3, 4, 4]
        cm = build_confusion_matrix(y_true, y_pred, 5)
        self.assertEqual(cm[0][0], 2)
        self.assertEqual(cm[0][1], 0)
        self.assertEqual(cm[1][1], 1)
        self.assertEqual(cm[1][2], 1)
        self.assertEqual(cm[2][2], 2)
        self.assertEqual(cm[4][4], 2)

    def test_exactitud_perfecta(self):
        cm = [[1, 0], [0, 1]]
        self.assertEqual(accuracy(cm), 1.0)

    def test_exactitud_mitad(self):
        cm = [[1, 1], [1, 1]]
        self.assertEqual(accuracy(cm), 0.5)

    def test_metricas_de_clase_perfecta(self):
        cm = [[2, 0], [0, 3]]
        m = per_class_metrics(cm)
        self.assertAlmostEqual(m[0]["precision"], 1.0)
        self.assertAlmostEqual(m[0]["recall"], 1.0)
        self.assertAlmostEqual(m[0]["f1"], 1.0)

    def test_precision_recall_diferenciados(self):
        # clase 0: 2 verdaderos positivos, 1 falso positivo (fila 1),
        #          4 falsos negativos (columna 1)
        cm = [[2, 4], [1, 1]]
        m = per_class_metrics(cm)
        self.assertAlmostEqual(m[0]["precision"], 2 / 3)
        self.assertAlmostEqual(m[0]["recall"], 1 / 3)
        expected_f1 = 2 * (2 / 3) * (1 / 3) / (2 / 3 + 1 / 3)
        self.assertAlmostEqual(m[0]["f1"], expected_f1)

    def test_division_por_cero_en_clase_vacia(self):
        cm = [[0, 0], [0, 2]]
        m = per_class_metrics(cm)
        self.assertEqual(m[0]["precision"], 0.0)
        self.assertEqual(m[0]["recall"], 0.0)
        self.assertEqual(m[0]["f1"], 0.0)

    def test_matriz_vacia_exactitud_cero(self):
        self.assertEqual(accuracy([[0, 0], [0, 0]]), 0.0)

    def test_f1_con_un_solo_valor_no_cero(self):
        cm = [[0, 3], [0, 1]]
        m = per_class_metrics(cm)
        self.assertEqual(m[0]["precision"], 0.0)
        self.assertEqual(m[0]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()