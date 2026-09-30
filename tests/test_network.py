"""Pruebas de la matemática de la red: forward pass, deltas de salida y de
capas ocultas, actualización de pesos con y sin momento, y argmax."""

import unittest

from core.network import BackpropagationNetwork


class TestForwardPass(unittest.TestCase):
    def test_dimensiones_de_salida(self):
        net = BackpropagationNetwork(35, [20], 5, seed=42)
        out = net.predict([0.0] * 35)
        self.assertEqual(len(out), 5)
        for value in out:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_dos_capas_ocultas(self):
        net = BackpropagationNetwork(35, [24, 12], 5, seed=42)
        acts = net.forward([1.0] * 35)
        self.assertEqual(len(acts), 3)
        self.assertEqual(len(acts[0]), 24)
        self.assertEqual(len(acts[1]), 12)
        self.assertEqual(len(acts[2]), 5)

    def test_sigmoide_valores_extremos(self):
        self.assertAlmostEqual(BackpropagationNetwork.sigmoid(1000), 1.0, places=9)
        self.assertAlmostEqual(BackpropagationNetwork.sigmoid(-1000), 0.0, places=9)
        self.assertAlmostEqual(BackpropagationNetwork.sigmoid(0), 0.5)

    def test_mismo_peso_reproducibilidad_por_semilla(self):
        a = BackpropagationNetwork(4, [3], 2, seed=7)
        b = BackpropagationNetwork(4, [3], 2, seed=7)
        wa, _ = a.copy_params()
        wb, _ = b.copy_params()
        self.assertEqual(wa, wb)


class TestDeltas(unittest.TestCase):
    def test_delta_salida_forma(self):
        net = BackpropagationNetwork(4, [3], 2, seed=1)
        # salida controlada de forma manual
        outputs = [0.8, 0.2]
        targets = [1.0, 0.0]
        deltas = net.delta_outputs(outputs, targets)
        self.assertAlmostEqual(deltas[0], (1.0 - 0.8) * 0.8 * 0.2)
        self.assertAlmostEqual(deltas[1], (0.0 - 0.2) * 0.2 * 0.8)

    def test_delta_salida_cero_cuando_y_igual_d(self):
        net = BackpropagationNetwork(4, [3], 2, seed=1)
        self.assertAlmostEqual(net.delta_outputs([1.0, 1.0], [1.0, 1.0])[0], 0.0)

    def test_backprop_multiplicadero_mantiene_dimensiones(self):
        net = BackpropagationNetwork(5, [4, 3], 2, seed=3)
        activations = net.forward([.5, .3, .9, .1, .7])
        deltas = [None] * net.num_layers
        deltas[-1] = net.delta_outputs(activations[-1], [1.0, 0.0])
        for l in range(net.num_layers - 2, -1, -1):
            y = activations[l]
            d_next = deltas[l + 1]
            w_next = net.weights[l + 1]
            dh = [0.0] * len(y)
            for i in range(len(y)):
                acc = sum(d_next[k] * w_next[i][k] for k in range(len(d_next)))
                dh[i] = y[i] * (1 - y[i]) * acc
            deltas[l] = dh
        self.assertEqual(len(deltas[0]), 4)
        self.assertEqual(len(deltas[1]), 3)


class TestUpdatePesos(unittest.TestCase):
    def test_update_sin_momento_cambia_pesos(self):
        net = BackpropagationNetwork(3, [2], 1, seed=5)
        antes, _ = net.copy_params()
        ep = net.train_on_pattern([1.0, 0.0, 1.0], [1.0], alpha=0.5, beta=0.0)
        despues, _ = net.copy_params()
        self.assertNotEqual(antes, despues)
        self.assertGreater(ep, 0.0)

    def test_momento_cero_equivale_a_beta_0(self):
        net1 = BackpropagationNetwork(3, [2], 1, seed=11)
        net2 = BackpropagationNetwork(3, [2], 1, seed=11)
        for (x, d) in [([1.0, 0.0, 1.0], [1.0]), ([0.0, 1.0, 1.0], [0.0])]:
            net1.train_on_pattern(x, d, alpha=0.3, beta=0.0)
            net2.train_on_pattern(x, d, alpha=0.3, beta=0.0)
        w1, b1 = net1.copy_params()
        w2, b2 = net2.copy_params()
        self.assertEqual(w1, w2)
        self.assertEqual(b1, b2)

    def test_momento_modifica_direccion(self):
        """Con beta>0 la actualización difiere de sin momento."""
        net1 = BackpropagationNetwork(3, [2], 1, seed=11)
        net2 = BackpropagationNetwork(3, [2], 1, seed=11)
        for (x, d) in [([1.0, 0.0, 1.0], [1.0]), ([0.0, 1.0, 1.0], [0.0])]:
            net1.train_on_pattern(x, d, alpha=0.3, beta=0.5)
            net2.train_on_pattern(x, d, alpha=0.3, beta=0.0)
        w1, _ = net1.copy_params()
        w2, _ = net2.copy_params()
        self.assertNotEqual(w1, w2)

    def test_entrenar_reduce_error_en_problema_lineal(self):
        net = BackpropagationNetwork(2, [4], 1, seed=9)
        datos = [([0.0, 0.0], 0.0), ([1.0, 0.0], 1.0),
                 ([0.0, 1.0], 1.0), ([1.0, 1.0], 1.0)]
        e0 = sum(net.error_on_pattern(x, [d]) for x, d in datos)
        for _ in range(500):
            for x, d in datos:
                net.train_on_pattern(x, [d], alpha=0.4, beta=0.5)
        e1 = sum(net.error_on_pattern(x, [d]) for x, d in datos)
        self.assertLess(e1, e0)
        self.assertLess(net.predict([0.0, 0.0])[0], 0.4)
        self.assertGreater(net.predict([1.0, 1.0])[0], 0.6)


class TestPredictClass(unittest.TestCase):
    def test_argmax(self):
        net = BackpropagationNetwork(35, [20], 5, seed=42)
        out = net.predict([0.0] * 35)
        self.assertEqual(net.predict_class([0.0] * 35), out.index(max(out)))


class TestPesosEfectivos(unittest.TestCase):
    def test_dimensiones(self):
        net = BackpropagationNetwork(35, [20], 5, seed=42)
        eff = net.effective_input_weights()
        self.assertEqual(len(eff), 5)
        for vector in eff:
            self.assertEqual(len(vector), 35)
            for v in vector:
                self.assertTrue(v == v)  # no NaN

    def test_varias_capas_ocultas(self):
        net = BackpropagationNetwork(35, [24, 12], 5, seed=3)
        eff = net.effective_input_weights()
        self.assertEqual(len(eff), 5)
        self.assertTrue(all(len(v) == 35 for v in eff))

    def test_salida_k_depende_de_entrada_por_sus_pesos(self):
        """El peso efectivo k->píxel i coincide con la propagación manual
        de la matriz de pesos para una red sin capa oculta."""
        import random
        rng = random.Random(0)
        net = BackpropagationNetwork(5, [3], 3, seed=0)
        w1 = net.weights[0]
        w2 = net.weights[1]
        for k in range(3):
            eff = net.effective_input_weights()[k]
            manual = [
                sum(w2[i][k] * w1[i_idx][i] for i in range(3))
                for i_idx in range(5)
            ]
            self.assertAlmostEqual(eff[3], manual[3], places=12)


class TestValidacionEntradas(unittest.TestCase):
    def test_dimension_incorrecta_lanza_error(self):
        net = BackpropagationNetwork(35, [20], 5, seed=42)
        with self.assertRaises(ValueError):
            net.predict([0.0] * 30)

    def test_tamano_capas_invalido(self):
        with self.assertRaises(ValueError):
            BackpropagationNetwork(0, [20], 5)
        with self.assertRaises(ValueError):
            BackpropagationNetwork(35, [0], 5)


if __name__ == "__main__":
    unittest.main()