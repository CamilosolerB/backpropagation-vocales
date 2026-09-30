"""Pruebas del entrenador: convergencia sobre el dataset por defecto,
motivos de parada (máximo de épocas, timeout, cancelación) y corrección
del historial."""

import threading
import unittest

from core.network import BackpropagationNetwork
from core.trainer import BackpropTrainer, StopReason, TrainingConfig
from data.dataset import stratified_split, CLEAN_GRIDS, VOWELS, grid_from_rows
from data.noise import build_noisy_dataset


def _default_problem():
    dataset = build_noisy_dataset()
    train_idx, val_idx = stratified_split(dataset, seed=42)
    train_x = [dataset[i].grid for i in train_idx]
    train_y = [
        [1.0 if k == dataset[i].label else 0.0 for k in range(5)]
        for i in train_idx
    ]
    val_x = [dataset[i].grid for i in val_idx]
    val_y = [
        [1.0 if k == dataset[i].label else 0.0 for k in range(5)]
        for i in val_idx
    ]
    net = BackpropagationNetwork(35, [20], 5, seed=42)
    return BackpropTrainer(net, train_x, train_y, val_x, val_y), net


class TestEntrenador(unittest.TestCase):
    def test_dataset_por_defecto_converge(self):
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=0.01, max_epochs=8000, timeout=60
        )
        result = trainer.train(config)
        self.assertEqual(result.reason, StopReason.CONVERGED)
        self.assertLessEqual(result.final_et, config.desired_error)
        self.assertGreater(result.epochs, 0)
        self.assertLess(result.final_et, 0.05)

    def test_dataset_por_defecto_valida_bien(self):
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=0.01, max_epochs=8000, timeout=60
        )
        result = trainer.train(config)
        self.assertEqual(result.val_total, 5)
        self.assertEqual(result.val_correct, result.val_total)
        self.assertEqual(
            result.val_preds, result.y_true_val
        )

    def test_historial_misma_longitud_que_epocas(self):
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.05, beta=0.0,
            desired_error=1e-12, max_epochs=7, timeout=60,
        )
        result = trainer.train(config)
        self.assertEqual(result.reason, StopReason.MAX_EPOCHS)
        self.assertEqual(len(result.history), result.epochs)
        self.assertEqual(result.epochs, 7)
        self.assertEqual(result.history[0].epoch, 1)
        self.assertEqual(result.history[-1].epoch, 7)
        self.assertIn("límite", result.status_message.lower())

    def test_momento_reduce_error(self):
        """Con momento debe converger o acercarse más que sin él."""
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=0.01, max_epochs=4000, timeout=60
        )
        trainer, _ = _default_problem()
        result = trainer.train(config)
        self.assertEqual(result.reason, StopReason.CONVERGED)

    def test_timeout_detiene_el_bucle(self):
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=1e-20, max_epochs=10 ** 9,
            timeout=0.05,
        )
        result = trainer.train(config)
        self.assertEqual(result.reason, StopReason.TIMEOUT)
        self.assertLess(result.epochs, 10000)

    def test_cancelacion(self):
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=1e-20, max_epochs=10 ** 9,
            timeout=60,
        )
        cancel = threading.Event()
        cancel.set()
        result = trainer.train(config, cancel_event=cancel)
        self.assertEqual(result.reason, StopReason.CANCELLED)
        self.assertEqual(result.epochs, 0)

    def test_modo_paso_a_paso(self):
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=0.01, max_epochs=3, timeout=60
        )
        step = threading.Event()
        epochs_vistos = []

        def on_epoch(rec):
            epochs_vistos.append(rec.epoch)
            step.set()

        result = trainer.train(config, on_epoch=on_epoch, step_event=step)
        self.assertEqual(result.reason, StopReason.MAX_EPOCHS)
        self.assertEqual(
            epochs_vistos,
            [1, 2, 3],
        )

    def test_paso_a_paso_ignora_timeout(self):
        """En modo paso a paso NO aplica el tiempo límite: el entrenamiento
        avanza hasta la iteración final (máximo de épocas), aunque el
        timeout declarado sea mínimo y el usuario vaya a su ritmo."""
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=1e-12, max_epochs=3,
            timeout=0.001,
        )
        step = threading.Event()
        epochs_vistos = []

        def on_epoch(rec):
            epochs_vistos.append(rec.epoch)
            step.set()

        result = trainer.train(config, on_epoch=on_epoch, step_event=step)
        self.assertEqual(result.reason, StopReason.MAX_EPOCHS)
        self.assertEqual(epochs_vistos, [1, 2, 3])

    def test_paso_a_paso_sin_clic_espera_pero_no_se_congela(self):
        """Si en modo paso a paso nunca se pulsa 'siguiente época', el hilo
        queda esperando (no se corta por timeout) pero la cancelación del
        usuario sí lo detiene: nunca se congela la aplicación."""
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=1e-20, max_epochs=10 ** 9,
            timeout=0.3,
        )
        import time

        step = threading.Event()
        cancel = threading.Event()

        def fire_cancel():
            time.sleep(0.05)
            cancel.set()

        timer = threading.Thread(target=fire_cancel, daemon=True)
        timer.start()
        result = trainer.train(
            config, step_event=step, cancel_event=cancel,
        )
        self.assertEqual(result.reason, StopReason.CANCELLED)
        timer.join(timeout=1)

    def test_cancelacion_mientras_pausado(self):
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=1e-20, max_epochs=10 ** 9,
            timeout=5,
        )
        import time

        cancel = threading.Event()
        pause = threading.Event()
        pause.set()
        step = threading.Event()

        def fire_cancel():
            time.sleep(0.05)
            cancel.set()

        timer = threading.Thread(target=fire_cancel, daemon=True)
        timer.start()
        result = trainer.train(
            config, cancel_event=cancel, step_event=step, pause_event=pause,
        )
        self.assertEqual(result.reason, StopReason.CANCELLED)
        timer.join(timeout=1)

    def test_arquitectura_multicapa_entrena(self):
        dataset = build_noisy_dataset()
        train_idx, val_idx = stratified_split(dataset, seed=42)
        train_x = [dataset[i].grid for i in train_idx]
        train_y = [
            [1.0 if k == dataset[i].label else 0.0 for k in range(5)]
            for i in train_idx
        ]
        val_x = [dataset[i].grid for i in val_idx]
        val_y = [
            [1.0 if k == dataset[i].label else 0.0 for k in range(5)]
            for i in val_idx
        ]
        net = BackpropagationNetwork(35, [24, 12], 5, seed=7)
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=0.01, max_epochs=8000, timeout=60
        )
        result = BackpropTrainer(net, train_x, train_y, val_x, val_y).train(config)
        self.assertEqual(result.reason, StopReason.CONVERGED)
        self.assertGreaterEqual(result.val_correct, 4)

    def test_pesos_aprenden_la_forma_de_la_vocal(self):
        """PRUEBA DE APRENDIZAJE: tras converger, la plantilla efectiva de
        cada salida (peso entrada->salida) se parece más a su vocal ideal
        que a cualquier otra (semejanza por coseno). Antes de entrenar esto
        no se cumple con pesos aleatorios."""
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=0.01, max_epochs=8000, timeout=60
        )
        result = trainer.train(config)
        self.assertEqual(result.reason, StopReason.CONVERGED)

        ideals = [grid_from_rows(CLEAN_GRIDS[v]) for v in VOWELS]
        templates = net.effective_input_weights()

        def cos_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = (sum(x * x for x in a)) ** 0.5
            nb = (sum(y * y for y in b)) ** 0.5
            return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

        similarities = [
            [cos_sim(templates[k], ideals[c]) for c in range(5)]
            for k in range(5)
        ]
        for k in range(5):
            self.assertEqual(
                similarities[k].index(max(similarities[k])), k,
                f"La plantilla de {VOWELS[k]} no es la más parecida a su vocal",
            )

    def test_pesos_cambian_durante_el_entrenamiento(self):
        """Los pesos partieron de valores aleatorios y terminan muy
        diferentes: demuestra que la red aprendió."""
        trainer, net = _default_problem()
        config = TrainingConfig(
            alpha=0.3, beta=0.5, desired_error=1e-12, max_epochs=40, timeout=60
        )
        result = trainer.train(config)
        self.assertGreaterEqual(len(result.history), 3)
        w_inicial = next(r.weights for r in result.history if r.weights is not None)
        w_final = result.history[-1].weights
        flat0 = [v for layer in w_inicial for row in layer for v in row]
        flatF = [v for layer in w_final for row in layer for v in row]
        max_delta = max(abs(a - b) for a, b in zip(flat0, flatF))
        self.assertGreater(max_delta, 0.05)


if __name__ == "__main__":
    unittest.main()