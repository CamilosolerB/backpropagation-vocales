"""Entrenador de la RNA-BP: controla épocas, criterios de parada,
historial por época y llamadas en vivo (sin congelar la interfaz).

Criterios de parada (configurables):
    * ET <= error deseado  (convergió)
    * máximo de épocas
    * tiempo límite (timeout)
    * cancelado por el usuario (evento de hilo)

El entrenamiento es online (los pesos se actualizan patrón a patrón) y al
terminar cada época se calcula ET = promedio de E_p sobre los patrones.
"""

import time
from dataclasses import dataclass, field

from core.network import BackpropagationNetwork


class StopReason:
    CONVERGED = "converged"
    MAX_EPOCHS = "max_epochs"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


MESSAGES = {
    StopReason.CONVERGED: "La red CONVERGIÓ: alcanzó el error deseado.",
    StopReason.MAX_EPOCHS: "No alcanzó el error deseado (límite de épocas).",
    StopReason.TIMEOUT: "Entrenamiento detenido por tiempo límite.",
    StopReason.CANCELLED: "Entrenamiento cancelado por el usuario.",
}


@dataclass
class TrainingConfig:
    alpha: float
    desired_error: float
    max_epochs: int
    timeout: float
    beta: float = 0.0


@dataclass
class EpochRecord:
    epoch: int
    et: float
    best_et: float
    train_correct: int
    train_total: int
    val_correct: int = 0
    val_total: int = 0
    weights: object = None
    biases: object = None
    train_preds: list = field(default_factory=list)
    val_preds: list = field(default_factory=list)


@dataclass
class TrainingResult:
    reason: str
    status_message: str
    epochs: int
    final_et: float
    best_et: float
    final_weights: object
    final_biases: object
    history: list = field(default_factory=list)
    train_preds: list = field(default_factory=list)
    val_preds: list = field(default_factory=list)
    y_true_train: list = field(default_factory=list)
    y_true_val: list = field(default_factory=list)
    train_correct: int = 0
    train_total: int = 0
    val_correct: int = 0
    val_total: int = 0
    converged_at: int = None

    @property
    def converged(self):
        return self.reason == StopReason.CONVERGED

    @property
    def train_accuracy(self):
        return (self.train_correct / self.train_total) if self.train_total else 0.0

    @property
    def val_accuracy(self):
        return (self.val_correct / self.val_total) if self.val_total else 0.0


SNAPSHOT_CAP = 3000


class BackpropTrainer:
    def __init__(self, network, train_x, train_y, val_x=None, val_y=None):
        if not isinstance(network, BackpropagationNetwork):
            raise TypeError("Se espera una BackpropagationNetwork.")
        self.network = network
        self.train_x = [list(map(float, row)) for row in train_x]
        self.train_y = [list(map(float, row)) for row in train_y]
        self.val_x = (
            [list(map(float, row)) for row in val_x] if val_x is not None else None
        )
        self.val_y = (
            [list(map(float, row)) for row in val_y] if val_y is not None else None
        )
        if not self.train_x or len(self.train_x) != len(self.train_y):
            raise ValueError(
                "Debe haber al menos un patrón de entrenamiento y las "
                "entradas deben coincidir con las salidas deseadas."
            )

    def _argmax(self, vec):
        return vec.index(max(vec))

    def _evaluate_epoch(self, y_true_train, y_true_val=None):
        """Clasificaciones y aciertos de la época actual sobre ambos conjuntos."""
        train_correct = 0
        train_preds = []
        for x, d in zip(self.train_x, self.train_y):
            pred = self.network.predict_class(x)
            train_preds.append(pred)
            if pred == self._argmax(d):
                train_correct += 1
        val_preds = []
        val_correct = 0
        if self.val_x is not None and y_true_val is not None:
            for x, d in zip(self.val_x, self.val_y):
                pred = self.network.predict_class(x)
                val_preds.append(pred)
                if pred == self._argmax(d):
                    val_correct += 1
        return train_correct, train_preds, val_correct, val_preds

    def train(self, config, on_epoch=None, cancel_event=None, step_event=None,
              pause_event=None):
        """Entrena hasta converger, agotar épocas, timeout o cancelación.

        `cancel_event` (threading.Event): si se activa, detiene el bucle.
        `step_event` (threading.Event): si se entrega, el entrenamiento pasa
            a modo paso a paso: tras cada época se espera a que el hilo de
            la UI lo active ("siguiente época"). En este modo el tiempo
            límite NO se aplica: el ritmo lo pone el usuario y el
            entrenamiento solo termina por convergencia, máximo de épocas o
            cancelación.
        `pause_event` (threading.Event): si está activo, el bucle espera a
            que se desactive antes de cada época (pausar/reanudar).
        """
        start = time.monotonic()
        config = self._coerce_config(config)
        history = []
        best_et = float("inf")
        converged_at = None
        in_step_mode = step_event is not None
        snapshot_every = max(1, config.max_epochs // SNAPSHOT_CAP)
        y_true_train = [self._argmax(d) for d in self.train_y]
        y_true_val = (
            [self._argmax(d) for d in self.val_y] if self.val_y is not None else None
        )

        def stopped():
            if cancel_event is not None and cancel_event.is_set():
                return True
            if in_step_mode:
                # El usuario controla la cadencia: no hay limite de tiempo.
                return False
            return (time.monotonic() - start) > config.timeout

        for epoch in range(1, config.max_epochs + 1):
            if pause_event is not None:
                while pause_event.is_set() and not stopped():
                    time.sleep(0.02)
            if stopped():
                break

            error_sum = 0.0
            for x, d in zip(self.train_x, self.train_y):
                error_sum += self.network.train_on_pattern(
                    x, d, config.alpha, config.beta
                )
            et = error_sum / len(self.train_x)
            if et < best_et:
                best_et = et
                converged_at = epoch

            train_correct, train_preds, val_correct, val_preds = (
                self._evaluate_epoch(y_true_train, y_true_val)
            )

            snapshot = bool(epoch % snapshot_every == 0 or epoch == 1)
            if snapshot:
                w, b = self.network.copy_params()
            else:
                w, b = None, None
            record = EpochRecord(
                epoch=epoch,
                et=et,
                best_et=best_et,
                train_correct=train_correct,
                train_total=len(self.train_x),
                val_correct=val_correct,
                val_total=len(self.val_x) if self.val_x is not None else 0,
                weights=w,
                biases=b,
                train_preds=train_preds,
                val_preds=val_preds,
            )
            history.append(record)

            if on_epoch is not None:
                on_epoch(record)

            if step_event is not None:
                while not step_event.is_set() and not stopped():
                    time.sleep(0.02)
                if not step_event.is_set():
                    # Se salió por cancelación o timeout mientras se
                    # esperaba el clic de "siguiente época".
                    break
                step_event.clear()

            if (time.monotonic() - start) > config.timeout and not in_step_mode:
                return self._build_result(
                    StopReason.TIMEOUT, len(history), history,
                    best_et, converged_at,
                    y_true_train, y_true_val,
                )
            if et <= config.desired_error:
                return self._build_result(
                    StopReason.CONVERGED, epoch, history,
                    best_et, converged_at,
                    y_true_train, y_true_val,
                )
            if (time.monotonic() - start) > config.timeout and not in_step_mode:
                return self._build_result(
                    StopReason.TIMEOUT, epoch - 1, history,
                    best_et, converged_at,
                    y_true_train, y_true_val,
                )

        reason = StopReason.CANCELLED if (
            cancel_event is not None and cancel_event.is_set()
        ) else StopReason.TIMEOUT if (
            not in_step_mode and (time.monotonic() - start) > config.timeout
        ) else StopReason.MAX_EPOCHS
        return self._build_result(
            reason, len(history), history, best_et, converged_at,
            y_true_train, y_true_val,
        )

    @staticmethod
    def _coerce_config(config):
        return TrainingConfig(
            alpha=float(config.alpha),
            beta=float(getattr(config, "beta", 0.0) or 0.0),
            desired_error=float(config.desired_error),
            max_epochs=int(config.max_epochs),
            timeout=float(config.timeout),
        )

    @staticmethod
    def _build_result(reason, epochs, history, best_et, converged_at,
                      y_true_train, y_true_val):
        final_et = history[-1].et if history else float("inf")
        w, b = None, None
        if history:
            for rec in reversed(history):
                if rec.weights is not None:
                    w, b = rec.weights, rec.biases
                    break
        last = history[-1] if history else None
        return TrainingResult(
            reason=reason,
            status_message=MESSAGES[reason],
            epochs=epochs,
            final_et=final_et,
            best_et=best_et,
            final_weights=w,
            final_biases=b,
            history=list(history),
            train_preds=list(last.train_preds) if last else [],
            val_preds=list(last.val_preds) if last else [],
            y_true_train=list(y_true_train),
            y_true_val=list(y_true_val) if y_true_val is not None else [],
            train_correct=last.train_correct if last else 0,
            train_total=last.train_total if last else 0,
            val_correct=last.val_correct if last else 0,
            val_total=last.val_total if last else 0,
            converged_at=converged_at,
        )