"""Estado compartido entre las pestañas del asistente."""

from PySide6.QtCore import QObject, Signal

from core.metrics import accuracy, build_confusion_matrix, per_class_metrics
from core.trainer import StopReason
from core.validator import validate_dataset
from data.dataset import (
    N_CLASSES,
    VOWELS,
    stratified_split,
)
from data.noise import build_noisy_dataset

STUDENTS = [
    "Camilo Soler",
    "Sebastian Valencia",
    "Juan Acevedo",
    "Wilson Cristancho",
]

HELP = {
    "net": "net = Σ(wᵢ·xᵢ) + θ : la suma ponderada de las entradas de una "
           "neurona antes de aplicar la sigmoide.",
    "y": "y = 1/(1+e⁻ⁿᵉᵗ) : salida de la neurona tras la activación "
         "sigmoide. Queda entre 0 y 1.",
    "delta": "δ = (d−y)·y·(1−y) en la salida: cuánto 'corregir' cada peso. "
             "En capas ocultas δ se propaga hacia atrás sumando δ·w.",
    "alpha": "α (tasa de aprendizaje): qué tan grandes son los pasos al "
             "corregir los pesos. Muy grande = inestable; muy pequeña = lento.",
    "beta": "β (momento): conserva parte del ajuste anterior para acelerar y "
            "suavizar el descenso. Su efecto se acumula con las épocas: con β "
            "muy pequeña (0 a 0.1) apenas se nota; acelera claramente hacia "
            "0.4-0.8, y valores muy altos (≈0.9) desestabilizan.",
    "epoch": "Época: una pasada completa por todos los patrones de "
             "entrenamiento, actualizando los pesos uno a uno.",
    "et": "ET (error total) = promedio de Eᵖ = 0.5·Σ(d−y)² sobre todos los "
          "patrones de la época. Baja cuando la red aprende.",
    "mse": "MSE / ET: medida del error que se quiere minimizar durante el "
           "entrenamiento.",
    "convergencia": "Converge cuando ET ≤ error deseado: la red aprendió el "
                    "conjunto de entrenamiento dentro de la tolerancia fijada.",
    "argmax": "Cada resultado se resuelve automáticamente: la vocal predicha "
              "es la salida con mayor valor (argmax). Nunca hay que comparar "
              "números a mano.",
}


class AppState(QObject):
    data_changed = Signal()
    config_changed = Signal()

    def __init__(self):
        super().__init__()
        self.vowels = list(VOWELS)
        self.n_classes = N_CLASSES
        self.alpha = 0.3
        self.beta = 0.5
        self.desired_error = 0.01
        self.max_epochs = 5000
        self.timeout = 60.0
        self.seed = 42
        self.ui_every = 10
        self.hidden_layers = [20]
        self.patterns = build_noisy_dataset()
        self.train_indices, self.val_indices = self._split()
        self.result = None
        self.network = None
        self.students = list(STUDENTS)

    def trained(self):
        return self.result is not None and self.network is not None

    def _split(self):
        return stratified_split(self.patterns, seed=self.seed)

    def recompute_split(self):
        self.train_indices, self.val_indices = self._split()

    def restore_defaults(self):
        self.alpha = 0.3
        self.beta = 0.5
        self.desired_error = 0.01
        self.max_epochs = 5000
        self.timeout = 60.0
        self.seed = 42
        self.ui_every = 10
        self.hidden_layers = [20]
        self.patterns = build_noisy_dataset()
        self.recompute_split()
        self.result = None
        self.network = None

    def set_dataset(self, patterns):
        self.patterns = list(patterns)
        self.recompute_split()

    @property
    def train_patterns(self):
        return [self.patterns[i] for i in self.train_indices]

    @property
    def val_patterns(self):
        return [self.patterns[i] for i in self.val_indices]

    @property
    def training_arrays(self):
        train_x = [self.patterns[i].grid for i in self.train_indices]
        train_y = [
            [1.0 if k == self.patterns[i].label else 0.0 for k in range(self.n_classes)]
            for i in self.train_indices
        ]
        val_x = [self.patterns[i].grid for i in self.val_indices]
        val_y = [
            [1.0 if k == self.patterns[i].label else 0.0 for k in range(self.n_classes)]
            for i in self.val_indices
        ]
        return train_x, train_y, val_x, val_y

    def dataset_errors(self):
        return validate_dataset(self.patterns)

    def label_counts(self):
        counts = {c: 0 for c in range(self.n_classes)}
        for p in self.patterns:
            counts[p.label] += 1
        return counts

    def val_confusion_matrix(self):
        if self.result is None:
            return None
        n = self.n_classes
        y_true = list(self.result.y_true_val)
        y_pred = list(self.result.val_preds)
        return build_confusion_matrix(y_true, y_pred, n)

    def val_metrics(self):
        cm = self.val_confusion_matrix()
        if cm is None:
            return None
        return {
            "cm": cm,
            "accuracy": accuracy(cm),
            "per_class": per_class_metrics(cm),
        }

    def summary_line(self):
        capas = " - ".join(str(s) for s in self.hidden_layers)
        return (
            f"α = {self.alpha:g} · β = {self.beta:g} · error deseado = "
            f"{self.desired_error:g} · máx épocas = {self.max_epochs} · "
            f"timeout = {self.timeout:g} s · capas ocultas: {capas} · "
            f"semilla = {self.seed}"
        )

    @staticmethod
    def reason_label(reason):
        if reason == StopReason.CONVERGED:
            return "La red CONVERGIÓ: alcanzó el error deseado."
        if reason == StopReason.MAX_EPOCHS:
            return "Terminó por límite de épocas sin alcanzar el error deseado."
        if reason == StopReason.TIMEOUT:
            return "Terminó por tiempo límite sin alcanzar el error deseado."
        if reason == StopReason.CANCELLED:
            return "Entrenamiento CANCELADO por el usuario."
        return reason