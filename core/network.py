"""Red Neuronal Artificial Backpropagation (RNA-BP) implementada a mano.

Arquitectura dinámica: capa de entrada -> N capas ocultas configurable ->
capa de salida. Todas las neuronas usan la función de activación sigmoide:

    net = sum(w_ji * x_i) + theta_j
    y   = 1 / (1 + exp(-net))

Propagación hacia atrás (regla delta generalizada) con dos capas o más de
capas ocultas y término de momento opcional:

    capa de salida:   delta_k = (d_k - y_k) * y_k * (1 - y_k)
    capa oculta:      delta_j = y_j * (1 - y_j) * sum_k(delta_k * w_kj)

    w(t+1) = w(t) + alpha * delta * entrada + beta * (w(t) - w(t-1))
    theta(t+1) = theta(t) + alpha * delta + beta * (theta(t) - theta(t-1))

Sin librerías de machine learning: solo math, random y listas nativas.
"""

import math
import random


class BackpropagationNetwork:
    """RNA backpropagation manual con capas configuradas por el usuario.

    Las matrices de pesos tienen forma `weights[l][i][j]`: conecta la
    neurona `i` de la capa `l` con la neurona `j` de la capa `l + 1`.
    Los sesgos (bias) son `biases[l][j]`.
    """

    def __init__(self, input_size, hidden_sizes, output_size, seed=42):
        if input_size <= 0:
            raise ValueError("El número de entradas debe ser mayor que 0.")
        if output_size <= 0:
            raise ValueError("El número de salidas debe ser mayor que 0.")
        for size in hidden_sizes:
            if size <= 0:
                raise ValueError("Cada capa oculta debe tener al menos 1 neurona.")

        self.input_size = int(input_size)
        self.hidden_sizes = [int(s) for s in hidden_sizes]
        self.output_size = int(output_size)
        self.layer_sizes = (
            [self.input_size] + self.hidden_sizes + [self.output_size]
        )

        rng = random.Random(int(seed))
        self.weights = []
        self.biases = []
        self._prev_w = []
        self._prev_b = []
        for l in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[l]
            n_out = self.layer_sizes[l + 1]
            w = [
                [rng.uniform(-0.5, 0.5) for _ in range(n_out)]
                for _ in range(n_in)
            ]
            b = [0.0] * n_out
            self.weights.append(w)
            self.biases.append(b)
            self._prev_w.append(
                [[0.0] * n_out for _ in range(n_in)]
            )
            self._prev_b.append([0.0] * n_out)

    @property
    def num_layers(self):
        """Número de conjuntos de pesos (capas de conexión)."""
        return len(self.weights)

    @staticmethod
    def sigmoid(z):
        """Sigmoide numéricamente estable."""
        if z >= 0:
            exp_neg = math.exp(-z)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(z)
        return exp_pos / (1.0 + exp_pos)

    def forward(self, inputs):
        """Propaga las entradas hacia adelante capa por capa.

        Devuelve la lista de activaciones de cada capa; la última es la
        salida de la red.
        """
        if len(inputs) != self.input_size:
            raise ValueError(
                f"El vector de entrada tiene {len(inputs)} valores, "
                f"pero la red espera {self.input_size}."
            )
        activations = []
        current = [float(v) for v in inputs]
        for w, b in zip(self.weights, self.biases):
            n_out = len(b)
            net = [b[j] for j in range(n_out)]
            for j in range(n_out):
                row = w
                acc = net[j]
                for i in range(len(current)):
                    acc += row[i][j] * current[i]
                net[j] = acc
            layer = [self.sigmoid(z) for z in net]
            activations.append(layer)
            current = layer
        return activations

    def predict(self, inputs):
        """Salida cruda de la capa de salida (vector de longitud output_size)."""
        return self.forward(inputs)[-1]

    def predict_proba(self, inputs):
        """Las salidas sigmoideas se usan como probabilidades informales."""
        return list(self.predict(inputs))

    def predict_class(self, inputs):
        """Clase predicha = índice de la salida con mayor valor (argmax)."""
        out = self.predict(inputs)
        return out.index(max(out))

    def delta_outputs(self, outputs, targets):
        """delta_k = (d_k - y_k) * y_k * (1 - y_k)."""
        return [
            (float(targets[k]) - outputs[k]) * outputs[k] * (1.0 - outputs[k])
            for k in range(len(outputs))
        ]

    def effective_input_weights(self):
        """Peso efectivo de cada entrada hacia cada neurona de salida.

        Propaga hacia atrás las matrices de pesos (linealizando las
        sigmoide) para obtener, por cada salida k, un vector de
        ``input_size`` valores: cuánto "pesa" cada píxel de la grilla en
        la activación de dicha salida. Sirve para visualizar qué patrón
        aprendió la red (p. ej. la plantilla que reconoce a la vocal "A").
        """
        results = []
        n_out = self.output_size
        for k in range(n_out):
            vector = [0.0] * n_out
            vector[k] = 1.0
            for l in range(self.num_layers - 1, -1, -1):
                w = self.weights[l]
                n_prev = len(w)
                new_vector = [0.0] * n_prev
                for i in range(n_prev):
                    acc = 0.0
                    for j in range(len(vector)):
                        acc += vector[j] * w[i][j]
                    new_vector[i] = acc
                vector = new_vector
            results.append(vector)
        return results

    def train_on_pattern(self, inputs, targets, alpha, beta):
        """Un paso de aprendizaje (una época parcial) para un patrón.

        Devuelve E_p = 0.5 * sum(d_k - y_k)^2 (error por patrón), que es
        la medida que el entrenamiento minimiza; el término
        delta_k = (d_k - y_k) * y_k * (1 - y_k) es el que actualiza los
        pesos.
        """
        activations = self.forward(inputs)
        outputs = activations[-1]
        if len(targets) != len(outputs):
            raise ValueError(
                f"La salida deseada tiene {len(targets)} valores, "
                f"pero la red produce {len(outputs)}."
            )

        deltas = [None] * self.num_layers
        deltas[-1] = self.delta_outputs(outputs, targets)
        for l in range(self.num_layers - 2, -1, -1):
            y_hidden = activations[l]
            delta_next = deltas[l + 1]
            w_next = self.weights[l + 1]
            d_h = [0.0] * len(y_hidden)
            for i in range(len(y_hidden)):
                acc = 0.0
                for k in range(len(delta_next)):
                    acc += delta_next[k] * w_next[i][k]
                d_h[i] = y_hidden[i] * (1.0 - y_hidden[i]) * acc
            deltas[l] = d_h

        for l in range(self.num_layers):
            current = inputs if l == 0 else activations[l - 1]
            w = self.weights[l]
            b = self.biases[l]
            pw = self._prev_w[l]
            pb = self._prev_b[l]
            for i in range(len(current)):
                xi = current[i]
                for j in range(len(w[i])):
                    change = (
                        alpha * deltas[l][j] * xi + beta * pw[i][j]
                    )
                    w[i][j] += change
                    pw[i][j] = change
            for j in range(len(b)):
                change = alpha * deltas[l][j] + beta * pb[j]
                b[j] += change
                pb[j] = change

        per_pattern = 0.5 * sum(
            (float(targets[k]) - outputs[k]) ** 2
            for k in range(len(outputs))
        )
        return per_pattern

    def error_on_pattern(self, inputs, targets):
        """E_p = 0.5 * sum(d_k - y_k)^2 sin modificar pesos."""
        outputs = self.predict(inputs)
        return 0.5 * sum(
            (float(targets[k]) - outputs[k]) ** 2
            for k in range(len(outputs))
        )

    def copy_params(self):
        """Copia profunda de pesos y sesgos."""
        w = [[list(row) for row in mat] for mat in self.weights]
        b = [list(vec) for vec in self.biases]
        return w, b

    def reset(self, seed):
        """Reinicia la red con una nueva semilla (para re-entrenar limpio)."""
        self.__init__(self.input_size, self.hidden_sizes, self.output_size, seed)