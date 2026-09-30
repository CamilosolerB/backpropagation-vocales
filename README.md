# BACKPROP_VOCALES — Laboratorio #3 de Redes Neuronales Artificiales (UDEC)

Aplicación de escritorio que implementa una **Red Neuronal Artificial con
Backpropagation (RNA-BP) entrenada manualmente** (sin librerías de machine
learning) para el caso de uso **"Clasificación de vocales"** en una grilla
binaria 5x7 (35 píxeles de entrada, 5 vocales de salida en codificación
one-hot).

Sigue la misma convención de los laboratorios anteriores (Perceptrón y
ADALINE): matemática manual en `core/`, interfaz PySide6 en `ui/`,
pruebas automáticas con `unittest` y empaquetado con PyInstaller.

## Características

- **Asistente guiado en 4 pasos**: 1. DATOS → 2. ARQUITECTURA →
  3. ENTRENAMIENTO → 4. RESULTADOS Y PRUEBA.
- Funciona **"de fábrica"**: dataset de 20 vocales (5 x 4 variantes con
  ruido) precargado y arquitectura por defecto (1 capa oculta de 20
  neuronas). Entrena con un solo clic.
- Editor visual de la grilla 5x7 (pintar con clic), generador de ruido,
  carga/exportación TXT y partición estratificada 70/30 automática.
- Entrenamiento **en hilo en segundo plano** con pausa, reanudar,
  cancelar, máximo de épocas y tiempo límite. Nunca se congela.
- Paso 3 con contadores y panel vivo de las 5 vocales (✓/✗ automático
  por argmax); **todas las gráficas se muestran en el Paso 4**: curva de
  error ET vs épocas, diagrama de arquitectura con el color/grosor de las
  conexiones según signo/magnitud del peso, matriz de confusión,
  plantillas aprendidas y evolución de los pesos.
- Modos **continuo** (con tiempo límite opcional) y **paso a paso** (una
  época por clic, **sin límite de tiempo**: avanza solo hasta converger,
  agotar el máximo de épocas o cancelarlo).
- Resultados: matriz de confusión en mapa de calor, precisión,
  sensibilidad y F1 por clase, exactitud global, curva final con el punto
  de convergencia y **comprobación instantánea** de cualquier patrón
  (dibujado o del dataset) con la vocal predicha en grande, barras de
  confianza e indicador ACIERTO/ERROR.
- **Pesos finales**: tablas exactas de las matrices W1, W2 (y capas
  extra) y de los sesgos θ con los valores con los que quedó la red
  (verde = positivo, rojo = negativo).
- **Prueba de aprendizaje**: gráfica de las "plantillas" que cada salida
  aprende (peso efectivo entrada→salida expresado como imagen 5x7)
  comparadas con la vocal ideal — si la red aprendió, la plantilla se
  parece a su vocal y su semejanza es la mayor de la fila, y además la
  curva de evolución de los pesos muestra que pasaron de aleatorios a
  aprendidos.
- Exportación de resultados a TXT, CSV y PNG (incluye
  `plantillas_aprendidas.png` y `evolucion_pesos.png`).
- Ayuda contextual ("?") junto a cada concepto técnico (net, y, δ, α, β,
  época, ET, convergencia).

## Estructura

```
core/            matemática (red, entrenador, métricas, validación, rutas)
data/            dataset de vocales, ruido, TXT I/O, dataset por defecto
visualization/   arquitectura en vivo, curva de error, matriz de confusión
ui/              PySide6 (asistente por pasos)
tests/           suite unittest
main.py          punto de entrada
assets/          logo_udec.png
BACKPROP_VOCALES.spec  configuración de PyInstaller
```

## Matemática implementada (manual)

- Activación **sigmoide en todas las neuronas** (ocultas y de salida):
  `y = 1/(1 + e^(-net))`, `net = Σ(w·x) + θ`. Por eso la regla delta usa
  `y·(1 − y)`, que es la derivada de la sigmoide:
  `d(sigmoid(net))/dnet = y·(1 − y)`.
- Forward pass capa por capa con arquitectura dinámica:
  35 entradas → N capas ocultas configurables → 5 salidas.
- Propagación hacia atrás (regla delta generalizada, para más de una capa
  oculta):
  - Salida: `δ_k = (d_k − y_k)·y_k·(1 − y_k)`
  - Ocultas: `δ_j = y_j·(1 − y_j)·Σ_k(δ_k·w_kj)`
- Actualización con momento opcional (β, por defecto 0.5):
  `w(t+1) = w(t) + α·δ·entrada + β·(w(t) − w(t−1))`.
  Observación práctica medida con el dataset por defecto: β es una
  aceleración que se *acumula* con las épocas, por lo que β muy pequeña
  (p. ej. 0→0.1) apenas cambia el número de épocas (~9 % menos); gana
  claramente hacia β≈0.4–0.8 (hasta ~4× menos épocas), y β muy altos
  (≈0.9) vuelven el descenso inestable.
- Error por patrón: `E_p = 0.5·Σ(d_k − y_k)²`; error por época:
  `ET = promedio de E_p`.
- Peso efectivo por salida: `w_eff(i→k) = Σ_j W2[j][k]·W1[i][j]` (se
  propaga hacia atrás por todas las capas) para visualizar la "plantilla"
  que aprende cada neurona de salida.
- Parada por: convergencia (`ET ≤ error deseado`), máximo de épocas,
  tiempo límite o cancelación.

## Ejecución

```bash
pip install -r requirements.txt
python main.py
```

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests
```

## Empaquetado para Windows

```bash
pyinstaller BACKPROP_VOCALES.spec
# genera dist/BACKPROP_VOCALES.exe (no requiere Python instalado)
```

## Autores

- Camilo Soler
- Sebastian Valencia
- Juan Acevedo
- Wilson Cristancho