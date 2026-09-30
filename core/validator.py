"""Validaciones en línea de configuración y de dataset. Todas devuelven una
lista de mensajes claros en español; lista vacía = todo correcto."""


def validate_config(alpha, beta, desired_error, max_epochs, timeout,
                    hidden_layers):
    errors = []
    if alpha is None or alpha <= 0:
        errors.append("La tasa de aprendizaje (alfa) debe ser mayor que 0.")
    if beta is None or beta < 0 or beta >= 1:
        errors.append("El momento (beta) debe estar entre 0 y 0.99.")
    if desired_error is None or desired_error <= 0:
        errors.append("El error deseado debe ser mayor que 0.")
    if max_epochs is None or max_epochs <= 0:
        errors.append("El máximo de épocas debe ser mayor que 0.")
    if timeout is None or timeout <= 0:
        errors.append("El tiempo límite (segundos) debe ser mayor que 0.")
    if not hidden_layers:
        errors.append("Debe existir al menos una capa oculta.")
    else:
        for i, size in enumerate(hidden_layers, start=1):
            if size is None or size <= 0:
                errors.append(f"La capa oculta {i} debe tener al menos 1 neurona.")
    return errors


def validate_dataset(patterns):
    """Comprueba que el dataset tenga al menos un patrón por clase."""
    errors = []
    if not patterns:
        errors.append("El dataset está vacío. Cargue patrones o restaure el "
                      "dataset por defecto.")
        return errors
    classes = sorted({p.label for p in patterns})
    missing = [c for c in range(5) if c not in classes]
    if missing:
        from data.dataset import VOWELS

        faltan = ", ".join(VOWELS[c] for c in missing)
        errors.append(
            f"Faltan patrones de la(s) vocal(es): {faltan}. Por eso el botón "
            "ENTRENAR está desactivado. Agregue al menos un patrón por vocal."
        )
    for i, p in enumerate(patterns, start=1):
        if not isinstance(p.grid, (list, tuple)) or len(p.grid) != 35:
            errors.append(
                f"El patrón {i} no tiene 35 píxeles (se esperan 7 filas x 5 "
                "columnas)."
            )
            break
        for value in p.grid:
            if value not in (0, 1):
                errors.append(
                    f"El patrón {i} tiene un valor no binario: {value}."
                )
                break
    return errors