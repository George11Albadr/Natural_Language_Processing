"""Cifras clave de cada notebook → resultados/hallazgos_0X.json.

La app lee de aquí: ninguna cifra se transcribe a mano.
"""
import json

import numpy as np
import pandas as pd

from . import RAIZ

RESULTADOS = RAIZ / "resultados"


def _limpiar(x):
    if isinstance(x, dict):
        return {str(k): _limpiar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_limpiar(v) for v in x]
    if isinstance(x, pd.Series):
        return _limpiar(x.to_dict())
    if isinstance(x, pd.DataFrame):
        return _limpiar(x.to_dict(orient="index"))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 4)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def guardar(notebook, cifras):
    RESULTADOS.mkdir(exist_ok=True)
    ruta = RESULTADOS / f"hallazgos_{notebook}.json"
    ruta.write_text(json.dumps(_limpiar(cifras), ensure_ascii=False, indent=1, sort_keys=True))
    print(f"{ruta.name}: {len(cifras)} claves →", ", ".join(sorted(cifras)))
    return ruta


def cargar(notebook):
    ruta = RESULTADOS / f"hallazgos_{notebook}.json"
    return json.loads(ruta.read_text()) if ruta.exists() else None
