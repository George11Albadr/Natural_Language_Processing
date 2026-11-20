"""¿Cuántas estrellas cuesta una falla? Regresión con efectos fijos por restaurante.

A cada variable se le resta su promedio dentro del restaurante (transformación "within"),
así el coeficiente compara reseñas del mismo local y no restaurantes buenos contra malos.
Los IC salen de un bootstrap que remuestrea restaurantes completos (las reseñas de un
mismo local no son independientes).
"""
import numpy as np


def centrar(M, c):
    """Resta a cada fila el promedio de su grupo c (códigos 0..k-1)."""
    s = np.zeros((c.max() + 1,) + M.shape[1:])
    np.add.at(s, c, M)
    n = np.bincount(c)
    return M - (s / (n if M.ndim == 1 else n[:, None]))[c]


def efectos_fijos(M, y, grupos, reps=200, semilla=0):
    """β dentro del grupo e IC 95% por bootstrap de grupos.

    M: matriz (n, k) de regresores; y: estrellas (n,); grupos: etiqueta de restaurante (n,).
    Devuelve (beta, ic_inf, ic_sup), cada uno de largo k.
    """
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float)
    cod = np.unique(np.asarray(grupos), return_inverse=True)[1]
    beta = np.linalg.lstsq(centrar(M, cod), centrar(y, cod), rcond=None)[0]

    filas_por_grupo = np.split(np.argsort(cod, kind="stable"), np.cumsum(np.bincount(cod))[:-1])
    tamanos = np.array([len(f) for f in filas_por_grupo])
    rng, boot = np.random.default_rng(semilla), []
    for _ in range(reps):
        elegidos = rng.integers(0, len(filas_por_grupo), len(filas_por_grupo))
        idx = np.concatenate([filas_por_grupo[k] for k in elegidos])
        grupo = np.repeat(np.arange(len(elegidos)), tamanos[elegidos])
        boot.append(np.linalg.lstsq(centrar(M[idx], grupo), centrar(y[idx], grupo), rcond=None)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return beta, lo, hi


def costos_ciudad(resenas, Y, reps=200, semilla=0):
    """β por etapa (efectos fijos por restaurante) con IC bootstrap. Y: DataFrame 0/1 por etapa, alineado."""
    import pandas as pd
    b, lo, hi = efectos_fijos(Y.values, resenas.estrellas.values, resenas.restaurante.values, reps=reps, semilla=semilla)
    return pd.DataFrame({"β": b, "IC inf": lo, "IC sup": hi, "% de reseñas": Y.mean().values * 100}, index=Y.columns)
