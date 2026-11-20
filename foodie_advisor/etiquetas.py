"""Etiquetas débiles, enmascarado del término disparador y muestras para anotar.

Unidad: la oración. Salida: 7 columnas binarias, una por etapa, que valen 1 si la oración
reporta una FALLA de esa etapa. Las 4 primeras son el ciclo de servicio; las 3 últimas,
lo que queda fuera del ciclo y sirve de comparación. Una oración sin ningún 1 es "ninguna".

Etiquetas débiles (Madrid, solo restaurantes de train):
  positivo de la etapa e  = la oración coincide con una regex de falla de e (lexico.ETAPAS)
  negativo de todas       = oración de una reseña de 5★ que no coincide con ninguna regex de falla
Los positivos de una etapa también son negativos de las demás.

Enmascarado: en el 50% de los positivos el término que disparó la regex se reemplaza por
<mask>. Si no, el modelo aprende a reconocer "rude" o "slow" (es decir, a copiar la regex)
y no el contexto ("nobody came to our table for 20 minutes") que delata la falla.
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import lexico

ETIQUETAS = ["llegada", "atencion", "cocina_mesa", "cobro", "sabor", "precio", "local"]
CICLO = ETIQUETAS[:4]
NOMBRE = dict(zip(ETIQUETAS, lexico.ETAPAS))          # llegada → "1. Llegada y mesa", ...
MASK = "<mask>"
SEMILLA = 0


# ---------------------------------------------------------------- Regex por etiqueta
def regex_por_etiqueta(etapas=lexico.ETAPAS):
    """Una regex compilada por etiqueta: la unión de sus términos de tipo falla."""
    tabla = lexico.info(etapas)
    out = {}
    for etq, etapa in zip(ETIQUETAS, etapas):
        rxs = tabla.rx[(tabla.etapa == etapa) & (tabla.tipo == "falla")]
        out[etq] = re.compile("|".join(f"(?:{r})" for r in rxs))
    return out


def etiquetas_regex(oraciones, etapas=lexico.ETAPAS, guardia_negacion=False):
    """Matriz 0/1 (n, 7): qué etapas marca la regex en cada oración.

    guardia_negacion=True (léxico v2) descarta coincidencias con un negador en las 3 palabras
    previas ("not slow", "never rude"): cuenta la etapa solo si queda alguna coincidencia sin negar.
    """
    minus = pd.Series(oraciones).str.lower()
    out = {}
    for e, rx in regex_por_etiqueta(etapas).items():
        m = minus.str.contains(rx).values
        if guardia_negacion and m.any():
            idx = np.flatnonzero(m)
            m[idx] = [any(not lexico.negado(t, x.start()) for x in rx.finditer(t)) for t in minus.values[idx]]
        out[e] = m.astype(np.int8)
    return pd.DataFrame(out, index=minus.index)


def enmascarar(texto, etapas=lexico.ETAPAS):
    """Reemplaza por <mask> todo término de falla (de cualquier etapa) presente en la oración."""
    t = texto.lower()
    for rx in regex_por_etiqueta(etapas).values():
        t = rx.sub(f" {MASK} ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- Partición
def particion_restaurantes(restaurantes, dev=0.15, semilla=SEMILLA):
    """Divide los restaurantes de Madrid en train y dev. Devuelve (set_train, set_dev).

    Se parte por restaurante y no por oración: si las reseñas de un local quedan a los dos
    lados, el modelo aprende el menú y el nombre del local y el dev sale optimista.
    """
    unicos = np.array(sorted(pd.unique(pd.Series(restaurantes))))
    tr, de = next(GroupShuffleSplit(n_splits=1, test_size=dev, random_state=semilla)
                  .split(unicos, groups=unicos))
    return set(unicos[tr]), set(unicos[de])


# ---------------------------------------------------------------- Conjunto débil
def conjunto_debil(orac, max_pos_por_etapa=6000, n_neg=30000, frac_mask=0.5, semilla=SEMILLA,
                   etapas=lexico.ETAPAS, guardia_negacion=False):
    """De oraciones (con columnas oracion, estrellas, restaurante) al conjunto de entrenamiento débil.

    Balanceo: hasta max_pos_por_etapa positivos por etapa (las raras, como Cobro, entran
    completas) y n_neg negativos de reseñas 5★. Devuelve un DataFrame con: texto (lo que ve
    el modelo, enmascarado o no), oracion (original), las 7 etiquetas, enmascarada, origen, peso.
    """
    rng = np.random.default_rng(semilla)
    Y = etiquetas_regex(orac.oracion, etapas, guardia_negacion)
    Y.index = orac.index
    # Negativos: ni siquiera una coincidencia negada ("not slow" tampoco es un buen negativo puro)
    hay = Y.any(axis=1) | etiquetas_regex(orac.oracion, etapas).set_axis(orac.index).any(axis=1)

    elegidos = set()
    for e in ETIQUETAS:
        idx = Y.index[Y[e] == 1]
        if len(idx) > max_pos_por_etapa:
            idx = rng.choice(idx, max_pos_por_etapa, replace=False)
        elegidos.update(idx)
    pos = orac.loc[sorted(elegidos)].copy()
    cand_neg = orac.index[(orac.estrellas == 5) & ~hay]
    neg = orac.loc[rng.choice(cand_neg, min(n_neg, len(cand_neg)), replace=False)].copy()

    pos["origen"], neg["origen"] = "débil +", "débil −"
    mascara = rng.random(len(pos)) < frac_mask
    pos["enmascarada"] = mascara
    pos["texto"] = [enmascarar(t, etapas) if m else t for t, m in zip(pos.oracion, mascara)]
    neg["enmascarada"] = False
    neg["texto"] = neg.oracion

    out = pd.concat([pos, neg])
    out[ETIQUETAS] = Y.loc[out.index].values
    out["peso"] = 1.0
    return out.sample(frac=1, random_state=semilla)


# ---------------------------------------------------------------- Anotación
COLUMNAS_ANOTACION = ETIQUETAS + ["dudosa", "revisada", "nota"]


def con_contexto(orac, idx):
    """Agrega la oración anterior y la siguiente de la misma reseña (ayudan a resolver 'it', 'they')."""
    o = orac[["review_id", "n", "oracion"]]
    clave = o.set_index(["review_id", "n"]).oracion
    sel = o.loc[idx]
    ant = [clave.get((r, n - 1), "") for r, n in zip(sel.review_id, sel.n)]
    sig = [clave.get((r, n + 1), "") for r, n in zip(sel.review_id, sel.n)]
    return pd.DataFrame({"id": sel.review_id + "#" + sel.n.astype(str), "anterior": ant,
                         "oracion": sel.oracion.values, "siguiente": sig})


def hoja_anotacion(orac, idx, semilla=SEMILLA):
    """CSV listo para anotar: id, contexto, oración y columnas vacías. Orden aleatorio."""
    hoja = con_contexto(orac, idx)
    for c in COLUMNAS_ANOTACION:
        hoja[c] = ""
    return hoja.sample(frac=1, random_state=semilla).reset_index(drop=True)


def leer_anotacion(path):
    """Lee una hoja anotada: vacío = 0 en las etiquetas; exige la columna revisada."""
    h = pd.read_csv(path, dtype=str, keep_default_na=False)
    h = h[h.revisada.str.strip() != ""].copy()
    for c in ETIQUETAS + ["dudosa"]:
        h[c] = h[c].str.strip().isin(["1", "x", "X", "si", "sí"]).astype(np.int8)
    return h


# ---------------------------------------------------------------- Muestras para anotar
# Palabras de TEMA (no de falla) para sobre-muestrear Cobro y Llegada en el gold sin
# usar la regex de fallas: "bill" o "table" no dicen si algo salió mal.
TEMA_COBRO = r"\b(?:bill|check|receipt|charg\w*|paid|pay\w*|euros?|tip|tips|card|cash)\b|€"
TEMA_LLEGADA = r"\b(?:reserv\w*|book\w*|table|tables|queue\w*|wait\w*|arriv\w*|seated|seat|door|entrance|host\w*|walk(?:ed)? in|turned away)\b"

# Asignación del gold de Barcelona: (grupo de estrellas, estrato) → oraciones
ASIGNACION_GOLD = {("≤3★", "tema cobro"): 40, ("≤3★", "tema llegada"): 40, ("≤3★", "resto"): 120,
                   ("4-5★", "tema cobro"): 15, ("4-5★", "tema llegada"): 15, ("4-5★", "resto"): 70}


def estratos_gold(orac):
    """Estrato disjunto de cada oración: grupo de estrellas × tema (cobro > llegada > resto)."""
    minus = orac.oracion.str.lower()
    tema = np.where(minus.str.contains(TEMA_COBRO), "tema cobro",
                    np.where(minus.str.contains(TEMA_LLEGADA), "tema llegada", "resto"))
    grupo = np.where(orac.estrellas <= 3, "≤3★", "4-5★")
    return pd.Series(list(zip(grupo, tema)), index=orac.index)


def muestra_gold(orac, asignacion=ASIGNACION_GOLD, semilla=SEMILLA):
    """Muestreo estratificado al azar. Devuelve la clave: índice, estrato, peso = N_estrato / n_estrato."""
    rng = np.random.default_rng(semilla)
    est = estratos_gold(orac)
    partes = []
    for clave, n in asignacion.items():
        pool = est.index[est.map(lambda e: e == clave)]
        sel = rng.choice(pool, n, replace=False)
        partes.append(pd.DataFrame({"idx": sel, "grupo": clave[0], "estrato": clave[1],
                                    "N_estrato": len(pool), "peso": len(pool) / n}))
    return pd.concat(partes, ignore_index=True)


def muestra_madrid_train(orac_train, n=300, semilla=SEMILLA):
    """Oraciones de reseñas 1-2★ de restaurantes de train que la regex de fallas NO cubre."""
    rng = np.random.default_rng(semilla)
    sin_regex = ~etiquetas_regex(orac_train.oracion).set_axis(orac_train.index).any(axis=1)
    pool = orac_train.index[(orac_train.estrellas <= 2) & sin_regex]
    return pd.DataFrame({"idx": rng.choice(pool, n, replace=False), "grupo": "1-2★", "estrato": "sin regex",
                         "N_estrato": len(pool), "peso": len(pool) / n})


# Ampliación del gold (decisión de George, 12-nov): +100 oraciones solo de los estratos de tema
ASIGNACION_GOLD_EXTRA = {("≤3★", "tema cobro"): 35, ("≤3★", "tema llegada"): 35,
                         ("4-5★", "tema cobro"): 15, ("4-5★", "tema llegada"): 15}


def muestra_gold_extra(orac, clave_base, asignacion=ASIGNACION_GOLD_EXTRA, semilla=SEMILLA + 1):
    """Oraciones extra de los estratos de tema, sin repetir oraciones ni restaurantes del gold base.

    Los pesos se recalculan en combinar_pesos() con el n combinado (base + extra) del estrato.
    """
    rng = np.random.default_rng(semilla)
    est = estratos_gold(orac)
    fuera = orac.restaurante.isin(set(orac.restaurante.loc[clave_base.idx])) | orac.index.isin(clave_base.idx)
    partes = []
    for clave, n in asignacion.items():
        en_estrato = est.map(lambda e: e == clave)
        pool = orac.index[en_estrato & ~fuera]
        sel = rng.choice(pool, n, replace=False)
        partes.append(pd.DataFrame({"idx": sel, "grupo": clave[0], "estrato": clave[1],
                                    "N_estrato": int(en_estrato.sum())}))
    return pd.concat(partes, ignore_index=True)


def combinar_pesos(*claves):
    """Une claves de muestreo y pone peso = N_estrato / n combinado del estrato."""
    todo = pd.concat(claves, ignore_index=True)
    n = todo.groupby(["grupo", "estrato"]).idx.transform("size")
    todo["peso"] = todo.N_estrato / n
    return todo


def quitar_disparador(texto, etapas=lexico.ETAPAS):
    """Borra (en vez de enmascarar) los términos de falla: la oración queda como una paráfrasis sin léxico."""
    return re.sub(r"\s+", " ", enmascarar(texto, etapas).replace(MASK, " ")).strip()


def anotadas(nombre, peso=5.0):
    """Hoja anotada → DataFrame con texto, 7 etiquetas y peso. Vacío si nadie ha revisado nada."""
    from . import ANOTACION
    h = leer_anotacion(ANOTACION / nombre)
    out = h[["id", "oracion"] + ETIQUETAS].rename(columns={"oracion": "texto"})
    out["oracion"] = out.texto
    out["peso"], out["origen"], out["enmascarada"] = peso, "anotada", False
    return out
