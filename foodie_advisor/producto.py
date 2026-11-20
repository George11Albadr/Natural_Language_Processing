"""Lógica del producto (la app del gerente en producto/app.py).

Reusa lo que ya calculan los notebooks: el contexto de la ciudad del scorecard (tasas contraídas,
percentiles, regla de la acción prioritaria) y las predicciones del regex v1 y de LogReg.
No entrena ni anota nada.

Dos niveles por queja, sin anotar nada nuevo:
  - explícita: la marca el léxico (regex v1); es la que cuenta en la tasa, como en el scorecard;
  - posible:   solo la marca LogReg (lee paráfrasis); se muestra como pista, no se cuenta.
LogReg aprendió de las etiquetas del léxico y marca casi todo lo que marca el regex, así que
cruzar ambas no agrega confianza; lo que agrega LogReg son las posibles.
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import beta as B

from . import PROCESADO, datos, etiquetas as et, scorecard as sc

FUENTE, SECUNDARIA = "regex_v1", ("LogReg", "LogReg_borrado")
MIN_NUEVAS = 30          # con menos reseñas no se da prioridad
VALIDADAS = {"atencion"}
NOMBRE_CORTO = {"llegada": "Llegada y mesa", "atencion": "Atención en sala", "cocina_mesa": "Cocina → mesa",
                "cobro": "Cobro", "sabor": "Sabor", "precio": "Precio", "local": "Local"}
SELLO_CORTO = {"atencion": "validada", "cocina_mesa": "señal débil", "cobro": "no validada", "llegada": "no validada"}


def costos():
    return pd.read_csv(PROCESADO / "costos_regex_v1.csv").set_index(["ciudad", "etapa"])


def cargar_ciudad(ciudad):
    """Contexto completo de una ciudad (pares con >= 100 reseñas)."""
    ctx = sc.contexto_ciudad(ciudad, FUENTE, costos().loc[ciudad, "β"], secundaria=SECUNDARIA)
    _, rs = ctx["secundaria"]
    ctx["res_sec"] = rs.set_index("review_id")[et.ETIQUETAS]
    ctx["costos_ic"] = costos().loc[ciudad]
    return ctx


def verbo(etapa):
    return "Atacar primero" if etapa in VALIDADAS else "Revisar primero"


def titular(prio):
    if prio in et.CICLO:
        return f"{verbo(prio)}: {NOMBRE_CORTO[prio]}", SELLO_CORTO[prio]
    if prio.startswith("fuera: "):
        return f"El problema está fuera del ciclo de servicio: {NOMBRE_CORTO[prio[7:]]}", "fuera del ciclo"
    return "Sin punto débil del servicio frente a sus pares", ""


# ------------------------------------------------------------------ Restaurante del dataset
def ficha(ctx, rest):
    """Tabla por etapa, prioridad y tendencia de un restaurante con >= 100 reseñas."""
    res = ctx["res"]
    r = res[res.restaurante == rest]
    lg = ctx["prioridad_largo"]
    lg = lg[lg.restaurante == rest].set_index("etapa")
    sec = ctx["res_sec"].loc[r.review_id]
    filas = []
    for e in et.ETIQUETAS:
        k, n = int(r[e].sum()), len(r)
        a, b = ctx["priors"][e]
        lo, hi = B.ppf([0.025, 0.975], k + a, n - k + b)
        conf = int((r[e].values * sec[e].values).sum())
        filas.append({"etapa": e, "Etapa": NOMBRE_CORTO[e], "ciclo": e in et.CICLO,
                      "reseñas con queja": k, "confirmadas": conf,
                      "posibles (solo LogReg)": int(((1 - r[e].values) * sec[e].values).sum()),
                      "% quejas": lg.loc[e, "tasa"] * 100, "IC inf": lo * 100, "IC sup": hi * 100,
                      "mediana pares %": lg.loc[e, "mediana"] * 100, "percentil": lg.loc[e, "percentil"],
                      "exceso ★": lg.loc[e, "exceso"], "candidata": bool(lg.loc[e, "candidata"]),
                      "costo ★": ctx["costo"][e], "sello": SELLO_CORTO.get(e, "fuera del ciclo")})
    tabla = pd.DataFrame(filas).set_index("etapa")
    anio = r.assign(año=r.fecha.dt.year).groupby("año")
    tend = (anio[et.CICLO].mean() * 100).assign(reseñas=anio.size(), estrellas=anio.estrellas.mean())
    tend = tend[tend.reseñas >= 20]
    prom = ctx["prom_pares"]
    return {"restaurante": rest, "nombre": rest.replace("_", " "), "n": len(r), "estrellas": r.estrellas.mean(),
            "mediana pares ★": prom.median(), "pct estrellas": (prom < r.estrellas.mean()).mean() * 100,
            "n pares": len(ctx["pares"]), "prioridad": ctx["prioridad"][rest], "tabla": tabla, "tendencia": tend,
            "review_ids": r.review_id}


def citas(ctx, f, etapa, n=4, semilla=0):
    """Oraciones del restaurante marcadas en la etapa, con su nivel de confianza y estrellas."""
    ids = f["review_ids"].tolist()
    o = pd.read_parquet(datos.ruta_oraciones(ctx["ciudad"]), filters=[("review_id", "in", ids)])
    p = ctx["pred"]
    p = p[p.review_id.isin(ids) & (p[etapa] == 1)][["review_id", "n"]]
    o = o.merge(p, on=["review_id", "n"])
    o = o[o.oracion.str.len().between(30, 300)]
    if o.empty:
        return pd.DataFrame(columns=["oracion", "estrellas", "confianza"])
    sec = ctx["res_sec"][etapa]
    o["confianza"] = np.where(sec.loc[o.review_id].values == 1, "confirmada", "explícita")
    o = o.merge(ctx["res"][["review_id", "estrellas"]], on="review_id")
    o["_orden"] = (o.confianza != "confirmada").astype(int)
    o = o.sample(frac=1, random_state=semilla).sort_values("_orden", kind="stable")
    return o.head(n)[["oracion", "estrellas", "confianza"]]


_RX = None


def resaltar(texto, etapa):
    """Marca en negritas el término del léxico que disparó la etapa."""
    global _RX
    if _RX is None:
        _RX = et.regex_por_etiqueta()
    minus, partes, i = texto.lower(), [], 0
    for m in _RX[etapa].finditer(minus):
        partes += [texto[i:m.start()], "**", texto[m.start():m.end()], "**"]
        i = m.end()
    return "".join(partes + [texto[i:]])


def ranking(ctx, etapa, solo_candidatas=True):
    lg = ctx["prioridad_largo"]
    t = lg[lg.etapa == etapa].copy()
    if solo_candidatas:
        t = t[t.candidata]
    t = t.assign(Restaurante=t.restaurante.str.replace("_", " "), reseñas=ctx["n"].loc[t.restaurante].values,
                 estrellas=ctx["prom_pares"].loc[t.restaurante].values,
                 prioridad=ctx["prioridad"].loc[t.restaurante].map(lambda x: titular(x)[0]).values)
    t["% quejas"] = t.tasa * 100
    return t.sort_values("exceso")[["Restaurante", "reseñas", "estrellas", "% quejas", "percentil", "exceso", "prioridad"]]


# ------------------------------------------------------------------ Reseñas nuevas
_LIN = None


def _logreg(oraciones):
    global _LIN
    if _LIN is None:
        from . import modelos
        _LIN = modelos.cargar_lineales("borrado")
    if not len(oraciones):
        return np.zeros((0, len(et.ETIQUETAS)), dtype=np.int8)
    return (_LIN["logreg"].predict_proba(_LIN["vec"].transform(oraciones)) >= 0.5).astype(np.int8)


def leer_resenas(texto):
    """Una reseña por bloque separado por línea en blanco; si no hay líneas en blanco, una por línea."""
    bloques = [b.strip() for b in re.split(r"\n\s*\n", texto.strip()) if b.strip()]
    if len(bloques) <= 1:
        bloques = [l.strip() for l in texto.splitlines() if l.strip()]
    return bloques


def evaluar_nuevas(resenas, ctx):
    """Scorecard de un local que no está en el dataset, contra los pares de la ciudad elegida."""
    filas = [(i, j, o) for i, t in enumerate(resenas) for j, o in enumerate(datos.dividir_oraciones(t))]
    orac = pd.DataFrame(filas, columns=["resena", "n", "oracion"])
    rx = et.etiquetas_regex(orac.oracion).values if len(orac) else np.zeros((0, 7), dtype=np.int8)
    lr = _logreg(orac.oracion.tolist())
    for j, e in enumerate(et.ETIQUETAS):
        orac[e] = rx[:, j] if len(orac) else []
        orac[e + "_lr"] = lr[:, j] if len(orac) else []
    n = len(resenas)
    por_r = orac.groupby("resena")[[*et.ETIQUETAS, *[e + "_lr" for e in et.ETIQUETAS]]].max().reindex(range(n), fill_value=0)
    out = []
    for e in et.ETIQUETAS:
        k = int(por_r[e].sum())
        a, b = ctx["priors"][e]
        tasa = (k + a) / (n + a + b)
        pares = ctx["tasa_contraida"][e]
        med = pares.median()
        inf80 = B.ppf((1 - sc.CRED) / 2, k + a, n - k + b)
        pct = (pares < tasa).mean() * 100
        out.append({"etapa": e, "Etapa": NOMBRE_CORTO[e], "ciclo": e in et.CICLO, "reseñas con queja": k,
                    "confirmadas": int((por_r[e] * por_r[e + "_lr"]).sum()),
                    "posibles (solo LogReg)": int(((1 - por_r[e]) * por_r[e + "_lr"]).sum()),
                    "% quejas": tasa * 100, "mediana pares %": med * 100, "percentil": pct,
                    "exceso ★": (tasa - med) * abs(ctx["costo"][e]),
                    "candidata": e in et.CICLO and k >= sc.MIN_MARCADAS and pct >= sc.MIN_PERCENTIL and inf80 > med,
                    "sello": SELLO_CORTO.get(e, "fuera del ciclo")})
    tabla = pd.DataFrame(out).set_index("etapa")
    if n < MIN_NUEVAS:
        prio = "muestra insuficiente"
    else:
        c = tabla[tabla.candidata]
        if len(c):
            prio = c["exceso ★"].idxmax()
        else:
            f = tabla[~tabla.ciclo & (tabla.percentil >= sc.MIN_PERCENTIL)]
            prio = "fuera: " + f["exceso ★"].idxmax() if len(f) else "sin punto débil diferencial"
    return {"n": n, "tabla": tabla, "prioridad": prio, "oraciones": orac}


def ejemplo(ciudad="Madrid", restaurante="El_Viajero"):
    """Reseñas reales de un local con menos de 100 reseñas (no entra en los pares): sirve de demo."""
    r = datos.cargar_resenas(ciudad, ["restaurante", "texto", "fecha"])
    return r[r.restaurante == restaurante].sort_values("fecha").texto.tolist()


# ------------------------------------------------------------------ Comparar modelos en vivo
_LSTM = None


def probar_oraciones(oraciones):
    """Qué etapas marca cada modelo (regex, LogReg, LSTM) en cada oración: 1 = marca."""
    global _LSTM
    from . import modelos
    if _LSTM is None:
        _LSTM = modelos.cargar_red("LSTM_borrado")
    rx = et.etiquetas_regex(pd.Series(oraciones)).values
    lr = _logreg(list(oraciones))
    ls = (modelos.proba_red_ordenada(_LSTM[0], _LSTM[1], list(oraciones)) >= 0.5).astype(np.int8)
    filas = []
    for i, o in enumerate(oraciones):
        for nombre, m in [("Regex (léxico)", rx), ("Regresión logística", lr), ("LSTM", ls)]:
            marcadas = [NOMBRE_CORTO[e] for j, e in enumerate(et.ETIQUETAS) if m[i, j]]
            filas.append({"oración": o, "modelo": nombre, "detecta": ", ".join(marcadas) or "—"})
    return pd.DataFrame(filas)


def escalera():
    """La comparación de modelos en el gold de Barcelona (resultados del notebook 02)."""
    import json
    from . import RAIZ
    h = json.loads((RAIZ / "resultados" / "hallazgos_06.json").read_text())
    orden = ["Regex v1", "Naive Bayes", "LogReg", "MLP", "RNN", "LSTM"]
    nombres = {"Regex v1": "Regex (léxico)", "LogReg": "Regresión logística", "MLP": "MLP + fastText"}
    t = pd.DataFrame([{"Modelo": nombres.get(k, k), "F1 ciclo": h["escalera"][k]["F1 ciclo"],
                       "Precisión atención": h["escalera"][k]["precisión atención"],
                       "Recall paráfrasis": h["escalera"][k]["recall paráfrasis"]} for k in orden])
    return t, h
