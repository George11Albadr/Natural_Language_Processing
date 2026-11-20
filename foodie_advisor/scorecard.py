"""El producto: un scorecard del ciclo de servicio por restaurante, para el gerente.

Para cada etapa:
  - % de reseñas con falla (una reseña tiene falla si alguna de sus oraciones la tiene),
    contraído hacia la media de los pares (empírico bayesiano beta-binomial), con IC 95%;
  - percentil frente a los pares de la misma ciudad con >= 100 reseñas;
  - estrellas en juego = tasa × costo por falla (β de efectos fijos de la ciudad);
  - tendencia anual;
  - 3 citas textuales como evidencia.

Corrección por error de medición (Rogan-Gladen): con sensibilidad (recall) y especificidad
medidas en el gold, la prevalencia verdadera es (p_obs + esp - 1) / (sens + esp - 1). El gold mide
oraciones, así que se corrige la tasa de oraciones del restaurante y la de reseñas se escala por
el mismo factor. Es una aproximación y así se reporta. Con el regex de relleno no se corrige.
"""
import numpy as np
import pandas as pd

from . import PROCESADO, datos, etiquetas as et, lexico

MIN_PARES = 100
MIN_RECALL_RG = 0.3          # por debajo, Rogan-Gladen amplifica el ruido: no se corrige


# ---------------------------------------------------------------- Predicciones por oración
def ruta_pred(fuente, ciudad):
    return PROCESADO / f"pred_{fuente}_{ciudad}.parquet"


def predecir_regex(ciudad, etapas=lexico.ETAPAS, fuente="regex_v1", forzar=False):
    """Predicciones 0/1 por oración con el regex (relleno hasta que el 06 elija el modelo)."""
    ruta = ruta_pred(fuente, ciudad)
    if ruta.exists() and not forzar:
        return pd.read_parquet(ruta)
    o = datos.cargar_oraciones(ciudad)
    pred = pd.concat([o[["review_id", "n"]], et.etiquetas_regex(o.oracion, etapas).set_axis(o.index)], axis=1)
    pred.to_parquet(ruta, index=False)
    return pred


# ---------------------------------------------------------------- Contexto de una ciudad
def rogan_gladen(p_obs, sens, esp):
    return np.clip((p_obs + esp - 1) / max(sens + esp - 1, 1e-6), 0, 1)


def prior_beta(k, n):
    """α, β de una beta ajustada por momentos a las tasas k/n de los pares (empírico bayesiano)."""
    p = k / n
    m, v = p.mean(), p.var()
    v_binom = (m * (1 - m) / n).mean()                  # parte de la varianza que es solo ruido de muestreo
    v_real = max(v - v_binom, 1e-6)
    s = max(m * (1 - m) / v_real - 1, 1.0)
    return m * s, (1 - m) * s


def contexto_ciudad(ciudad, fuente, costo, medicion=None, secundaria=None, notas=None, rotulo="% estimado"):
    """Todo lo que el scorecard necesita de una ciudad.

    costo: Serie etapa → β (estrellas por reseña con falla, de costos.costos_ciudad).
    medicion: dict etapa → (sens, esp) del gold, para Rogan-Gladen; None = sin corrección.
    secundaria: (nombre, fuente) de una señal secundaria que se muestra cruda, sin percentil.
    notas: dict etapa → texto (p. ej. "no validada: 6 positivos en el gold").
    rotulo: cómo se llama la tasa principal en la tabla y la figura.
    """
    pred = pd.read_parquet(ruta_pred(fuente, ciudad))
    res = datos.cargar_resenas(ciudad, ["review_id", "restaurante", "estrellas", "fecha"])
    por_resena = pred.groupby("review_id")[et.ETIQUETAS].max()
    res = res.join(por_resena, on="review_id").fillna({e: 0 for e in et.ETIQUETAS})
    n = res.groupby("restaurante").size()
    pares = n[n >= MIN_PARES].index
    k = res[res.restaurante.isin(pares)].groupby("restaurante")[et.ETIQUETAS].sum()
    nn = n.loc[pares]
    priors = {e: prior_beta(k[e], nn) for e in et.ETIQUETAS}
    shr = pd.DataFrame({e: (k[e] + priors[e][0]) / (nn + sum(priors[e])) for e in et.ETIQUETAS})
    sec = None
    if secundaria:
        p2 = pd.read_parquet(ruta_pred(secundaria[1], ciudad)).groupby("review_id")[et.ETIQUETAS].max()
        sec = (secundaria[0], res[["review_id", "restaurante"]].join(p2, on="review_id").fillna({e: 0 for e in et.ETIQUETAS}))
    prioridad, largo = tabla_prioridad(k, nn, costo)
    return {"ciudad": ciudad, "fuente": fuente, "secundaria": sec, "notas": notas or {}, "rotulo": rotulo,
            "prioridad": prioridad, "prioridad_largo": largo,
            "pred": pred, "res": res, "pares": pares, "k": k, "n": nn,
            "priors": priors, "tasa_contraida": shr, "costo": costo, "medicion": medicion,
            "prom_pares": res[res.restaurante.isin(pares)].groupby("restaurante").estrellas.mean()}


# ---------------------------------------------------------------- Scorecard
def scorecard(restaurante, ctx, n_citas=3, semilla=0):
    res, pred = ctx["res"], ctx["pred"]
    r = res[res.restaurante == restaurante]
    assert len(r) >= MIN_PARES, f"{restaurante} tiene {len(r)} reseñas; el scorecard pide {MIN_PARES} o más"
    filas, avisos = [], []
    for e in et.ETIQUETAS:
        k, n = int(r[e].sum()), len(r)
        a, b = ctx["priors"][e]
        tasa = (k + a) / (n + a + b)
        muestras = np.random.default_rng(semilla).beta(k + a, n - k + b, 20_000)
        lo, hi = np.percentile(muestras, [2.5, 97.5])
        factor, corregida = 1.0, False
        if ctx["medicion"] and e in ctx["medicion"]:
            sens, esp = ctx["medicion"][e]
            if sens < MIN_RECALL_RG:
                avisos.append(f"{e}: recall en el gold {sens:.2f} < {MIN_RECALL_RG}; no se corrige (amplificaría el ruido)")
            else:
                p_obs = pred.loc[pred.review_id.isin(r.review_id), e].mean()
                if p_obs > 0:
                    factor, corregida = rogan_gladen(p_obs, sens, esp) / p_obs, True
        tasa, lo, hi = (float(np.clip(x * factor, 0, 1)) for x in (tasa, lo, hi))
        filas.append({"etapa": e, "nombre": et.NOMBRE[e], "reseñas con falla": k, "% observado": k / n * 100,
                      "% estimado": tasa * 100, "corregida (aprox.)": corregida, "IC inf %": lo * 100, "IC sup %": hi * 100,
                      "mediana pares %": ctx["tasa_contraida"][e].median() * 100,
                      "percentil": (ctx["tasa_contraida"][e] < tasa / factor).mean() * 100,
                      "costo por falla ★": ctx["costo"][e], "estrellas en juego": tasa * ctx["costo"][e],
                      "nota": ctx["notas"].get(e, "")})
        if ctx["secundaria"]:
            nombre_sec, rs = ctx["secundaria"]
            filas[-1][f"% {nombre_sec} (señal amplia, no validada)"] = rs.loc[rs.restaurante == restaurante, e].mean() * 100
    tabla = pd.DataFrame(filas).set_index("etapa")
    lg = ctx["prioridad_largo"]; lg = lg[lg.restaurante == restaurante].set_index("etapa")
    tabla["exceso ★ (vs mediana)"] = lg["exceso"]; tabla["inf80 %"] = lg["inf80"] * 100
    tabla["candidata"] = lg["candidata"]; tabla["sello"] = [SELLOS.get(e, "") for e in tabla.index]
    tabla = tabla.rename(columns={"% estimado": ctx["rotulo"]})
    tabla["% estimado"] = tabla[ctx["rotulo"]]          # alias estable para el resto del código

    anual = r.groupby(r.fecha.dt.year)[et.CICLO].agg(["mean", "size"])
    tendencia = pd.DataFrame({e: anual[(e, "mean")] * 100 for e in et.CICLO})
    tendencia = tendencia[anual[(et.CICLO[0], "size")] >= 50]          # años con pocas reseñas son puro ruido

    # Las citas son evidencia de la acción prioritaria; si no hay prioridad, de la etapa con más estrellas en juego
    prio = ctx["prioridad"][restaurante]
    peor = prio if prio in et.ETIQUETAS else prio[7:] if prio.startswith("fuera: ") else tabla["estrellas en juego"].idxmin()
    o = datos.cargar_oraciones(ctx["ciudad"])
    o = o[o.review_id.isin(r.review_id)].merge(pred, on=["review_id", "n"])
    candidatas = o[o[peor] == 1]
    citas = candidatas.oracion.sample(min(n_citas, len(candidatas)), random_state=semilla).tolist()

    prio = ctx["prioridad"][restaurante]
    return {"restaurante": restaurante, "ciudad": ctx["ciudad"], "fuente": ctx["fuente"], "reseñas": len(r), "rotulo": ctx["rotulo"],
            "prioridad": prio, "accion": texto_prioridad(prio),
            "promedio": r.estrellas.mean(), "mediana pares ★": ctx["prom_pares"].median(), "n pares": len(ctx["pares"]),
            "tabla": tabla, "tendencia": tendencia, "peor etapa": peor, "citas": citas, "avisos": avisos,
            "en juego ciclo": tabla.loc[et.CICLO, "estrellas en juego"].sum(),
            "en juego fuera": tabla.drop(et.CICLO)["estrellas en juego"].sum()}


def figura(sc, ruta=None):
    """Una página: barras de estrellas en juego por etapa (con percentil), tendencia del ciclo y citas."""
    import matplotlib.pyplot as plt
    import textwrap
    ROJO, GRIS, AZUL = "#b5483a", "#8c8c8c", "#3b6ea5"
    t = sc["tabla"].iloc[::-1]
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.1], width_ratios=[1.35, 1], hspace=0.42, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    colores = [ROJO if e in et.CICLO else GRIS for e in t.index]
    sello = [t.loc[e, "nota"] or t.loc[e, "sello"] for e in t.index]          # la nota (con n) manda sobre el sello
    etiquetas_y = [f"{n_} [{s_}]" if s_ else n_ for n_, s_ in zip(t.nombre, sello)]
    ax.barh(etiquetas_y, t["estrellas en juego"], color=colores)
    ancho = abs(t["estrellas en juego"].min())
    for i, (_, x) in enumerate(t.iterrows()):
        marca = "*" if x["corregida (aprox.)"] else ""
        ax.text(ancho * 0.03, i, f"{x['% estimado']:.1f}%{marca} · p{x['percentil']:.0f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(-ancho * 1.05, ancho * 0.45)
    ax.axvline(0, color="black", lw=0.8)
    nota = "  · * tasa corregida (aprox.)" if t["corregida (aprox.)"].any() else ""
    ax.set(xlabel="Estrellas en juego (tasa × costo por falla)" + nota, title=f"Dónde pierde estrellas · {sc.get('rotulo', '% de reseñas')} · percentil")
    ax2 = fig.add_subplot(gs[0, 1])
    for e, c in zip(et.CICLO, ["#9db7d5", ROJO, "#e3a33b", AZUL]):
        if e in sc["tendencia"]:
            ax2.plot(sc["tendencia"].index, sc["tendencia"][e], marker="o", ms=3, label=et.NOMBRE[e], color=c)
    ax2.set(title="Ciclo de servicio por año", ylabel="% de reseñas con falla")
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.legend(fontsize=7)
    ax3 = fig.add_subplot(gs[1, :]); ax3.axis("off")
    ax3.text(0, 1.0, sc["accion"], va="top", fontsize=11.5, fontweight="bold", color=ROJO, family="DejaVu Sans")
    texto = "\n\n".join("“" + "\n".join(textwrap.wrap(c, 140)) + "”" for c in sc["citas"])
    ax3.text(0, 0.84, f"Lo que dicen sus reseñas sobre {et.NOMBRE[sc['peor etapa']]}:\n\n{texto}", va="top", fontsize=9, family="DejaVu Sans")
    fig.text(0.01, 0.005, LEYENDA, fontsize=8.5, style="italic", color=GRIS)
    fig.suptitle(f"{sc['restaurante'].replace('_', ' ')} ({sc['ciudad']}) · {sc['reseñas']:,} reseñas · {sc['promedio']:.2f}★ "
                 f"(mediana de {sc['n pares']} pares: {sc['mediana pares ★']:.2f}★) · en juego: ciclo {sc['en juego ciclo']:.2f}★, "
                 f"fuera {sc['en juego fuera']:.2f}★", fontsize=10.5)
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return fig


def predecir_modelo(ciudad, final, variante, forzar=False, bloque=400_000):
    """Predicciones 0/1 por oración del modelo elegido en el 06, para toda una ciudad (por bloques).

    final: "LSTM", "RNN", "MLP", "LogReg" o "Naive Bayes"; variante: "borrado" o "<mask>".
    """
    from . import modelos as m
    fuente = f"{final.replace(' ', '_')}_{'mask' if variante == '<mask>' else variante}"
    ruta = ruta_pred(fuente, ciudad)
    if ruta.exists() and not forzar:
        return fuente
    o = datos.cargar_oraciones(ciudad)
    if final in ("LogReg", "Naive Bayes"):
        lin = m.cargar_lineales(variante)
        mod = lin["logreg" if final == "LogReg" else "nb"]
        proba = lambda tx: mod.predict_proba(lin["vec"].transform(tx))
    else:
        base = {"MLP": "MLP_promedio_fastText", "RNN": "RNN", "LSTM": "LSTM"}[final]
        red, vocab = m.cargar_red(f"{base}_{'mask' if variante == '<mask>' else variante}")
        proba = lambda tx: m.proba_red_ordenada(red, vocab, tx)
    partes = []
    for i in range(0, len(o), bloque):
        partes.append((proba(o.oracion.iloc[i:i + bloque].tolist()) >= 0.5).astype(np.int8))
    pred = pd.concat([o[["review_id", "n"]].reset_index(drop=True),
                      pd.DataFrame(np.vstack(partes), columns=et.ETIQUETAS)], axis=1)
    pred.to_parquet(ruta, index=False)
    return fuente


# ---------------------------------------------------------------- Acción prioritaria (regla del 08)
SELLOS = {"atencion": "validada (6/6 en el gold)", "cocina_mesa": "señal débil (2/8 en el gold)",
          "cobro": "no validada", "llegada": "no validada"}
LEYENDA = "estrellas en juego = asociación, no ganancia esperada por arreglar"
MIN_MARCADAS, MIN_PERCENTIL, CRED = 5, 75, 0.80


def tabla_prioridad(k, n, beta_costo):
    """Regla de la acción prioritaria, para todos los restaurantes a la vez.

    k: DataFrame restaurante × etapa con reseñas marcadas; n: reseñas por restaurante; beta_costo: β por etapa.
    Exceso = (tasa contraída − mediana de los pares) × |β|. Candidatas: etapas del ciclo con >= 5 reseñas
    marcadas, percentil >= 75 y límite inferior del intervalo creíble al 80% por encima de la mediana.
    Prioridad: la candidata de mayor exceso. Si no hay: una etapa fuera del ciclo en percentil >= 75
    ("fuera del ciclo: X"); si tampoco, "sin punto débil diferencial".
    """
    from scipy.stats import beta as B
    out = {}
    for e in et.ETIQUETAS:
        a, b = prior_beta(k[e], n)
        shr = (k[e] + a) / (n + a + b)
        med = shr.median()
        out[e] = pd.DataFrame({"tasa": shr, "mediana": med,
                               "percentil": (shr.rank(method="min") - 1) / len(shr) * 100,
                               "inf80": B.ppf((1 - CRED) / 2, k[e] + a, n - k[e] + b),
                               "marcadas": k[e], "exceso": (shr - med) * abs(beta_costo[e])})
    largo = pd.concat(out, names=["etapa", "restaurante"]).reset_index()
    largo["candidata"] = (largo.etapa.isin(et.CICLO) & (largo.marcadas >= MIN_MARCADAS)
                          & (largo.percentil >= MIN_PERCENTIL) & (largo.inf80 > largo.mediana))
    def decidir(g):
        c = g[g.candidata]
        if len(c):
            return c.loc[c.exceso.idxmax(), "etapa"]
        f = g[~g.etapa.isin(et.CICLO) & (g.percentil >= MIN_PERCENTIL)]
        if len(f):
            return "fuera: " + f.loc[f.exceso.idxmax(), "etapa"]
        return "sin punto débil diferencial"
    prioridad = largo.groupby("restaurante")[["etapa", "candidata", "exceso", "percentil"]].apply(decidir)
    return prioridad, largo


def texto_prioridad(p):
    if p in et.CICLO:
        return f"Atacar primero: {et.NOMBRE[p]} — {SELLOS.get(p, '')}"
    if p.startswith("fuera: "):
        return f"El problema está fuera del ciclo: {et.NOMBRE[p[7:]]}"
    return "Sin punto débil diferencial frente a pares"
