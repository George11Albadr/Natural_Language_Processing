"""Validación temporal (notebook 09): ¿el scorecard hasta 2017 predice y apunta en 2018-2019?

Todo lo que se usa para predecir se estima SOLO con reseñas hasta 2017: tasas, β de efectos fijos y el
modelo Ridge. 2020 se excluye. Solo entran restaurantes con >= 50 reseñas hasta 2017 y >= 30 en 2018-2019,
lo que introduce un sesgo de supervivencia (los que cerraron no están).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from . import costos, datos, etiquetas as et, lexico, scorecard as sc

CORTE, FIN = 2017, 2019
MIN_ANTES, MIN_DESPUES = 50, 30


def preparar(ciudad, fuentes=("regex_v1", "LogReg_borrado")):
    """Reseñas con año, periodo y, por cada fuente, la falla por etapa (máximo sobre sus oraciones)."""
    r = datos.cargar_resenas(ciudad, ["review_id", "restaurante", "estrellas", "fecha", "texto"])
    r = r[r.fecha.dt.year <= FIN].copy()
    r["periodo"] = np.where(r.fecha.dt.year <= CORTE, "antes", "despues")
    n = r.groupby(["restaurante", "periodo"]).size().unstack(fill_value=0)
    muestra = n.index[(n.get("antes", 0) >= MIN_ANTES) & (n.get("despues", 0) >= MIN_DESPUES)]
    Y = {}
    for f in fuentes:
        p = pd.read_parquet(sc.ruta_pred(f, ciudad)).groupby("review_id")[et.ETIQUETAS].max()
        Y[f] = r[["review_id"]].join(p, on="review_id")[et.ETIQUETAS].fillna(0).astype(float).set_index(r.index)
    return r, Y, muestra


def beta_hasta(r, Yf):
    """β de efectos fijos por restaurante, solo con reseñas hasta 2017 (de toda la ciudad)."""
    a = (r.periodo == "antes").values
    cod = np.unique(r.restaurante.values[a], return_inverse=True)[1]
    b = np.linalg.lstsq(costos.centrar(Yf.values[a], cod), costos.centrar(r.estrellas.values[a].astype(float), cod), rcond=None)[0]
    return pd.Series(b, index=et.ETIQUETAS)


def promedio_contraido(est, grupos):
    """Promedio de estrellas por restaurante contraído hacia la media (normal-normal, momentos)."""
    g = est.groupby(grupos)
    m, v, n = g.mean(), g.var(ddof=1), g.size()
    mu = m.mean()
    var_w = v.mean()
    tau2 = max(m.var() - (var_w / n).mean(), 1e-6)
    peso = tau2 / (tau2 + var_w / n)
    return mu + peso * (m - mu)


def tablas(ciudad, fuentes=("regex_v1", "LogReg_borrado")):
    """Una fila por restaurante de la muestra: base, resultado y features de cada fuente (hasta 2017)."""
    r, Y, muestra = preparar(ciudad, fuentes)
    antes = r[(r.periodo == "antes") & r.restaurante.isin(muestra)]
    desp = r[(r.periodo == "despues") & r.restaurante.isin(muestra)]
    base = pd.DataFrame({"prom_contraido": promedio_contraido(antes.estrellas.astype(float), antes.restaurante),
                         "log_n": np.log(antes.groupby("restaurante").size())})
    y = desp.groupby("restaurante").estrellas.mean().loc[base.index]
    feats, prior = {}, {}
    for f in fuentes:
        beta = beta_hasta(r, Y[f])
        k_a = Y[f].loc[antes.index].groupby(antes.restaurante.values).sum().loc[base.index]
        n_a = antes.groupby("restaurante").size().loc[base.index]
        k_d = Y[f].loc[desp.index].groupby(desp.restaurante.values).sum().loc[base.index]
        n_d = desp.groupby("restaurante").size().loc[base.index]
        tasa = pd.DataFrame({e: (k_a[e] + sc.prior_beta(k_a[e], n_a)[0]) / (n_a + sum(sc.prior_beta(k_a[e], n_a)))
                             for e in et.ETIQUETAS})
        feats[f] = (tasa * beta).add_prefix("en_juego_")
        prior[f] = {"beta_hasta_2017": beta, "prio_antes": sc.tabla_prioridad(k_a, n_a, beta)[0],
                    "prio_despues": sc.tabla_prioridad(k_d, n_d, beta)[0]}
    return {"ciudad": ciudad, "base": base, "y": y, "feats": feats, "prior": prior, "antes": antes, "n": len(base)}


def feature_ridge(t, folds=5, semilla=0, max_train=120_000):
    """Predicción Ridge TF-IDF de las estrellas, promediada por restaurante, con validación cruzada por
    restaurante: el valor de cada restaurante sale de un Ridge que no vio sus reseñas (todas hasta 2017)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    antes, rest = t["antes"], t["base"].index.values
    out = pd.Series(np.nan, index=t["base"].index)
    for i_tr, i_te in KFold(folds, shuffle=True, random_state=semilla).split(rest):
        tr = antes[antes.restaurante.isin(rest[i_tr])]
        if len(tr) > max_train:
            tr = tr.sample(max_train, random_state=semilla)
        te = antes[antes.restaurante.isin(rest[i_te])]
        vec = TfidfVectorizer(tokenizer=lexico.tokenizador_en(), lowercase=False, token_pattern=None,
                              ngram_range=(1, 2), min_df=20, max_features=30_000)
        mod = Ridge(alpha=3.0).fit(vec.fit_transform(tr.texto), tr.estrellas)
        pred = pd.Series(np.clip(mod.predict(vec.transform(te.texto)), 1, 5), index=te.index)
        out.loc[rest[i_te]] = pred.groupby(te.restaurante.values).mean().loc[rest[i_te]].values
    return out.to_frame("ridge_pred")


def delta_r2(base, extra, y, folds=5, reps=1000, semilla=0):
    """ΔR² fuera de muestra de (base + extra) sobre base, con KFold por restaurante e IC bootstrap."""
    X0, X1, yv = base.values, pd.concat([base, extra], axis=1).values, y.values
    p0, p1 = np.zeros(len(yv)), np.zeros(len(yv))
    for i_tr, i_te in KFold(folds, shuffle=True, random_state=semilla).split(X0):
        p0[i_te] = LinearRegression().fit(X0[i_tr], yv[i_tr]).predict(X0[i_te])
        p1[i_te] = LinearRegression().fit(X1[i_tr], yv[i_tr]).predict(X1[i_te])
    r2 = lambda p, ix: 1 - ((yv[ix] - p[ix]) ** 2).sum() / ((yv[ix] - yv[ix].mean()) ** 2).sum()
    todo = np.arange(len(yv))
    rng = np.random.default_rng(semilla)
    boot = [r2(p1, ix) - r2(p0, ix) for ix in (rng.integers(0, len(yv), len(yv)) for _ in range(reps))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"R2 base": r2(p0, todo), "R2 con fuente": r2(p1, todo), "ΔR2": r2(p1, todo) - r2(p0, todo), "IC inf": lo, "IC sup": hi}


def coincidencia(prio_antes, prio_despues, perms=1000, semilla=0):
    """¿La etapa prioritaria hasta 2017 se repite en 2018-2019? Solo restaurantes con prioridad del ciclo antes.
    Se compara con una permutación (barajar las prioridades de 2018-2019) y con "siempre atención"."""
    sub = prio_antes.index[prio_antes.isin(et.CICLO)]
    a, d = prio_antes.loc[sub].values, prio_despues.loc[sub].values
    obs = (a == d).mean() if len(sub) else np.nan
    rng = np.random.default_rng(semilla)
    todos_d = prio_despues.values
    nulo = [(a == rng.choice(todos_d, len(a), replace=False)).mean() for _ in range(perms)] if len(sub) else [np.nan]
    return {"restaurantes con prioridad del ciclo (antes)": len(sub), "coincidencia": obs,
            "permutación media": float(np.mean(nulo)), "permutación p95": float(np.percentile(nulo, 95)),
            "siempre atención": (d == "atencion").mean() if len(sub) else np.nan}
