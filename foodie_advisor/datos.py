"""Carga y limpieza de los dos corpus del proyecto.

- SemEval-2016 Tarea 5, restaurantes en español (XML oficial en data/).
- TripAdvisor de Kaggle (inigolopezrioboo), 6 ciudades, en inglés.

La limpieza de TripAdvisor es exactamente la del avance del 27-oct (con ella Madrid
queda en 154,819 reseñas) y se aplica igual a las 6 ciudades.
"""
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from . import DATA, PROCESADO, CIUDADES

KAGGLE_ID = "inigolopezrioboo/a-tripadvisor-dataset-for-nlp-tasks"
CACHE_KAGGLE = Path.home() / ".cache/kagglehub/datasets" / KAGGLE_ID / "versions/1"

COLUMNAS = ["restaurant_name", "rating_review", "sample", "review_id", "title_review",
            "review_full", "date", "author_id"]
MIN_PALABRAS = 5
MIN_RESENAS_REST = 20


# ---------------------------------------------------------------- SemEval ES
def limpiar_semeval(texto):
    """Quita el texto sobrante del formulario ("Escriba aquí su comentario") y las marcas <+>/<->."""
    texto = re.sub(r"Escriba aquí su comentario…?", " ", texto or "")
    return re.sub(r"^\s*<[+-]>", "", texto).strip()


def _parse_semeval(path):
    filas = []
    for rev in ET.parse(path).getroot().findall("Review"):
        for s in rev.findall(".//sentence"):
            base = dict(rid=rev.get("rid"), sid=s.get("id"), texto=limpiar_semeval(s.find("text").text))
            ops = s.findall(".//Opinion")
            if not ops:
                filas.append({**base, "target": None, "cat": None, "pol": None})
            for o in ops:
                filas.append({**base, "target": o.get("target"), "cat": o.get("category"), "pol": o.get("polarity")})
    return pd.DataFrame(filas)


def cargar_semeval():
    """Una fila por opinión (las oraciones sin opinión quedan con cat/pol vacíos)."""
    sem = pd.concat([_parse_semeval(DATA / "restaurants_train_spanish.xml").assign(split="train"),
                     _parse_semeval(DATA / "restaurants_test_spanish.xml").assign(split="test")],
                    ignore_index=True)
    return sem[sem.texto != ""].reset_index(drop=True)


# ---------------------------------------------------------------- TripAdvisor
def ruta_kaggle():
    """Usa la caché local si existe; si no, descarga con kagglehub (~2.3 GB, sin cuenta)."""
    if CACHE_KAGGLE.exists():
        return CACHE_KAGGLE
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_ID))


def leer_crudo(ciudad):
    path = ruta_kaggle() / f"{ciudad}_reviews.csv"
    if ciudad == "New_Delhi":
        # El CSV de New Delhi trae filas malformadas: el parser C falla; el de Python las salta.
        return pd.read_csv(path, usecols=COLUMNAS, dtype=str, engine="python", on_bad_lines="skip")
    # dtype=str: en algunas ciudades rating_review mezcla números y texto; se convierte después
    return pd.read_csv(path, usecols=COLUMNAS, dtype=str)


def limpiar_tripadvisor(crudo, ciudad):
    """Devuelve (reseñas limpias, tabla de pasos). Mismos filtros que el avance."""
    ta = crudo.rename(columns={"restaurant_name": "restaurante", "review_full": "texto",
                               "title_review": "titulo", "author_id": "autor", "sample": "muestra"})
    ta["estrellas"] = pd.to_numeric(ta.rating_review, errors="coerce")
    ta["fecha"] = pd.to_datetime(ta.date, format="%B %d, %Y", errors="coerce")
    ta["palabras"] = ta.texto.astype(str).str.split().str.len()

    pasos = [("Reseñas en el CSV", len(ta))]
    ta = ta.dropna(subset=["estrellas", "fecha", "texto"]);           pasos.append(("Con estrellas, fecha y texto", len(ta)))
    ta = ta[ta.estrellas.between(1, 5)];                              pasos.append(("Estrellas entre 1 y 5", len(ta)))
    ta = ta.drop_duplicates("texto");                                 pasos.append(("Sin textos duplicados", len(ta)))
    ta = ta[ta.palabras >= MIN_PALABRAS];                             pasos.append((f"Con {MIN_PALABRAS} palabras o más", len(ta)))
    n_rest = ta.groupby("restaurante").size()
    ta = ta[ta.restaurante.isin(n_rest[n_rest >= MIN_RESENAS_REST].index)].copy()
    pasos.append((f"En restaurantes con {MIN_RESENAS_REST} reseñas o más", len(ta)))

    ta["estrellas"] = ta.estrellas.astype("int8")
    ta["ciudad"] = ciudad
    # author_id viene como UID_n; lo prefijamos con la ciudad para no mezclar numeraciones entre archivos
    ta["autor"] = ciudad + ":" + ta.autor.astype(str)
    cols = ["review_id", "ciudad", "restaurante", "estrellas", "fecha", "titulo", "texto", "palabras", "autor", "muestra"]
    pasos = pd.DataFrame(pasos, columns=["Paso", ciudad]).set_index("Paso")
    return ta[cols].reset_index(drop=True), pasos


# ---------------------------------------------------------------- Oraciones
# Separador por regex: corta tras . ! ? (uno o más) seguidos de espacio, en saltos de línea,
# o donde hay 2+ espacios seguidos (en el CSV de Kaggle los saltos de párrafo quedaron como
# espacios múltiples: "...extra charge surprise on it   There are no more words...").
# Es mucho más rápido que punkt en 2.7 M reseñas, y TripAdvisor casi no trae abreviaturas.
_CORTE = re.compile(r"(?<=[.!?])\s+|\s*[\r\n]+\s*|\s{2,}")


def dividir_oraciones(texto, min_palabras=3):
    return [o.strip() for o in _CORTE.split(texto) if len(o.split()) >= min_palabras]


def oraciones_de(resenas):
    """Una fila por oración: review_id, n (posición en la reseña), oracion."""
    filas = [(rid, i, o) for rid, t in zip(resenas.review_id, resenas.texto)
             for i, o in enumerate(dividir_oraciones(t))]
    orac = pd.DataFrame(filas, columns=["review_id", "n", "oracion"])
    orac["n"] = orac.n.astype("int16")
    return orac


# ---------------------------------------------------------------- Parquet
def ruta_resenas(ciudad):
    return PROCESADO / f"{ciudad}_reseñas.parquet"


def ruta_oraciones(ciudad):
    return PROCESADO / f"{ciudad}_oraciones.parquet"


def cargar_resenas(ciudad, columnas=None):
    return pd.read_parquet(ruta_resenas(ciudad), columns=columnas)


def cargar_oraciones(ciudad, con_resena=False):
    """Oraciones de una ciudad; con_resena=True agrega restaurante, estrellas, fecha y autor."""
    orac = pd.read_parquet(ruta_oraciones(ciudad))
    if con_resena:
        meta = cargar_resenas(ciudad, ["review_id", "restaurante", "estrellas", "fecha", "autor"])
        orac = orac.merge(meta, on="review_id", how="left")
    return orac


def procesar_ciudad(ciudad, forzar=False):
    """CSV crudo → parquet de reseñas y de oraciones. Devuelve la tabla de pasos de limpieza."""
    PROCESADO.mkdir(parents=True, exist_ok=True)
    ruta_pasos = PROCESADO / f"{ciudad}_pasos.csv"
    if not forzar and ruta_resenas(ciudad).exists() and ruta_oraciones(ciudad).exists() and ruta_pasos.exists():
        return pd.read_csv(ruta_pasos, index_col=0)
    ta, pasos = limpiar_tripadvisor(leer_crudo(ciudad), ciudad)
    ta.to_parquet(ruta_resenas(ciudad), index=False)
    oraciones_de(ta).to_parquet(ruta_oraciones(ciudad), index=False)
    pasos.to_csv(ruta_pasos)
    return pasos


def procesar_todo(forzar=False):
    pasos = [procesar_ciudad(c, forzar) for c in CIUDADES]
    sem = cargar_semeval()
    sem.to_parquet(PROCESADO / "semeval.parquet", index=False)
    return pd.concat(pasos, axis=1)


# ---------------------------------------------------------------- Citas
def verificar_citas(citas, fuentes=("Madrid",)):
    """Comprueba que cada cita exista literalmente en el corpus. Devuelve {cita: fuente}.

    fuentes: ciudades de TripAdvisor y/o "semeval". Falla con AssertionError si alguna no
    aparece: en este proyecto solo se cita texto que existe, nunca un ejemplo inventado.
    """
    textos = {}
    for f in fuentes:
        if f == "semeval":
            textos[f] = pd.read_parquet(PROCESADO / "semeval.parquet", columns=["texto"]).texto.drop_duplicates()
        else:
            textos[f] = cargar_resenas(f, ["texto"]).texto
    encontradas, faltan = {}, []
    for c in citas:
        donde = next((f for f, tx in textos.items() if tx.str.contains(c, regex=False).any()), None)
        (encontradas.__setitem__(c, donde) if donde else faltan.append(c))
    assert not faltan, f"Citas que no existen literalmente en {list(fuentes)}: {faltan}"
    return encontradas
