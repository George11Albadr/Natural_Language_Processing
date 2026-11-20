"""Tokenizadores y léxico regex de las etapas del ciclo de servicio.

ETAPAS es la misma celda del avance (sección 4.4). El léxico revisado (lexico_v2) del
notebook 03 se agrega aquí sin modificar el original, para poder reproducir el avance.
"""
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- Tokenizadores
# Tokens alfabéticos; conserva tildes, ñ, ç y apóstrofes internos (Bal d'Onsera queda como una palabra)
_TOKEN_ES = re.compile(r"[a-záéíóúüñç]+(?:['’][a-záéíóúüñç]+)*")


def tokenizar(texto):
    return _TOKEN_ES.findall(texto.lower())


# Palabras que NLTK trata como stopwords pero que cargan polaridad ("no", "nunca", "muy").
# Quitarlas de las stopwords fue la corrección del avance: F1 de polaridad 0.811 → 0.837.
NEGADORES_ES = {"no", "ni", "nunca", "jamás", "jamas", "tampoco", "sin", "pero", "nada", "poco", "mal", "muy"}
NEGADORES_EN = {"no", "not", "nor", "never", "but", "very", "too", "don't", "didn't",
                "wasn't", "weren't", "isn't", "aren't", "won't", "couldn't"}


def _stopwords(idioma):
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words(idioma))


def tokenizador_es(quitar_negadores_de_stop=True):
    """Tokenizador del pipeline final en español: stopwords de NLTK menos los negadores."""
    stop = _stopwords("spanish") - (NEGADORES_ES if quitar_negadores_de_stop else set())
    return lambda t: [w for w in tokenizar(t) if w not in stop]


_TOKEN_EN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenizador_en():
    stop = _stopwords("english") - NEGADORES_EN
    return lambda t: [w for w in _TOKEN_EN.findall(t.lower()) if w not in stop]


# ---------------------------------------------------------------- SemEval ES: PMI y palabras semilla
def pmi_bigramas(textos_tok, min_frec=8):
    """PMI = log2 P(a,b) / (P(a) P(b)) de bigramas contiguos con frecuencia ≥ min_frec."""
    import collections
    import math
    uni = collections.Counter(w for s in textos_tok for w in s)
    bi = collections.Counter(b for s in textos_tok for b in zip(s, s[1:]))
    N = sum(uni.values())
    filas = [(f"{a} {b}", c, math.log2((c / N) / ((uni[a] / N) * (uni[b] / N))))
             for (a, b), c in bi.items() if c >= min_frec and uni[a] >= min_frec and uni[b] >= min_frec]
    return pd.DataFrame(filas, columns=["bigrama", "frecuencia", "PMI"]).sort_values("PMI", ascending=False)


# Dimensiones gruesas de SemEval y palabras semilla escritas a mano (avance, sección 4.3)
DIM_SEMEVAL = {"FOOD#QUALITY": "Cocina", "FOOD#STYLE_OPTIONS": "Cocina", "DRINKS#QUALITY": "Cocina",
               "DRINKS#STYLE_OPTIONS": "Cocina", "SERVICE#GENERAL": "Servicio", "AMBIENCE#GENERAL": "Ambiente",
               "LOCATION#GENERAL": "Ambiente", "FOOD#PRICES": "Precio", "DRINKS#PRICES": "Precio",
               "RESTAURANT#PRICES": "Precio", "RESTAURANT#GENERAL": "General", "RESTAURANT#MISCELLANEOUS": "General"}
SEMILLAS = {
    "Cocina":   ["comida", "plato", "sabor", "cocina", "carne", "pescado", "postre", "menú", "carta", "ensalada"],
    "Servicio": ["servicio", "camarero", "atención", "camareros", "trato", "espera", "personal", "atento", "rápido", "lento"],
    "Ambiente": ["ambiente", "local", "decoración", "música", "ruido", "terraza", "ubicación", "mesas", "espacio", "sitio"],
    "Precio":   ["precio", "precios", "caro", "barato", "cuenta", "económico", "coste", "pagar", "euros"],
}


def evaluar_semillas(sem, lexico_semillas):
    """Precisión, cobertura y F1 por dimensión: la oración se marca si contiene alguna semilla."""
    op = sem.dropna(subset=["cat"]).assign(dim=lambda d: d.cat.map(DIM_SEMEVAL))
    textos = sem.drop_duplicates("sid").set_index("sid").texto
    tok = {sid: set(tokenizar(t)) for sid, t in textos.items()}
    filas = {}
    for d, palabras in lexico_semillas.items():
        gold = set(op[op.dim == d].sid)
        pred = {sid for sid, ws in tok.items() if ws & set(palabras)}
        tp = len(pred & gold); p = tp / max(len(pred), 1); r = tp / len(gold)
        filas[d] = {"precisión": round(p, 2), "cobertura": round(r, 2), "F1": round(2 * p * r / max(p + r, 1e-9), 3)}
    return pd.DataFrame(filas).T


def expandir_semillas(kv, semillas, topn=60, max_nuevas=20, conteo_corpus=None, min_frec=3):
    """Agrega las palabras más cercanas al centroide de las semillas en un espacio de embeddings."""
    base = [w for w in semillas if w in kv]
    centro = np.mean([kv.get_vector(w, norm=True) for w in base], axis=0)
    nuevas = [w for w, _ in kv.similar_by_vector(centro, topn=topn) if w.isalpha() and w not in base]
    if conteo_corpus is not None:
        nuevas = [w for w in nuevas if conteo_corpus[w] >= min_frec]
    return base + nuevas[:max_nuevas]


# ---------------------------------------------------------------- Léxico de etapas (avance, sección 4.4)
ETAPAS = {  # etapa: {etiqueta: (tipo, expresión regular en inglés)}
    "1. Llegada y mesa": {
        "espera de mesa / reserva": ("falla", r"\b(?:wait(?:ed)? (?:for|over|more than|about) (?:a |an )?(?:table|\d+|an hour|half an hour)|queue|no table|(?:lost|no record of) (?:our|the|my) reservation)\b"),
    },
    "2. Atención en sala": {
        "trato grosero":  ("falla", r"\brude\b"),
        "ignorados":      ("falla", r"\b(?:ignored|unfriendly|unprofessional|arrogant)\b"),
        "olvidos":        ("falla", r"\b(?:forgot|forgotten)\b"),
        "amabilidad":     ("acierto", r"\bfriendly\b"),
        "atención":       ("acierto", r"\battentive\b"),
    },
    "3. Cocina → mesa": {
        "lentitud":          ("falla", r"\bslow(?:ly)?\b"),
        "espera de comida":  ("falla", r"\b(?:took (?:ages|forever|so long|too long|a long time)|waited (?:for|over)? ?(?:our|the) (?:food|order|meal|dishes|mains))\b"),
        "comida fría":       ("falla", r"\b(?:was|were|arrived|came) cold\b"),
        "pedido equivocado": ("falla", r"\bwrong (?:order|dish|food)\b"),
    },
    "4. Cobro": {
        "cobro indebido":     ("falla", r"\b(?:overcharg\w*|charged (?:us )?for|extra charge|hidden charge|charge for (?:the )?bread|service charge)\b"),
        "error en la cuenta": ("falla", r"\b(?:wrong bill|check (?:the|your) bill|mistake (?:in|on) (?:the )?bill|bill was wrong)\b"),
    },
    "Fuera del ciclo: cocina": {
        "sin sabor": ("falla", r"\b(?:bland|tasteless)\b"),
        "salado":    ("falla", r"\bsalty\b"),
        "grasoso":   ("falla", r"\b(?:greasy|oily)\b"),
        "delicioso": ("acierto", r"\bdelicious\b"),
    },
    "Fuera del ciclo: precio": {
        "caro":             ("falla", r"\b(?:overpriced|expensive|rip ?off)\b"),
        "trampa turística": ("falla", r"\btourist trap\b"),
        "calidad-precio":   ("acierto", r"\b(?:value for money|good value)\b"),
    },
    "Fuera del ciclo: local": {
        "ruido":    ("falla", r"\b(?:noisy|loud|noise)\b"),
        "suciedad": ("falla", r"\bdirty\b"),
    },
}
CICLO = [e for e in ETAPAS if e[0].isdigit()]
FUERA = [e for e in ETAPAS if e not in CICLO]

# Indicador de saturación del avance (P4)
LLENO = r"\b(?:busy|crowded|packed|full house|saturday|friday night|weekend)\b"


def info(etapas=ETAPAS):
    """Tabla término → etapa, tipo (falla/acierto) y regex."""
    return pd.DataFrame([(e, t, tipo, rx) for e, d in etapas.items() for t, (tipo, rx) in d.items()],
                        columns=["etapa", "término", "tipo", "rx"]).set_index("término")


def matriz_terminos(textos, etapas=ETAPAS):
    """Presencia (0/1) de cada término del léxico en cada texto."""
    minus = pd.Series(textos).str.lower()
    tabla = info(etapas)
    return pd.DataFrame({t: minus.str.contains(rx).astype(np.int8).values for t, rx in tabla.rx.items()},
                        index=minus.index)


def fallas_por_etapa(X, etapas=ETAPAS):
    """De la matriz de términos a una columna por etapa: 1 si menciona al menos una falla de esa etapa."""
    tabla = info(etapas)
    return pd.DataFrame({e: X[tabla.index[(tabla.etapa == e) & (tabla.tipo == "falla")]].max(axis=1)
                         for e in etapas})


# ---------------------------------------------------------------- Léxico v2 (notebook 03)
# El v1 (ETAPAS) se conserva intacto: es el escalón "regex actual" de la escalera y define el
# subconjunto de paráfrasis del gold. El v2 solo se usa para las etiquetas débiles. Cambios:
#   1. "slow" deja de marcar "slow cooked / slow-roasted ..." (falso positivo frecuente en 5★).
#   2. Seis términos nuevos que salen de patrones POS + PMI y pasan la regla del 03
#      (≥ 50 oraciones en Madrid train y ≥ 10× más frecuentes en 1-2★ que en 5★).
#   3. Guardia de negación (en etiquetas.etiquetas_regex): no cuenta "not slow", "never rude".
import copy

ETAPAS_V2 = copy.deepcopy(ETAPAS)
ETAPAS_V2["3. Cocina → mesa"]["lentitud"] = (
    "falla", r"\bslow(?:ly)?\b(?![- ]?(?:cook|roast|brais|bak|grill|smok)\w*)")
ETAPAS_V2["2. Atención en sala"].update({
    "servicio malo (v2)": ("falla", r"\b(?:poor|bad|terrible|awful|horrible|appalling|lousy|worst|dreadful|disgraceful) (?:customer )?service\b|\bservice (?:was|is|were) (?:\w+ )?(?:terrible|awful|poor|bad|horrible|appalling|dreadful|non[- ]?existent)\b"),
    "nadie atendió (v2)": ("falla", r"\b(?:no one|nobody|noone) (?:came|bothered|attended|served|took|checked|paid attention)\b|\bwait(?:ed|ing)? (?:\w+ ){0,5}to take (?:our|the|my) order\b"),
    "maleducado (v2)":    ("falla", r"\b(?:impolite|disrespectful|dismissive|condescending|unpolite|rudely|rudeness|unhelpful|unwelcoming)\b"),
})
ETAPAS_V2["4. Cobro"].update({
    "agregado a la cuenta (v2)": ("falla", r"\b(?:added|included|appeared) (?:to|on|in) (?:the|our) (?:bill|check)\b|\bcharged (?:us )?(?:extra|twice|more)\b"),
})
ETAPAS_V2["Fuera del ciclo: cocina"].update({
    "comida mala (v2)": ("falla", r"\b(?:bad|terrible|awful|horrible|poor|mediocre|disgusting) (?:quality )?(?:food|meal|dish(?:es)?)\b|\bfood (?:was|is) (?:\w+ )?(?:terrible|awful|bad|horrible|mediocre|inedible|disgusting|poor)\b|\binedible\b|\b(?:poor|low) quality\b"),
})
ETAPAS_V2["Fuera del ciclo: precio"].update({
    "tirar el dinero (v2)": ("falla", r"\b(?:waste|save) (?:of )?(?:your )?money\b|\bhigh prices?\b|\bnot worth (?:the|it)\b"),
})

# Candidatos que NO pasaron la regla (se documentan en el 03)
DESCARTADOS_V2 = {
    "apurados": r"\b(?:rushed|hurried) (?:us|out|through)\b|\bfelt rushed\b",
    "rechazo en la puerta": r"\b(?:turned (?:us )?away|refused (?:us )?(?:a )?table|(?:didn't|did not|couldn't|could not) find (?:our|the) (?:reservation|booking)|lost (?:our|the) booking|despite (?:our|having a|a) (?:reservation|booking))\b",
    "la comida tardó X minutos": r"\b(?:food|meal|dishes|order|mains) (?:took|came after|arrived after) (?:\w+ ){0,2}(?:hour|minutes|ages)\b|\b(?:\d+|forty|thirty|fifty) minutes? (?:for|to get) (?:our|the) (?:food|order|mains|meal)\b",
}

# Negador en las 3 palabras previas al término: "not slow", "never felt rushed", "wasn't rude at all"
NEGACION_PREVIA = re.compile(r"\b(?:not|never|no|nor|without|wasn't|weren't|isn't|aren't|didn't|don't|n't)\b(?:\W+\w+){0,3}\W*$")


def negado(texto, inicio):
    return bool(NEGACION_PREVIA.search(texto[max(0, inicio - 40):inicio]))


# ---------------------------------------------------------------- Candidatos con POS + PMI
# Patrones de frase sobre etiquetas Penn Treebank (nltk.pos_tag), con RegexpParser:
#   AN: adjetivo + sustantivo           ("terrible service", "cold food")
#   VN: verbo + (partícula) + sustantivo ("take our order", "waste your money")
#   NV: sustantivo + verbo + adjetivo    ("service was awful", "food was average")
GRAMATICA_POS = r"""
 AN: {<RB.*>?<JJ.*><NN.*>+}
 VN: {<VB.*><RP|IN|DT|PRP\$>*<NN.*>+}
 NV: {<NN.*><VBD|VBZ|VBP><RB.*>?<JJ.*|VBN>}
"""


def frases_pos(textos):
    """Por cada texto, el conjunto de frases (patrón, frase) que encuentra el chunker."""
    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    gram = nltk.RegexpParser(GRAMATICA_POS)
    out = []
    for t in textos:
        arbol = gram.parse(nltk.pos_tag(nltk.word_tokenize(t.lower())))
        out.append({(st.label(), " ".join(w for w, _ in st.leaves()))
                    for st in arbol.subtrees(lambda s: s.label() in ("AN", "VN", "NV"))})
    return out


def pmi_clase(frases_a, frases_b, min_frec=12):
    """PMI de cada frase con la clase A (muestras de igual tamaño): log2 P(A | frase) / P(A).

    Con clases balanceadas P(A) = 0.5, así que el máximo es 1 (la frase solo aparece en A).
    """
    import collections
    import math
    ca = collections.Counter(x for s in frases_a for x in s)
    cb = collections.Counter(x for s in frases_b for x in s)
    filas = [(p, f, c, cb[(p, f)], math.log2((c / (c + cb[(p, f)])) / 0.5))
             for (p, f), c in ca.items() if c >= min_frec]
    return (pd.DataFrame(filas, columns=["patrón", "frase", "n en 1-2★", "n en 5★", "PMI"])
            .sort_values(["PMI", "n en 1-2★"], ascending=False).reset_index(drop=True))
