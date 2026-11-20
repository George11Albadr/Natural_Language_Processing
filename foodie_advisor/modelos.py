"""Modelos del proyecto.

1. Línea clásica en español (SemEval): polaridad con TF-IDF + NB / LogReg.
2. Escalera del detector de fallas (TripAdvisor, inglés): mayoritaria → regex → NB →
   LogReg → MLP sobre fastText → RNN vs LSTM.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

from . import lexico


# ---------------------------------------------------------------- SemEval: polaridad
def oraciones_polaridad(sem):
    """Oraciones con una sola polaridad (positiva o negativa); y = 1 si es negativa."""
    op = sem.dropna(subset=["cat"])
    pura = (op[op.pol.isin(["positive", "negative"])]
            .groupby(["split", "sid"]).agg(rid=("rid", "first"), texto=("texto", "first"), pols=("pol", set))
            .reset_index())
    pura = pura[pura.pols.apply(len) == 1].copy()
    pura["y"] = pura.pols.apply(lambda s: int("negative" in s))
    return pura.drop(columns="pols")


def variantes_stopwords(sem):
    """F1 macro (test oficial) del avance, sección 4.1: ¿qué pasa con los negadores al quitar stopwords?"""
    from sklearn.feature_extraction.text import CountVectorizer
    pura = oraciones_polaridad(sem)
    tr, te = pura[pura.split == "train"], pura[pura.split == "test"]
    base = lexico._stopwords("spanish")
    variantes = {"Stopwords sin negadores (pipeline final)": base - lexico.NEGADORES_ES,
                 "Stopwords completas de NLTK": base, "Sin quitar stopwords": set()}
    res = []
    for nombre, stop in variantes.items():
        tok = (lambda st: (lambda t: [w for w in lexico.tokenizar(t) if w not in st]))(stop)
        fila = {"Variante": nombre}
        for rep, V in [("TF-IDF", TfidfVectorizer), ("BoW", CountVectorizer)]:
            vec = V(tokenizer=tok, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2)
            A, B = vec.fit_transform(tr.texto), vec.transform(te.texto)
            for mod, M in [("LogReg", LogisticRegression(max_iter=2000, class_weight="balanced")),
                           ("NB", MultinomialNB(alpha=0.5))]:
                fila[f"{rep} + {mod}"] = round(f1_score(te.y, M.fit(A, tr.y).predict(B), average="macro"), 3)
        res.append(fila)
    return pd.DataFrame(res).set_index("Variante")


def polaridad_semeval(sem, Cs=(0.1, 0.3, 1, 3, 10, 30)):
    """Pipeline final del avance: TF-IDF (1-2 gramas, stopwords sin negadores), C por CV agrupada por reseña.

    Devuelve dict con F1 macro en el test oficial de LogReg y NB, el C elegido y los objetos ajustados.
    """
    pura = oraciones_polaridad(sem)
    tr, te = pura[pura.split == "train"], pura[pura.split == "test"]
    vec = TfidfVectorizer(tokenizer=lexico.tokenizador_es(), lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=2)
    A, B = vec.fit_transform(tr.texto), vec.transform(te.texto)
    barrido = {C: cross_val_score(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"), A, tr.y,
                                  groups=tr.rid, cv=GroupKFold(n_splits=5), scoring="f1_macro").mean()
               for C in Cs}
    mejor_C = max(barrido, key=barrido.get)
    lr = LogisticRegression(C=mejor_C, max_iter=2000, class_weight="balanced").fit(A, tr.y)
    nb = MultinomialNB(alpha=0.5).fit(A, tr.y)
    return {
        "f1_logreg": f1_score(te.y, lr.predict(B), average="macro"),
        "f1_nb": f1_score(te.y, nb.predict(B), average="macro"),
        "C": mejor_C, "barrido": barrido, "vec": vec, "logreg": lr, "nb": nb, "train": tr, "test": te,
    }


# ================================================================ Detector de fallas (inglés)
# Piezas: nn.Embedding, nn.RNN / nn.LSTM, nn.Linear, BCEWithLogitsLoss, Adam, bucle
# zero_grad → forward → loss → backward → step, y FastText de gensim.
import re as _re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from . import MODELOS
from .etiquetas import MASK

PAD, UNK = "<pad>", "<unk>"
MAX_LARGO = 60
_TOKEN_NN = _re.compile(r"<mask>|[a-z]+(?:'[a-z]+)?|\d+")


def tokens(texto):
    """Tokenizador de las redes: minúsculas, sin quitar stopwords (el orden y el 'not' importan)."""
    return _TOKEN_NN.findall(texto.lower())


def dispositivo():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- FastText
FT_CONFIG = dict(vector_size=100, window=5, min_count=2, sg=1, negative=5, epochs=10, seed=42, workers=1)


def entrenar_fasttext(oraciones_tok, nombre="fasttext_madrid_train", **cambios):
    """FastText (skip-gram, d = 100, ventana 5), sobre oraciones ya tokenizadas."""
    from gensim.models import FastText
    ft = FastText(sentences=oraciones_tok, **{**FT_CONFIG, **cambios})
    MODELOS.mkdir(parents=True, exist_ok=True)
    ft.save(str(MODELOS / f"{nombre}.model"))
    return ft


def cargar_fasttext(nombre="fasttext_madrid_train"):
    from gensim.models import FastText
    return FastText.load(str(MODELOS / f"{nombre}.model"))


# ---------------------------------------------------------------- Vocabulario y embeddings
def vocabulario(oraciones_tok, min_frec=2):
    """palabra → índice. 0 = <pad>, 1 = <unk>, 2 = <mask>; el resto, por frecuencia."""
    from collections import Counter
    frec = Counter(w for s in oraciones_tok for w in s)
    v = {PAD: 0, UNK: 1, MASK: 2}
    for w, c in frec.most_common():
        if c >= min_frec and w not in v:
            v[w] = len(v)
    return v


def matriz_embeddings(vocab, ft, semilla=0):
    """Fila i = vector fastText de la palabra i. <pad> = ceros; <unk> = aleatorio pequeño."""
    g = torch.Generator().manual_seed(semilla)
    W = torch.zeros(len(vocab), ft.wv.vector_size)
    for w, i in vocab.items():
        if w == UNK:
            W[i] = torch.randn(ft.wv.vector_size, generator=g) * 0.1
        elif w != PAD:
            W[i] = torch.tensor(ft.wv[w])
    return W


def a_indices(textos, vocab):
    return [[vocab.get(w, 1) for w in tokens(t)][:MAX_LARGO] or [1] for t in textos]


# ---------------------------------------------------------------- Dataset y lotes
class OracionesDataset(Dataset):
    def __init__(self, indices, Y=None, pesos=None):
        self.x = [torch.tensor(s, dtype=torch.long) for s in indices]
        n = len(self.x)
        self.y = torch.tensor(np.zeros((n, 1)) if Y is None else np.asarray(Y), dtype=torch.float32)
        self.w = torch.tensor(np.ones(n) if pesos is None else np.asarray(pesos), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.w[i]


def collate(batch):
    seqs, ys, ws = zip(*batch)
    largos = torch.tensor([len(s) for s in seqs])
    x = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)   # relleno a la derecha
    return x, largos, torch.stack(ys), torch.stack(ws)


# ---------------------------------------------------------------- Modelos
class MLPPromedio(nn.Module):
    """Promedio de los vectores fastText de la oración → capa oculta → 7 salidas. Ignora el orden."""
    def __init__(self, pesos, n_salidas, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pesos, freeze=True, padding_idx=0)
        self.fc1 = nn.Linear(pesos.shape[1], hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_salidas)

    def forward(self, x, largos):
        suma = self.embedding(x).sum(dim=1)                      # el <pad> es un vector de ceros
        promedio = suma / largos.unsqueeze(1).to(suma.device, suma.dtype)
        return self.fc2(self.act(self.fc1(promedio)))


class Recurrente(nn.Module):
    """Red recurrente que clasifica oraciones.

    El nn.Linear se aplica al estado oculto de la ÚLTIMA palabra real, que resume la
    oración leída de izquierda a derecha. celda="lstm" usa nn.LSTM; celda="rnn", nn.RNN.
    """
    def __init__(self, pesos, n_salidas, hidden_dim=128, celda="lstm"):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pesos, freeze=False, padding_idx=0)
        Celda = nn.LSTM if celda == "lstm" else nn.RNN
        self.rnn = Celda(pesos.shape[1], hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_salidas)

    def forward(self, x, largos):
        salidas, _ = self.rnn(self.embedding(x))                # (lote, largo, hidden)
        filas = torch.arange(x.shape[0], device=x.device)
        ultimo = salidas[filas, largos.to(x.device) - 1]        # estado tras la última palabra real
        return self.fc(ultimo)                                  # logits; la sigmoide va en la pérdida


# ---------------------------------------------------------------- Entrenamiento
def predecir_proba(modelo, loader, disp=None):
    disp = disp or dispositivo()
    modelo.eval()
    probs = []
    with torch.no_grad():
        for x, largos, _, _ in loader:
            probs.append(torch.sigmoid(modelo(x.to(disp), largos)).cpu())
    return torch.cat(probs).numpy()


def entrenar(modelo, train_loader, dev_loader=None, Y_dev=None, epocas=5, lr=0.002, semilla=42, verbose=True,
             pos_weight=None, evaluar=None):
    """Entrena con BCEWithLogitsLoss ponderada por oración y Adam. Si hay dev, guarda la mejor época.

    pos_weight: peso de los positivos por etiqueta (n_neg / n_pos); evita que el modelo prediga
    "ninguna falla" siempre, que es lo que minimiza la pérdida cuando los positivos son raros.
    evaluar: función P_dev → número para elegir la época; por defecto, F1 macro de todas las etiquetas.
    Devuelve el historial (pérdida de train y métrica de dev por época).
    """
    from sklearn.metrics import f1_score
    torch.manual_seed(semilla)
    disp = dispositivo()
    modelo.to(disp)
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32, device=disp)
    crit = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)   # una pérdida por oración y etiqueta
    evaluar = evaluar or (lambda P: f1_score(Y_dev, P >= 0.5, average="macro", zero_division=0))
    opt = optim.Adam(modelo.parameters(), lr=lr)
    historial, mejor, mejor_estado = [], -1.0, None

    for ep in range(epocas):
        modelo.train()
        total, n = 0.0, 0
        for x, largos, y, w in train_loader:
            x, y, w = x.to(disp), y.to(disp), w.to(disp)
            opt.zero_grad()                                     # 1. limpiar gradientes
            logits = modelo(x, largos)                          # 2. forward
            perdida = (crit(logits, y).mean(dim=1) * w).sum() / w.sum()   # 3. pérdida (anotadas pesan ×5)
            perdida.backward()                                  # 4. backward
            opt.step()                                          # 5. paso del optimizador
            total += perdida.item() * len(x); n += len(x)
        fila = {"época": ep + 1, "pérdida train": total / n}
        if dev_loader is not None:
            P = predecir_proba(modelo, dev_loader, disp)
            fila["F1 macro dev"] = float(evaluar(P))
            if fila["F1 macro dev"] > mejor:
                mejor = fila["F1 macro dev"]
                mejor_estado = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
        historial.append(fila)
        if verbose:
            print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in fila.items()))
    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return pd.DataFrame(historial)


# ---------------------------------------------------------------- Escalera: baselines lineales y evaluación
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support


class TokenizadorTfidf:
    """Mismos tokens que las redes (conserva <mask> y números), sin stopwords salvo negadores.
    Es una clase y no una lambda para que joblib pueda guardar el vectorizador."""
    def __init__(self):
        self.stop = lexico._stopwords("english") - lexico.NEGADORES_EN

    def __call__(self, t):
        return [w for w in tokens(t) if w not in self.stop]


def tokenizador_tfidf():
    return TokenizadorTfidf()


def tfidf(min_df=3, max_features=100_000):
    return TfidfVectorizer(tokenizer=tokenizador_tfidf(), lowercase=False, token_pattern=None,
                           ngram_range=(1, 2), min_df=min_df, max_features=max_features, sublinear_tf=True)


def nb_multietiqueta(alpha=0.5):
    return OneVsRestClassifier(MultinomialNB(alpha=alpha))


def logreg_multietiqueta(C=1.0):
    # Una regresión logística por etapa; class_weight compensa que los positivos son pocos
    return OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


def pos_weight_de(Y):
    """n_neg / n_pos por etiqueta (como en BCEWithLogitsLoss de PyTorch)."""
    Y = np.asarray(Y)
    pos = Y.sum(axis=0).clip(min=1)
    return (len(Y) - pos) / pos


def metricas(Y, P, nombres, umbral=0.5, pesos=None):
    """P/R/F1 por etiqueta (pesos de muestreo opcionales) y F1 macro del ciclo y de todas."""
    Yh = (np.asarray(P) >= umbral).astype(int)
    p, r, f, s = precision_recall_fscore_support(np.asarray(Y), Yh, average=None, zero_division=0, sample_weight=pesos)
    tabla = pd.DataFrame({"precisión": p, "recall": r, "F1": f, "positivos": np.asarray(Y).sum(axis=0)}, index=nombres)
    ciclo = [n for n in nombres if n in ("llegada", "atencion", "cocina_mesa", "cobro")]
    return tabla, float(tabla.loc[ciclo, "F1"].mean()), float(tabla.F1.mean())


def guardar_red(modelo, vocab, nombre, **meta):
    MODELOS.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": modelo.state_dict(), "vocab": vocab, "meta": meta}, MODELOS / f"{nombre}.pt")


# ---------------------------------------------------------------- ¿El texto predice las estrellas? (avance 5.2)
def r2_estrellas(resenas, test=0.25, semilla=0, excluir_autores=False):
    """Ridge sobre TF-IDF, restaurantes de test no vistos. excluir_autores=True además saca del test
    las reseñas de autores que también escribieron en train (posible fuga por estilo personal)."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    i_tr, i_te = next(GroupShuffleSplit(n_splits=1, test_size=test, random_state=semilla)
                      .split(resenas, groups=resenas.restaurante))
    tr, te = resenas.iloc[i_tr], resenas.iloc[i_te]
    if excluir_autores:
        te = te[~te.autor.isin(set(tr.autor))]
    vec = TfidfVectorizer(tokenizer=lexico.tokenizador_en(), lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=20, max_features=30_000)
    A, B = vec.fit_transform(tr.texto), vec.transform(te.texto)
    pred = np.clip(Ridge(alpha=3.0).fit(A, tr.estrellas).predict(B), 1, 5)
    return {"reseñas test": len(te), "R²": r2_score(te.estrellas, pred), "MAE": mean_absolute_error(te.estrellas, pred),
            "MAE media": mean_absolute_error(te.estrellas, np.full(len(te), tr.estrellas.mean()))}


# ---------------------------------------------------------------- Cargar modelos guardados y predecir (06-08)
def cargar_red(clave):
    """Reconstruye una red guardada por guardar_red (MLP, RNN o LSTM). Devuelve (modelo, vocab)."""
    d = torch.load(MODELOS / f"{clave}.pt", map_location="cpu", weights_only=False)
    W = torch.zeros_like(d["estado"]["embedding.weight"])
    n = d["estado"]["fc.weight"].shape[0] if "fc.weight" in d["estado"] else d["estado"]["fc2.weight"].shape[0]
    if clave.startswith("MLP"):
        modelo = MLPPromedio(W, n)
    else:
        modelo = Recurrente(W, n, celda="lstm" if clave.startswith("LSTM") else "rnn")
    modelo.load_state_dict(d["estado"])
    return modelo.eval(), d["vocab"]


def proba_red(modelo, vocab, textos, batch_size=512):
    ds = OracionesDataset(a_indices(list(textos), vocab))
    return predecir_proba(modelo.to(dispositivo()), DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate))


def cargar_lineales(variante):
    import joblib
    return joblib.load(MODELOS / f"lineales_{'mask' if variante == '<mask>' else variante}.joblib")


def proba_red_ordenada(modelo, vocab, textos, batch_size=1024):
    """Como proba_red, pero agrupa oraciones de largo parecido en cada lote (mucho menos relleno)
    y devuelve las probabilidades en el orden original."""
    idx = a_indices(list(textos), vocab)
    orden = np.argsort([len(s) for s in idx], kind="stable")
    ds = OracionesDataset([idx[i] for i in orden])
    P = predecir_proba(modelo.to(dispositivo()), DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate))
    salida = np.empty_like(P)
    salida[orden] = P
    return salida


# ---------------------------------------------------------------- Control de circularidad (07, P2 y P3)
def clasificador_negatividad(o_train, n=60_000, semilla=0):
    """"¿Esta oración suena negativa?": TF-IDF + LogReg entrenada con oraciones de reseñas 1-2★ (1)
    contra 5★ (0). Es circular a propósito: mide cuánto separa CUALQUIER tono negativo, para comparar
    contra eso la razón de cada etapa del ciclo."""
    neg = o_train[o_train.estrellas <= 2].sample(min(n, (o_train.estrellas <= 2).sum()), random_state=semilla)
    pos = o_train[o_train.estrellas == 5].sample(len(neg), random_state=semilla)
    d = pd.concat([neg.assign(y=1), pos.assign(y=0)])
    vec = tfidf(min_df=5, max_features=50_000).fit(d.oracion)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(vec.transform(d.oracion), d.y)
    return lambda textos: lr.predict_proba(vec.transform(textos))[:, 1] >= 0.5
