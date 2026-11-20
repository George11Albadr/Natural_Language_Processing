"""Foodie Advisor · tablero del gerente.

    .venv/bin/streamlit run producto/app.py

Necesita los parquets de data/procesado/ (notebooks 00 y 07) y los modelos de modelos/ (05).
"""
import sys
from pathlib import Path

import html
import re

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodie_advisor import CIUDADES, etiquetas as et, producto as p  # noqa: E402

st.set_page_config(page_title="Foodie Advisor", page_icon="🍽️", layout="wide")

ROJO, GRIS, TINTA = "#b5483a", "#9a9a9a", "#2b2b2b"
COLOR_ETAPA = {"Llegada y mesa": "#3b6ea5", "Atención en sala": ROJO, "Cocina → mesa": "#d18f2b", "Cobro": "#5b8c5a"}
NOMBRE_CIUDAD = {"Madrid": "Madrid", "Barcelona": "Barcelona", "London": "Londres", "Paris": "París",
                 "New_York": "Nueva York", "New_Delhi": "Nueva Delhi"}

st.markdown("""
<style>
.bloque {border-left: 6px solid %s; padding: .8rem 1.1rem; background: rgba(181,72,58,.07); border-radius: 4px; margin: .4rem 0 1rem 0;}
.bloque h2 {margin: 0; font-size: 1.55rem;}
.bloque p {margin: .25rem 0 0 0; color: #555;}
.neutro {border-left-color: %s; background: rgba(150,150,150,.08);}
.cita {border-left: 3px solid #ddd; padding: .2rem .8rem; margin: .35rem 0; font-size: .95rem;}
.chip {display:inline-block; color:#2b2b2b; vertical-align:middle; font-size:.75rem; padding: 0 .45rem; border-radius: 9px; background:#eee; margin-left:.3rem;}
.chip.ok {background:#e3efe2;} .chip.med {background:#f6ead3;} .chip.no {background:#eee;}
</style>""" % (ROJO, GRIS), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando la ciudad…")
def ciudad(c):
    return p.cargar_ciudad(c)


def negritas(t):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def chip_sello(sello):
    clase = {"validada": "ok", "señal débil": "med"}.get(sello, "no")
    return f'<span class="chip {clase}">{sello}</span>' if sello else ""


def bloque_prioridad(prio, detalle=""):
    texto, sello = p.titular(prio) if prio != "muestra insuficiente" else (
        f"Muestra insuficiente: hacen falta al menos {p.MIN_NUEVAS} reseñas para recomendar", "")
    neutro = "" if prio in et.CICLO else " neutro"
    st.markdown(f'<div class="bloque{neutro}"><h2>{texto} {chip_sello(sello)}</h2><p>{detalle}</p></div>',
                unsafe_allow_html=True)


def grafica_ciclo(tabla):
    t = tabla[tabla.ciclo].reset_index()
    t["Etapa"] = pd.Categorical(t.Etapa, list(COLOR_ETAPA))
    base = alt.Chart(t).encode(y=alt.Y("Etapa:N", sort=list(COLOR_ETAPA), title=None))
    barras = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("% quejas:Q", title="% de reseñas con queja explícita",
                scale=alt.Scale(domain=[0, max(t["% quejas"].max(), t["mediana pares %"].max()) * 1.18]), axis=alt.Axis(tickCount=6)),
        color=alt.condition(alt.datum.candidata, alt.value(ROJO), alt.value("#c9b6b2")),
        tooltip=[alt.Tooltip("Etapa:N"), alt.Tooltip("% quejas:Q", format=".1f"),
                 alt.Tooltip("mediana pares %:Q", format=".1f", title="mediana de pares %"),
                 alt.Tooltip("percentil:Q", format=".0f"), alt.Tooltip("reseñas con queja:Q"),
                 alt.Tooltip("posibles (solo LogReg):Q"), alt.Tooltip("sello:N")])
    mediana = base.mark_tick(color=TINTA, thickness=2, size=30).encode(
        x="mediana pares %:Q", tooltip=[alt.Tooltip("mediana pares %:Q", format=".1f", title="mediana de pares %")])
    texto = base.mark_text(align="left", dx=6, color=TINTA, fontSize=12).encode(
        x="% quejas:Q", text=alt.Text("etiqueta:N"))
    t["etiqueta"] = t.apply(lambda f: f"p{f.percentil:.0f}", axis=1)
    texto = texto.properties(data=t)
    return (barras + mediana + texto).properties(height=190)


def tabla_mostrar(tabla, ciclo=True):
    t = tabla[tabla.ciclo == ciclo]
    cols = ["Etapa", "% quejas", "mediana pares %", "percentil", "reseñas con queja", "posibles (solo LogReg)", "exceso ★", "sello"]
    return t[cols].style.format({"% quejas": "{:.1f}", "mediana pares %": "{:.1f}", "percentil": "{:.0f}",
                                 "exceso ★": "{:+.3f}"})


def leyenda():
    with st.expander("Cómo leer esto"):
        st.markdown(f"""
- **% de quejas** = reseñas con una queja explícita de esa etapa (léxico del proyecto), contraída hacia la media de la ciudad para no exagerar con pocas reseñas. Es un **piso**: las quejas parafraseadas no se cuentan.
- **Percentil** = posición frente a los pares de la misma ciudad (restaurantes con ≥ 100 reseñas). La raya negra es la mediana de los pares.
- **Prioridad** = la etapa del ciclo con más exceso sobre la mediana × lo que cuesta en estrellas, si está en p ≥ {p.sc.MIN_PERCENTIL}, tiene ≥ {p.sc.MIN_MARCADAS} reseñas y su intervalo creíble al 80% queda por encima de la mediana.
- **Quejas** = explícitas (léxico). **Posibles** = las que solo detecta LogReg, que lee paráfrasis: se muestran como pista, no se cuentan porque su precisión no es suficiente.
- **Sello**: atención en sala está validada en la prueba de Barcelona; cocina → mesa es señal débil; llegada y cobro no están validadas, por eso dicen *revisar* y no *atacar*.
- Las estrellas en juego son asociación dentro del mismo restaurante, no una ganancia garantizada al arreglarlo. Entre 2017 y 2018-19 la etapa prioritaria se repitió en el 30.7% de los restaurantes de Barcelona, contra 11.3% por azar.
""")


# ======================================================================= Navegación
st.sidebar.title("🍽️ Foodie Advisor")
st.sidebar.caption("Qué etapa del servicio en sala atacar primero, según lo que dicen las reseñas.")
VISTAS = ["1 · El problema", "2 · Cómo funciona", "3 · Mi restaurante", "4 · Reseñas nuevas", "5 · Mi ciudad", "6 · Resultados"]
vista = st.sidebar.radio("Recorrido", VISTAS)[4:]
c_sel = st.sidebar.selectbox("Ciudad", CIUDADES, format_func=NOMBRE_CIUDAD.get)
ctx = ciudad(c_sel)

# ======================================================================= Mi restaurante
if vista == "El problema":
    import json
    from foodie_advisor import RAIZ
    h0 = json.loads((RAIZ / "resultados" / "hallazgos_00.json").read_text())
    cs = p.costos().reset_index()
    at = cs[cs.etapa == "atencion"]["β"]
    st.title("¿Qué le dicen las reseñas al gerente?")
    st.markdown("El gerente de un restaurante conoce su promedio de estrellas, pero no sabe **qué parte del servicio "
                "falla, cuánto le cuesta ni cómo está frente a restaurantes parecidos**. El punto de venta no registra "
                "eso; solo queda escrito en las reseñas.")
    st.subheader("La actividad operativa: el ciclo de servicio en sala")
    etapas = [("1. Llegada y mesa", "Espera sin reserva, reserva perdida", "Anfitrión"),
              ("2. Atención en sala", "Trato grosero, clientes ignorados, olvidos", "Meseros"),
              ("3. Cocina → mesa", "Lentitud, comida fría, pedido equivocado", "El pase"),
              ("4. Cobro", "Cargos no pedidos, errores en la cuenta", "Caja")]
    for col, (t_, falla, quien) in zip(st.columns(4), etapas):
        col.markdown(f'<div class="bloque neutro"><h2 style="font-size:1.15rem">{t_}</h2><p>{falla}</p>'
                     f'<p><b>Responsable:</b> {quien}</p></div>', unsafe_allow_html=True)
    st.caption("La comida, el precio y el local quedan fuera del ciclo y sirven como punto de comparación.")
    st.subheader("Los datos")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Reseñas de TripAdvisor", f"{h0['resenas_total']:,}")
    d2.metric("Restaurantes", f"{h0['restaurantes_total']:,}")
    d3.metric("Ciudades", "6", help="Madrid, Barcelona, Londres, París, Nueva York y Nueva Delhi")
    d4.metric("Costo de una queja de atención", f"{at.max():.1f} a {at.min():.1f}★",
              help="Estrellas menos por reseña con queja explícita de atención, dentro del mismo restaurante, en las 6 ciudades.")
    st.markdown("Además, SemEval 2016 en español (reseñas anotadas por los organizadores) para comparar Naive Bayes y "
                "regresión logística en polaridad.")
    st.markdown('<div class="bloque"><h2>La meta</h2><p>Que cada gerente sepa qué etapa del servicio atacar primero, '
                'frente a restaurantes parecidos y con evidencia de sus propias reseñas.</p></div>', unsafe_allow_html=True)

elif vista == "Cómo funciona":
    st.title("Cómo funciona el detector")
    st.write("Cada oración de cada reseña se clasifica en las 4 etapas del servicio en sala: llegada y mesa, "
             "atención, cocina → mesa y cobro. Con eso se calcula qué tanto se queja la gente de cada etapa, "
             "cuánto cuesta en estrellas y qué etapa conviene atacar primero.")

    st.subheader("Los retos")
    r1, r2, r3 = st.columns(3)
    r1.markdown("**1. La gente no usa las palabras de la lista.** Un léxico encuentra *rude* o *slow*, pero no "
                "*nobody came to take our order*. Hacía falta un modelo que lea el contexto.")
    r2.markdown("**2. No había datos etiquetados en inglés.** Entrené con el léxico como profesor imperfecto "
                "y borré la palabra clave en la mitad de los ejemplos, para que el modelo aprenda el contexto.")
    r3.markdown("**3. No podía evaluar contra el mismo léxico.** Sería circular. Por eso hice un conjunto gold "
                "aparte, etiquetado con una guía, para medir de verdad.")

    tabla_m, h6 = p.escalera()
    st.subheader("El gold: la prueba final")
    g1, g2, g3 = st.columns(3)
    g1.metric("Oraciones de prueba", h6["n_gold"], help="De Barcelona, una ciudad que los modelos nunca vieron en entrenamiento.")
    g2.metric("Sin palabra del léxico", h6["n_parafrasis"], help="Muestreadas al azar, no por palabras clave: así entran las paráfrasis.")
    g3.metric("Acuerdo entre anotadores (κ)", f"{h6['kappa_ciclo']:.2f}", help="Doble anotación en 100 oraciones; la meta era 0.65.")

    st.subheader("Qué modelo elegí")
    izq, der = st.columns([1.2, 1])
    with izq:
        base = alt.Chart(tabla_m).encode(y=alt.Y("Modelo:N", sort=list(tabla_m.Modelo), title=None))
        barras = base.mark_bar(cornerRadiusEnd=4, height=20).encode(
            x=alt.X("F1 ciclo:Q", title="F1 en las 4 etapas (gold de Barcelona)", scale=alt.Scale(domain=[0, 0.75])),
            color=alt.condition(alt.datum.Modelo == "Regresión logística", alt.value(ROJO), alt.value("#c9b6b2")),
            tooltip=["Modelo:N", alt.Tooltip("F1 ciclo:Q", format=".3f"), alt.Tooltip("Precisión atención:Q", format=".2f"),
                     alt.Tooltip("Recall paráfrasis:Q", format=".2f")])
        meta = alt.Chart(pd.DataFrame({"x": [0.70]})).mark_rule(strokeDash=[4, 3], color=TINTA).encode(x="x:Q")
        st.altair_chart((barras + meta).properties(height=230), use_container_width=True)
        st.caption("Línea punteada = meta fijada antes de evaluar (F1 0.70).")
    with der:
        d = h6["diferencia_logreg_lstm"]["F1 ciclo"]
        st.markdown(f"""
- **Ganó la regresión logística** (F1 {tabla_m.set_index('Modelo').loc['Regresión logística', 'F1 ciclo']:.3f}).
- **El LSTM** sacó {tabla_m.set_index('Modelo').loc['LSTM', 'F1 ciclo']:.3f}. La diferencia, {d['diferencia (LogReg − LSTM)']:+.3f}, tiene un intervalo de {d['IC inf']:+.3f} a {d['IC sup']:+.3f}: **empatan**. Con el mismo rendimiento, me quedo con el más simple.
- **Lo bueno:** encuentra el {tabla_m.set_index('Modelo').loc['Regresión logística', 'Recall paráfrasis']:.0%} de las quejas que el léxico no ve.
- **Lo que falta:** marca de más (precisión de {tabla_m.set_index('Modelo').loc['Regresión logística', 'Precisión atención']:.2f} en atención, contra 0.80 pedido).
- **Decisión:** la app cuenta las quejas explícitas (precisión 1.0 en atención) y muestra las del modelo como *posibles*.
""")
    st.dataframe(tabla_m.style.format({"F1 ciclo": "{:.3f}", "Precisión atención": "{:.2f}", "Recall paráfrasis": "{:.2f}"}),
                 width="stretch", hide_index=True)

    st.subheader("Pruébalo")
    ejemplo_txt = "\n".join([
        "The waiter was rude.",                                              # explícita: aciertan los tres
        "We waited 40 minutes and nobody came to take our order.",           # paráfrasis: el léxico no la ve
        "The waitress rolled her eyes when we asked for the menu.",          # paráfrasis: el léxico no la ve
        "They added a 10% service fee without telling us.",                  # LogReg ve cobro, el LSTM no
        "Our mains came out after an hour.",                                 # los dos modelos se equivocan de etapa
        "Terrible night, I will never come back.",                           # falso positivo: tono negativo
        "They had no lemon sorbet left so I asked for the bill and left.",   # falso positivo: menciona la cuenta
        "The waiter was not rude at all, very kind.",                        # negación: fallan los tres
    ])
    entrada = st.text_area("Oraciones", ejemplo_txt, height=230, label_visibility="collapsed")
    oraciones = [l.strip() for l in entrada.splitlines() if l.strip()]
    if oraciones:
        with st.spinner("Clasificando con los tres modelos…"):
            res = p.probar_oraciones(oraciones)
        piv = res.pivot(index="oración", columns="modelo", values="detecta").reindex(oraciones)
        st.dataframe(piv[["Regex (léxico)", "Regresión logística", "LSTM"]], width="stretch")

elif vista == "Mi restaurante":
    nombres = sorted(ctx["pares"])
    defecto = {"Madrid": "C_Tapas", "Barcelona": "El_Cafe_D_en_Victor"}.get(c_sel, nombres[0])
    rest = st.selectbox("Restaurante", nombres, index=nombres.index(defecto), format_func=lambda x: x.replace("_", " "))
    f = p.ficha(ctx, rest)
    t = f["tabla"]
    st.title(f["nombre"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reseñas", f"{f['n']:,}")
    m2.metric("Promedio", f"{f['estrellas']:.2f} ★", f"{f['estrellas'] - f['mediana pares ★']:+.2f} vs pares",
              delta_color="normal")
    m3.metric("Percentil ★", f"p{f['pct estrellas']:.0f}", help=f"Posición en estrellas frente a {f['n pares']:,} pares")
    ciclo = t[t.ciclo]
    m4.metric("Quejas del ciclo", f"{int(ciclo['reseñas con queja'].sum())}",
              help=f"Explícitas (léxico). Otras {int(ciclo['posibles (solo LogReg)'].sum())} son posibles según LogReg y no se cuentan.")

    prio = f["prioridad"]
    if prio in et.CICLO:
        x = t.loc[prio]
        det = (f"{x['% quejas']:.1f}% de sus reseñas se quejan de esto, contra {x['mediana pares %']:.1f}% de la mediana "
               f"de sus pares (percentil {x.percentil:.0f}). Cada queja así se asocia con {abs(x['costo ★']):.2f}★ menos.")
    elif prio.startswith("fuera: "):
        x = t.loc[prio[7:]]
        det = (f"El servicio en sala no destaca frente a sus pares; lo que sí destaca es {p.NOMBRE_CORTO[prio[7:]].lower()} "
               f"(percentil {x.percentil:.0f}).")
    else:
        det = "Ninguna etapa del ciclo ni fuera de él sobresale frente a la mediana de la ciudad."
    bloque_prioridad(prio, det)

    izq, der = st.columns([1.15, 1])
    with izq:
        st.subheader("El ciclo de servicio frente a sus pares")
        st.altair_chart(grafica_ciclo(t), use_container_width=True)
        st.caption("Barra roja = etapa candidata a prioridad. Raya negra = mediana de los pares. Etiqueta = percentil.")
    with der:
        st.subheader("Evolución por año")
        tend = f["tendencia"]
        if len(tend) >= 2:
            largo = tend[et.CICLO].rename(columns=p.NOMBRE_CORTO).reset_index().melt("año", var_name="Etapa", value_name="%")
            graf = alt.Chart(largo).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
                x=alt.X("año:O", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("%:Q", title="% de reseñas con queja"),
                color=alt.Color("Etapa:N", scale=alt.Scale(domain=list(COLOR_ETAPA), range=list(COLOR_ETAPA.values())),
                                legend=alt.Legend(orient="bottom", title=None, columns=2)),
                tooltip=["año:O", "Etapa:N", alt.Tooltip("%:Q", format=".1f")]).properties(height=230)
            st.altair_chart(graf, use_container_width=True)
        else:
            st.info("Pocos años con 20 reseñas o más para mostrar una tendencia.")

    foco = prio if prio in et.CICLO else (prio[7:] if prio.startswith("fuera: ") else t["exceso ★"].idxmax())
    st.subheader(f"Lo que dicen sus clientes · {p.NOMBRE_CORTO[foco]}")
    cs = p.citas(ctx, f, foco)
    if cs.empty:
        st.write("Sin quejas explícitas de esta etapa.")
    for _, c in cs.iterrows():
        st.markdown(f'<div class="cita">{"★" * int(c.estrellas)}<br>“{negritas(html.escape(p.resaltar(c.oracion, foco)))}”</div>'
                    , unsafe_allow_html=True)
    otras = [e for e in et.CICLO if e != foco and t.loc[e, "reseñas con queja"] > 0]
    if otras:
        with st.expander("Citas de las otras etapas del ciclo"):
            for e in otras:
                st.markdown(f"**{p.NOMBRE_CORTO[e]}**")
                for _, c in p.citas(ctx, f, e, n=2).iterrows():
                    st.markdown(f"- {'★' * int(c.estrellas)} “{p.resaltar(c.oracion, e)}”")

    st.subheader("Detalle por etapa")
    st.dataframe(tabla_mostrar(t), width="stretch", hide_index=True)
    with st.expander("Fuera del ciclo: sabor, precio y local"):
        st.dataframe(tabla_mostrar(t, ciclo=False), width="stretch", hide_index=True)
    leyenda()

# ======================================================================= Reseñas nuevas
elif vista == "Reseñas nuevas":
    st.title("Analizar las reseñas de mi local")
    st.write(f"Pega las reseñas de tu restaurante (en inglés), separadas por una línea en blanco, o sube un CSV con una "
             f"columna de texto. Se comparan contra los {len(ctx['pares']):,} restaurantes de "
             f"{NOMBRE_CIUDAD[c_sel]} con el mismo léxico y la misma regla de prioridad.")
    if "texto_nuevas" not in st.session_state:
        st.session_state.texto_nuevas = ""
    a, b = st.columns([1, 3])
    if a.button("Cargar un ejemplo real"):
        st.session_state.texto_nuevas = "\n\n".join(p.ejemplo())
        st.session_state.origen = "97 reseñas de El Viajero (Madrid). Tiene menos de 100, así que no forma parte de los pares."
    if st.session_state.get("origen"):
        b.caption(st.session_state.origen)
    texto = st.text_area("Reseñas", key="texto_nuevas", height=220)
    archivo = st.file_uploader("…o un CSV", type="csv")
    resenas = p.leer_resenas(texto) if texto.strip() else []
    if archivo is not None:
        df = pd.read_csv(archivo)
        col = st.selectbox("Columna con el texto", df.columns, index=int(df.dtypes.eq(object).values.argmax()))
        resenas = df[col].dropna().astype(str).tolist()
    if resenas:
        with st.spinner("Clasificando…"):
            out = p.evaluar_nuevas(resenas, ctx)
        t = out["tabla"]
        st.metric("Reseñas analizadas", out["n"])
        prio = out["prioridad"]
        det = ""
        if prio in et.CICLO:
            x = t.loc[prio]
            det = (f"{x['% quejas']:.1f}% de tus reseñas, contra {x['mediana pares %']:.1f}% de la mediana de "
                   f"{NOMBRE_CIUDAD[c_sel]} (percentil {x.percentil:.0f}).")
        bloque_prioridad(prio, det)
        st.altair_chart(grafica_ciclo(t), use_container_width=True)
        st.dataframe(tabla_mostrar(t), width="stretch", hide_index=True)
        o = out["oraciones"]
        st.subheader("Oraciones con queja")
        filas = []
        for e in et.ETIQUETAS:
            for _, r in o[(o[e] == 1) | (o[e + "_lr"] == 1)].iterrows():
                conf = "explícita" if r[e] else "posible (solo LogReg)"
                filas.append({"Etapa": p.NOMBRE_CORTO[e], "confianza": conf, "reseña": r.resena + 1, "oración": r.oracion})
        vistas = pd.DataFrame(filas)
        if len(vistas):
            ver = st.multiselect("Tipo", ["explícita", "posible (solo LogReg)"], default=["explícita"])
            st.dataframe(vistas[vistas.confianza.isin(ver)], width="stretch", hide_index=True)
        leyenda()
    else:
        st.info("Esperando reseñas.")

# ======================================================================= Mi ciudad
elif vista == "Mi ciudad":
    st.title(f"{NOMBRE_CIUDAD[c_sel]}: dónde falla el servicio")
    st.write(f"{len(ctx['pares']):,} restaurantes con 100 reseñas o más · mediana {ctx['prom_pares'].median():.2f} ★")
    izq, der = st.columns(2)
    with izq:
        st.subheader("Cuánto cuesta cada queja")
        ci = ctx["costos_ic"].reset_index()
        ci["Etapa"] = ci.etapa.map(p.NOMBRE_CORTO)
        ci["grupo"] = ci.etapa.isin(et.CICLO).map({True: "ciclo de servicio", False: "fuera del ciclo"})
        orden = ci.sort_values("β").Etapa.tolist()
        base = alt.Chart(ci).encode(y=alt.Y("Etapa:N", sort=orden, title=None))
        graf = (base.mark_rule(strokeWidth=2, color=GRIS).encode(x=alt.X("IC inf:Q", title="estrellas por reseña con queja"), x2="IC sup:Q")
                + base.mark_point(filled=True, size=90).encode(
                    x="β:Q", color=alt.Color("grupo:N", scale=alt.Scale(range=[ROJO, GRIS]), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Etapa:N", alt.Tooltip("β:Q", format=".2f"), alt.Tooltip("IC inf:Q", format=".2f"),
                             alt.Tooltip("IC sup:Q", format=".2f")])).properties(height=250)
        st.altair_chart(graf, use_container_width=True)
        st.caption("Dentro del mismo restaurante, con intervalo bootstrap al 95%. Asociación, no causa.")
    with der:
        st.subheader("Qué le toca atacar a cada restaurante")
        d = ctx["prioridad"].value_counts(normalize=True).mul(100).rename("%").reset_index()
        d.columns = ["prioridad", "%"]
        d["Prioridad"] = d.prioridad.map(lambda x: p.NOMBRE_CORTO.get(x) or (
            "Fuera: " + p.NOMBRE_CORTO[x[7:]] if x.startswith("fuera: ") else "Sin punto débil"))
        d["ciclo"] = d.prioridad.isin(et.CICLO)
        graf = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=20).encode(
            y=alt.Y("Prioridad:N", sort="-x", title=None), x=alt.X("%:Q", title="% de restaurantes"),
            color=alt.condition(alt.datum.ciclo, alt.value(ROJO), alt.value(GRIS)),
            tooltip=["Prioridad:N", alt.Tooltip("%:Q", format=".1f")]).properties(height=250)
        st.altair_chart(graf, use_container_width=True)
        st.caption("Ninguna etapa se lleva a todos: la regla distingue restaurantes.")
    st.subheader("Restaurantes con esa etapa como punto débil")
    e = st.selectbox("Etapa", et.CICLO, format_func=p.NOMBRE_CORTO.get, index=1)
    rk = p.ranking(ctx, e)
    st.caption(f"{len(rk)} restaurantes son candidatos en {p.NOMBRE_CORTO[e].lower()} · sello: {p.SELLO_CORTO[e]}")
    rk["exceso"] = rk.exceso.abs()
    st.dataframe(rk.sort_values("exceso", ascending=False).style.format(
        {"estrellas": "{:.2f}", "% quejas": "{:.1f}", "percentil": "{:.0f}", "exceso": "{:.3f}"}),
        width="stretch", hide_index=True)
    leyenda()

# ======================================================================= Resultados
if vista == "Resultados":
    import json
    from foodie_advisor import RAIZ
    h9 = json.loads((RAIZ / "resultados" / "hallazgos_09.json").read_text())
    st.title("Resultados")
    st.markdown("""
1. **El costo de cada falla:** una queja explícita de atención se asocia con 1.3 a 1.6 estrellas menos, y es la más cara del ciclo en las 6 ciudades.
2. **Una app** que le dice a cada gerente qué etapa atacar primero, frente a sus pares y con citas de sus clientes, también para un local que no está en los datos.
""")
    st.subheader("¿La recomendación se sostiene en el tiempo?")
    st.write("Calculé la prioridad con las reseñas hasta 2017 y revisé si se repetía en 2018-2019.")
    rx = h9["apuntar"]["regex v1"]
    d = pd.DataFrame([{"Ciudad": NOMBRE_CIUDAD[c], "Serie": serie, "%": v[clave] * 100}
                      for c, v in rx.items() for serie, clave in [("Se repite la prioridad", "coincidencia"),
                                                                  ("Por azar (p95)", "permutación p95")]])
    graf = alt.Chart(d).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Ciudad:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="Serie:N",
        y=alt.Y("%:Q", title="% de restaurantes"),
        color=alt.Color("Serie:N", scale=alt.Scale(domain=["Se repite la prioridad", "Por azar (p95)"], range=[ROJO, GRIS]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Ciudad:N", "Serie:N", alt.Tooltip("%:Q", format=".1f")]).properties(height=260)
    st.altair_chart(graf, use_container_width=True)
    st.write(f"Sí apunta: la etapa prioritaria se repite muy por encima del azar en todas las ciudades. "
             f"No predice: no mejora el pronóstico de las estrellas futuras sobre el promedio pasado.")
    st.subheader("Límites")
    st.markdown("""
- Son **asociaciones** dentro del mismo restaurante, no causas.
- El léxico mide un **piso**: las quejas indirectas o con sarcasmo se le escapan.
- La prueba en el tiempo solo ve restaurantes que **siguieron abiertos**.
- **Siguiente paso:** modelos con contexto preentrenado para las paráfrasis, y validar la recomendación con gerentes reales.
""")
