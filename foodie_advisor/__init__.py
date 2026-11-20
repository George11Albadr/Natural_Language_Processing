"""Foodie Advisor: diagnóstico del ciclo de servicio en sala a partir de reseñas.

Módulos
-------
datos      carga de SemEval-2016 ES y TripAdvisor (6 ciudades), limpieza y división en oraciones
lexico     tokenizadores (con negadores) y léxico regex de etapas (ETAPAS)
etiquetas  etiquetas débiles, enmascarado del término disparador y muestras para anotar
modelos    escalera de modelos: mayoritaria → regex → NB → LogReg → MLP fastText → RNN / LSTM
costos     regresión con efectos fijos por restaurante e IC bootstrap
scorecard  el producto para el gerente
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
PROCESADO = DATA / "procesado"
ANOTACION = DATA / "anotacion"
MODELOS = RAIZ / "modelos"
FIGURAS = RAIZ / "figuras"

CIUDADES = ["Madrid", "Barcelona", "London", "Paris", "New_York", "New_Delhi"]
