"""
app.py — Indicadores de concentración de mercado con Streamlit.

Ejecutar con:
    pip install streamlit numpy pandas
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------
# Funciones de cálculo
# ----------------------------------------------------------------------
def validar_cuotas(cuotas, tol=1e-6):
    """
    Valida y devuelve el vector de cuotas como arreglo de NumPy.

    Parámetros
    ----------
    cuotas : array-like
        Cuotas de mercado s_i, con 0 <= s_i <= 1 y suma igual a 1.
    tol : float
        Tolerancia numérica para la condición de suma = 1.

    Retorna
    -------
    np.ndarray
        Vector de cuotas como float.
    """
    s = np.asarray(cuotas, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("El vector de cuotas está vacío.")
    if np.any(np.isnan(s)):
        raise ValueError("El vector de cuotas contiene valores no numéricos.")
    if np.any(s < 0) or np.any(s > 1):
        raise ValueError("Cada cuota debe estar entre 0 y 1.")
    if not np.isclose(s.sum(), 1.0, atol=tol):
        raise ValueError(f"Las cuotas deben sumar 1 (suma actual: {s.sum():.6f}).")
    return s


def ratio_concentracion(cuotas, k):
    """
    Ratio de Concentración CR_k: suma de las k mayores cuotas.

        CR_k = sum_{i=1}^{k} s_(i),   con s_(1) >= s_(2) >= ... >= s_(n)

    Rango: (0, 1]. Valores cercanos a 1 indican alta concentración.
    """
    s = validar_cuotas(cuotas)
    if not (1 <= k <= s.size):
        raise ValueError(f"k debe estar entre 1 y {s.size}.")
    return float(np.sort(s)[::-1][:k].sum())


def indice_hhi(cuotas, escala_10000=False):
    """
    Índice Herfindahl-Hirschman (IHH):

        IHH = sum_i s_i^2

    Rango: [1/n, 1]. Si escala_10000=True, se expresa en la escala
    habitual de las autoridades de competencia (0 - 10.000).
    """
    s = validar_cuotas(cuotas)
    ihh = float(np.sum(s ** 2))
    return ihh * 10000 if escala_10000 else ihh


def indice_dominancia(cuotas):
    """
    Índice de Dominancia (García Alba, 1994):

        h_i = s_i^2 / IHH            (participación de la empresa i en el IHH)
        ID  = sum_i h_i^2

    Rango: [1/n, 1]. Mide cuánto del IHH está explicado por las empresas
    más grandes: crece cuando una o pocas empresas dominan el mercado.
    """
    s = validar_cuotas(cuotas)
    ihh = np.sum(s ** 2)
    h = (s ** 2) / ihh
    return float(np.sum(h ** 2))


def indice_entropia(cuotas, normalizado=False):
    """
    Índice de Entropía (Shannon):

        IE = sum_i s_i * ln(1 / s_i) = -sum_i s_i * ln(s_i)

    Convención: 0 * ln(0) = 0 (las cuotas nulas no aportan).
    Rango: [0, ln(n)]. A diferencia de los demás, MENOR entropía implica
    MAYOR concentración (IE = 0 es monopolio; IE = ln(n) es competencia
    simétrica). Con normalizado=True se divide por ln(n) y queda en [0, 1].
    """
    s = validar_cuotas(cuotas)
    positivas = s[s > 0]
    ie = float(-np.sum(positivas * np.log(positivas)))
    if normalizado:
        n = s.size
        return ie / np.log(n) if n > 1 else 0.0
    return ie


# ----------------------------------------------------------------------
# Interfaz Streamlit
# ----------------------------------------------------------------------
def parsear_texto(texto):
    """Convierte texto separado por comas, espacios o saltos de línea en lista de floats."""
    limpio = texto.replace(";", " ").replace(",", " ").split()
    return [float(x.replace("%", "")) for x in limpio]


def main():
    st.set_page_config(page_title="Concentración de mercado", page_icon="📊")
    st.title("📊 Indicadores de concentración de mercado")
    st.write("Calcula CR_k, IHH, Índice de Dominancia e Índice de Entropía "
             "a partir de un vector de cuotas de mercado.")

    modo = st.radio("Forma de ingreso de los datos",
                    ["Escribir cuotas", "Tabla editable", "Cargar CSV"],
                    horizontal=True)

    cuotas = None
    try:
        if modo == "Escribir cuotas":
            texto = st.text_area(
                "Cuotas (separadas por coma, espacio o salto de línea)",
                value="0.40, 0.25, 0.15, 0.10, 0.06, 0.04")
            cuotas = parsear_texto(texto)

        elif modo == "Tabla editable":
            base = pd.DataFrame({"Empresa": ["A", "B", "C", "D"],
                                 "Cuota": [0.40, 0.30, 0.20, 0.10]})
            editada = st.data_editor(base, num_rows="dynamic",
                                     use_container_width=True)
            cuotas = editada["Cuota"].dropna().tolist()

        else:
            archivo = st.file_uploader("CSV con una columna de cuotas", type="csv")
            if archivo is not None:
                df = pd.read_csv(archivo)
                columnas_num = df.select_dtypes(include="number").columns.tolist()
                if not columnas_num:
                    st.error("El CSV no tiene columnas numéricas.")
                else:
                    col = st.selectbox("Columna de cuotas", columnas_num)
                    cuotas = df[col].dropna().tolist()

        if cuotas is None:
            st.stop()

        # Opción para ingresar porcentajes o normalizar
        c1, c2 = st.columns(2)
        en_porcentaje = c1.checkbox("Los datos están en porcentaje (0-100)")
        normalizar = c2.checkbox("Normalizar para que sumen 1")

        s = np.asarray(cuotas, dtype=float)
        if en_porcentaje:
            s = s / 100.0
        if normalizar and s.sum() > 0:
            s = s / s.sum()

        validar_cuotas(s)  # lanza ValueError si hay problemas

        st.subheader("Parámetros")
        k = st.slider("k para CR_k", min_value=1, max_value=len(s),
                      value=min(4, len(s)))
        p1, p2 = st.columns(2)
        ihh_10000 = p1.checkbox("Mostrar IHH en escala 0-10.000",
