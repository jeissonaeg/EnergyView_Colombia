from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DASHBOARD_DIR = BASE_DIR / "outputs" / "dashboard"
MODELO_DIR = BASE_DIR / "outputs" / "modelo"
REPORTES_DIR = BASE_DIR / "outputs" / "reportes"
LOG_DIR = BASE_DIR / "outputs" / "logs"

REPORTES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_GENERACION_MENSUAL = DASHBOARD_DIR / "generacion_mensual.csv"
INPUT_TECNOLOGIA = DASHBOARD_DIR / "generacion_por_tecnologia.csv"
INPUT_RANKING_RECURSOS = DASHBOARD_DIR / "ranking_recursos.csv"

INPUT_COMPARACION_MODELO = MODELO_DIR / "comparacion_real_vs_predicho.csv"
INPUT_PREDICCIONES = MODELO_DIR / "predicciones_generacion.csv"
INPUT_CLASIFICACION = MODELO_DIR / "clasificacion_generacion.csv"

SALIDA_RECOMENDACIONES_CSV = REPORTES_DIR / "recomendaciones_ia.csv"
SALIDA_RECOMENDACIONES_EXCEL = REPORTES_DIR / "recomendaciones_ia.xlsx"
SALIDA_RECOMENDACIONES_DASHBOARD = DASHBOARD_DIR / "recomendaciones_ia_dashboard.csv"
SALIDA_RESUMEN_TXT = REPORTES_DIR / "resumen_agente_ia.txt"
REPORTE_LOG = LOG_DIR / "reporte_agente_ia.txt"


# ============================================================
# CONFIGURACIÓN DE REGLAS
# ============================================================

UMBRAL_ERROR_MODELO_PCT = 5.0
UMBRAL_PARTICIPACION_ALTA = 60.0
UMBRAL_CRECIMIENTO_TECNOLOGIA = 10.0


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def cargar_csv_obligatorio(ruta, nombre):
    """
    Carga un archivo CSV obligatorio.
    Si no existe, detiene el proceso con un error claro.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo requerido para el agente IA:\n{ruta}\n"
            f"Archivo esperado: {nombre}"
        )

    print(f"Cargando {nombre}: {ruta}")

    return pd.read_csv(ruta, encoding="utf-8-sig")


def cargar_datos():
    """
    Carga todas las tablas necesarias para generar recomendaciones.
    """
    generacion_mensual = cargar_csv_obligatorio(
        INPUT_GENERACION_MENSUAL,
        "generacion_mensual.csv"
    )

    tecnologia = cargar_csv_obligatorio(
        INPUT_TECNOLOGIA,
        "generacion_por_tecnologia.csv"
    )

    ranking_recursos = cargar_csv_obligatorio(
        INPUT_RANKING_RECURSOS,
        "ranking_recursos.csv"
    )

    comparacion_modelo = cargar_csv_obligatorio(
        INPUT_COMPARACION_MODELO,
        "comparacion_real_vs_predicho.csv"
    )

    predicciones = cargar_csv_obligatorio(
        INPUT_PREDICCIONES,
        "predicciones_generacion.csv"
    )

    clasificacion = cargar_csv_obligatorio(
        INPUT_CLASIFICACION,
        "clasificacion_generacion.csv"
    )

    generacion_mensual["fecha_mes"] = pd.to_datetime(
        generacion_mensual["fecha_mes"],
        errors="coerce"
    )

    comparacion_modelo["fecha_mes"] = pd.to_datetime(
        comparacion_modelo["fecha_mes"],
        errors="coerce"
    )

    predicciones["fecha_mes"] = pd.to_datetime(
        predicciones["fecha_mes"],
        errors="coerce"
    )

    clasificacion["fecha_mes"] = pd.to_datetime(
        clasificacion["fecha_mes"],
        errors="coerce"
    )

    return {
        "generacion_mensual": generacion_mensual,
        "tecnologia": tecnologia,
        "ranking_recursos": ranking_recursos,
        "comparacion_modelo": comparacion_modelo,
        "predicciones": predicciones,
        "clasificacion": clasificacion,
    }


def crear_recomendacion(
    lista,
    modulo,
    tipo_alerta,
    prioridad,
    fecha,
    variable_analizada,
    descripcion,
    recomendacion,
    valor=None,
    unidad="",
    fuente="",
):
    """
    Agrega una recomendación a la lista general.
    """
    lista.append({
        "id_recomendacion": len(lista) + 1,
        "modulo": modulo,
        "tipo_alerta": tipo_alerta,
        "prioridad": prioridad,
        "fecha_referencia": fecha,
        "variable_analizada": variable_analizada,
        "descripcion": descripcion,
        "recomendacion": recomendacion,
        "valor": valor,
        "unidad": unidad,
        "fuente_datos": fuente
    })


# ============================================================
# REGLAS DEL AGENTE IA
# ============================================================

def regla_clasificacion_semaforo(datos, recomendaciones):
    """
    Genera recomendaciones a partir de la clasificación BAJO, NORMAL, ALTO.
    """
    clasificacion = datos["clasificacion"].copy()

    # Nos enfocamos en registros reales recientes y predicciones futuras.
    registros_relevantes = clasificacion[
        clasificacion["clasificacion_generacion"].isin(["BAJO", "ALTO"])
    ].copy()

    registros_relevantes = registros_relevantes.sort_values("fecha_mes")

    for _, fila in registros_relevantes.iterrows():
        fecha = fila["fecha_mes"].date()
        tipo_registro = fila["tipo_registro"]
        clasificacion_mes = fila["clasificacion_generacion"]
        variacion = fila["variacion_vs_historico_pct"]
        generacion = fila["generacion_total_gwh"]

        if clasificacion_mes == "BAJO":
            crear_recomendacion(
                recomendaciones,
                modulo="Clasificación tipo semáforo",
                tipo_alerta="Generación por debajo del histórico",
                prioridad="ALTA",
                fecha=fecha,
                variable_analizada="Generación mensual total",
                descripcion=(
                    f"La generación del mes se clasificó como BAJO, "
                    f"con una variación de {variacion:.2f}% frente al promedio histórico."
                ),
                recomendacion=(
                    "Revisar condiciones operativas, disponibilidad de recursos, "
                    "posibles mantenimientos, restricciones del sistema o factores externos "
                    "que hayan reducido la generación."
                ),
                valor=generacion,
                unidad="GWh",
                fuente="clasificacion_generacion.csv"
            )

        elif clasificacion_mes == "ALTO" and tipo_registro == "PREDICCION_FUTURA":
            crear_recomendacion(
                recomendaciones,
                modulo="Predicción futura",
                tipo_alerta="Generación futura superior al histórico",
                prioridad="MEDIA",
                fecha=fecha,
                variable_analizada="Generación mensual predicha",
                descripcion=(
                    f"La predicción futura se clasificó como ALTO, "
                    f"con una variación estimada de {variacion:.2f}% frente al promedio histórico."
                ),
                recomendacion=(
                    "Monitorear la capacidad operativa, la disponibilidad de recursos "
                    "y la planeación energética para atender una generación esperada superior "
                    "al comportamiento histórico."
                ),
                valor=generacion,
                unidad="GWh",
                fuente="clasificacion_generacion.csv"
            )


def regla_error_modelo(datos, recomendaciones):
    """
    Genera recomendaciones cuando el modelo tuvo errores porcentuales altos.
    """
    comparacion = datos["comparacion_modelo"].copy()

    errores_altos = comparacion[
        comparacion["error_porcentual"] >= UMBRAL_ERROR_MODELO_PCT
    ].copy()

    for _, fila in errores_altos.iterrows():
        fecha = fila["fecha_mes"].date()
        error_pct = fila["error_porcentual"]
        real = fila["generacion_real_gwh"]
        predicho = fila["generacion_predicha_gwh"]

        crear_recomendacion(
            recomendaciones,
            modulo="Modelo predictivo",
            tipo_alerta="Error de predicción elevado",
            prioridad="MEDIA",
            fecha=fecha,
            variable_analizada="Real vs predicho",
            descripcion=(
                f"El modelo presentó un error porcentual de {error_pct:.2f}% "
                f"en este mes. Valor real: {real:.2f} GWh, "
                f"valor predicho: {predicho:.2f} GWh."
            ),
            recomendacion=(
                "Revisar si en este mes existieron eventos atípicos, cambios de despacho, "
                "condiciones climáticas, mantenimientos o variaciones operativas que no estén "
                "representadas directamente en las variables del modelo."
            ),
            valor=error_pct,
            unidad="%",
            fuente="comparacion_real_vs_predicho.csv"
        )


def regla_dependencia_tecnologica(datos, recomendaciones):
    """
    Genera recomendaciones si una tecnología tiene participación muy alta.
    """
    tecnologia = datos["tecnologia"].copy()

    for anio in sorted(tecnologia["anio"].unique()):
        datos_anio = tecnologia[tecnologia["anio"] == anio].copy()

        if len(datos_anio) == 0:
            continue

        dominante = datos_anio.sort_values(
            "participacion_porcentual",
            ascending=False
        ).iloc[0]

        participacion = dominante["participacion_porcentual"]
        tipo_generacion = dominante["tipo_generacion"]

        if participacion >= UMBRAL_PARTICIPACION_ALTA:
            crear_recomendacion(
                recomendaciones,
                modulo="Análisis por tecnología",
                tipo_alerta="Alta dependencia tecnológica",
                prioridad="MEDIA",
                fecha=f"{int(anio)}",
                variable_analizada=f"Participación de {tipo_generacion}",
                descripcion=(
                    f"En el año {int(anio)}, la tecnología {tipo_generacion} "
                    f"representó el {participacion:.2f}% de la generación total."
                ),
                recomendacion=(
                    "Evaluar la diversificación de la matriz de generación y monitorear "
                    "riesgos asociados a la dependencia de una sola tecnología."
                ),
                valor=participacion,
                unidad="%",
                fuente="generacion_por_tecnologia.csv"
            )


def regla_crecimiento_tecnologico(datos, recomendaciones):
    """
    Detecta crecimiento fuerte de tecnologías entre años consecutivos.
    """
    tecnologia = datos["tecnologia"].copy()

    tabla = tecnologia[[
        "anio",
        "tipo_generacion",
        "generacion_total_gwh"
    ]].copy()

    tabla = tabla.sort_values(["tipo_generacion", "anio"])

    tabla["generacion_anterior_gwh"] = (
        tabla
        .groupby("tipo_generacion")["generacion_total_gwh"]
        .shift(1)
    )

    tabla["crecimiento_pct"] = (
        (
            tabla["generacion_total_gwh"]
            - tabla["generacion_anterior_gwh"]
        )
        / tabla["generacion_anterior_gwh"]
        * 100
    )

    crecimiento_fuerte = tabla[
        (tabla["crecimiento_pct"] >= UMBRAL_CRECIMIENTO_TECNOLOGIA)
        & (tabla["anio"] < 2026)
    ].copy()

    for _, fila in crecimiento_fuerte.iterrows():
        anio = int(fila["anio"])
        tipo_generacion = fila["tipo_generacion"]
        crecimiento = fila["crecimiento_pct"]

        prioridad = "MEDIA"

        if tipo_generacion in ["SOLAR", "EOLICA"]:
            prioridad = "ALTA"

        crear_recomendacion(
            recomendaciones,
            modulo="Tendencia tecnológica",
            tipo_alerta="Crecimiento significativo",
            prioridad=prioridad,
            fecha=f"{anio}",
            variable_analizada=f"Generación {tipo_generacion}",
            descripcion=(
                f"La generación {tipo_generacion} creció {crecimiento:.2f}% "
                f"frente al año anterior."
            ),
            recomendacion=(
                "Analizar las causas del crecimiento y evaluar oportunidades de inversión, "
                "planeación o expansión asociadas a esta tecnología."
            ),
            valor=crecimiento,
            unidad="%",
            fuente="generacion_por_tecnologia.csv"
        )


def regla_top_recursos(datos, recomendaciones):
    """
    Genera recomendaciones a partir de los recursos con mayor generación.
    """
    ranking = datos["ranking_recursos"].copy()

    if len(ranking) == 0:
        return

    top_1 = ranking.sort_values("ranking").iloc[0]

    crear_recomendacion(
        recomendaciones,
        modulo="Ranking de recursos",
        tipo_alerta="Recurso estratégico de alta generación",
        prioridad="MEDIA",
        fecha="2019-2026",
        variable_analizada=top_1["recurso"],
        descripcion=(
            f"El recurso {top_1['recurso']} ocupa el primer lugar del ranking "
            f"con {top_1['generacion_total_gwh']:.2f} GWh acumulados."
        ),
        recomendacion=(
            "Priorizar el seguimiento de este recurso en el dashboard, ya que su comportamiento "
            "puede influir de manera importante en la generación total del sistema."
        ),
        valor=top_1["generacion_total_gwh"],
        unidad="GWh",
        fuente="ranking_recursos.csv"
    )


def regla_anio_2026_parcial(datos, recomendaciones):
    """
    Genera una advertencia porque 2026 no está completo.
    """
    generacion_mensual = datos["generacion_mensual"].copy()

    fecha_maxima = generacion_mensual["fecha_mes"].max()

    if fecha_maxima.year == 2026 and fecha_maxima.month < 12:
        crear_recomendacion(
            recomendaciones,
            modulo="Calidad temporal de datos",
            tipo_alerta="Año parcial",
            prioridad="ALTA",
            fecha=str(fecha_maxima.date()),
            variable_analizada="Cobertura temporal 2026",
            descripcion=(
                f"La base contiene información de 2026 solo hasta "
                f"{fecha_maxima.date()}."
            ),
            recomendacion=(
                "No comparar el total acumulado de 2026 contra años completos sin aclarar "
                "que 2026 es un año parcial. Para análisis anuales, usar filtros o notas "
                "explicativas en el dashboard."
            ),
            valor=fecha_maxima.month,
            unidad="meses disponibles de 2026",
            fuente="generacion_mensual.csv"
        )


# ============================================================
# GENERACIÓN DEL AGENTE
# ============================================================

def generar_recomendaciones(datos):
    """
    Ejecuta todas las reglas del agente IA.
    """
    recomendaciones = []

    regla_clasificacion_semaforo(datos, recomendaciones)
    regla_error_modelo(datos, recomendaciones)
    regla_dependencia_tecnologica(datos, recomendaciones)
    regla_crecimiento_tecnologico(datos, recomendaciones)
    regla_top_recursos(datos, recomendaciones)
    regla_anio_2026_parcial(datos, recomendaciones)

    recomendaciones_df = pd.DataFrame(recomendaciones)

    if len(recomendaciones_df) == 0:
        recomendaciones_df = pd.DataFrame([{
            "id_recomendacion": 1,
            "modulo": "Agente IA",
            "tipo_alerta": "Sin alertas",
            "prioridad": "BAJA",
            "fecha_referencia": "",
            "variable_analizada": "General",
            "descripcion": "No se identificaron alertas relevantes con las reglas actuales.",
            "recomendacion": "Continuar monitoreando los indicadores del dashboard.",
            "valor": "",
            "unidad": "",
            "fuente_datos": "Reglas del agente"
        }])

    orden_prioridad = {
        "ALTA": 1,
        "MEDIA": 2,
        "BAJA": 3
    }

    recomendaciones_df["orden_prioridad"] = recomendaciones_df["prioridad"].map(
        orden_prioridad
    ).fillna(4)

    recomendaciones_df = recomendaciones_df.sort_values(
        ["orden_prioridad", "modulo", "id_recomendacion"]
    ).reset_index(drop=True)

    recomendaciones_df = recomendaciones_df.drop(columns=["orden_prioridad"])

    recomendaciones_df["id_recomendacion"] = range(1, len(recomendaciones_df) + 1)

    return recomendaciones_df


def crear_resumen_recomendaciones(recomendaciones_df):
    """
    Crea una tabla resumen por prioridad y módulo.
    """
    resumen_prioridad = (
        recomendaciones_df
        .groupby("prioridad", as_index=False)
        .agg(cantidad_recomendaciones=("id_recomendacion", "count"))
    )

    resumen_modulo = (
        recomendaciones_df
        .groupby("modulo", as_index=False)
        .agg(cantidad_recomendaciones=("id_recomendacion", "count"))
        .sort_values("cantidad_recomendaciones", ascending=False)
    )

    return resumen_prioridad, resumen_modulo


def guardar_salidas(recomendaciones_df, resumen_prioridad, resumen_modulo):
    """
    Guarda las recomendaciones en CSV, Excel y archivo TXT.
    """
    recomendaciones_df.to_csv(
        SALIDA_RECOMENDACIONES_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    recomendaciones_df.to_csv(
        SALIDA_RECOMENDACIONES_DASHBOARD,
        index=False,
        encoding="utf-8-sig"
    )

    with pd.ExcelWriter(SALIDA_RECOMENDACIONES_EXCEL, engine="xlsxwriter") as writer:
        recomendaciones_df.to_excel(writer, sheet_name="Recomendaciones", index=False)
        resumen_prioridad.to_excel(writer, sheet_name="Resumen_Prioridad", index=False)
        resumen_modulo.to_excel(writer, sheet_name="Resumen_Modulo", index=False)

    with open(SALIDA_RESUMEN_TXT, "w", encoding="utf-8") as archivo:
        archivo.write("ENERGYVIEW COLOMBIA - RESUMEN DEL AGENTE DE IA\n")
        archivo.write("=" * 80)
        archivo.write("\n\n")

        archivo.write("DESCRIPCIÓN DEL AGENTE\n")
        archivo.write("-" * 80)
        archivo.write(
            "El agente de IA analiza indicadores del dashboard, resultados del modelo "
            "predictivo y clasificación tipo semáforo para generar recomendaciones "
            "automáticas de apoyo a la toma de decisiones.\n\n"
        )

        archivo.write("REGLAS UTILIZADAS\n")
        archivo.write("-" * 80)
        archivo.write("1. Detectar meses con generación BAJA frente al histórico.\n")
        archivo.write("2. Detectar predicciones futuras ALTAS frente al histórico.\n")
        archivo.write("3. Detectar errores elevados del modelo predictivo.\n")
        archivo.write("4. Detectar alta dependencia tecnológica.\n")
        archivo.write("5. Detectar crecimiento significativo por tecnología.\n")
        archivo.write("6. Identificar recursos estratégicos de alta generación.\n")
        archivo.write("7. Advertir cuando un año está parcial, como ocurre con 2026.\n\n")

        archivo.write("RESUMEN POR PRIORIDAD\n")
        archivo.write("-" * 80)
        for _, fila in resumen_prioridad.iterrows():
            archivo.write(
                f"{fila['prioridad']}: "
                f"{fila['cantidad_recomendaciones']} recomendaciones\n"
            )

        archivo.write("\n")

        archivo.write("RESUMEN POR MÓDULO\n")
        archivo.write("-" * 80)
        for _, fila in resumen_modulo.iterrows():
            archivo.write(
                f"{fila['modulo']}: "
                f"{fila['cantidad_recomendaciones']} recomendaciones\n"
            )

        archivo.write("\n")

        archivo.write("TOP 10 RECOMENDACIONES\n")
        archivo.write("-" * 80)
        for _, fila in recomendaciones_df.head(10).iterrows():
            archivo.write(
                f"{fila['id_recomendacion']}. "
                f"[{fila['prioridad']}] "
                f"{fila['modulo']} - "
                f"{fila['tipo_alerta']}: "
                f"{fila['recomendacion']}\n"
            )

    with open(REPORTE_LOG, "w", encoding="utf-8") as log:
        log.write("ENERGYVIEW COLOMBIA - REPORTE DEL AGENTE IA\n")
        log.write("=" * 80)
        log.write("\n\n")
        log.write(f"Total recomendaciones generadas: {len(recomendaciones_df)}\n\n")

        log.write("Archivos generados:\n")
        log.write(f"{SALIDA_RECOMENDACIONES_CSV}\n")
        log.write(f"{SALIDA_RECOMENDACIONES_EXCEL}\n")
        log.write(f"{SALIDA_RECOMENDACIONES_DASHBOARD}\n")
        log.write(f"{SALIDA_RESUMEN_TXT}\n")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("======================================================")
    print("ENERGYVIEW COLOMBIA - AGENTE IA DE RECOMENDACIONES")
    print("======================================================")

    print("Cargando datos del dashboard, modelo y clasificación...")
    datos = cargar_datos()

    print("Ejecutando reglas del agente IA...")
    recomendaciones_df = generar_recomendaciones(datos)

    print("Creando resúmenes...")
    resumen_prioridad, resumen_modulo = crear_resumen_recomendaciones(
        recomendaciones_df
    )

    print("Guardando salidas...")
    guardar_salidas(
        recomendaciones_df,
        resumen_prioridad,
        resumen_modulo
    )

    print("\nAgente IA ejecutado correctamente.")
    print(f"Recomendaciones CSV: {SALIDA_RECOMENDACIONES_CSV}")
    print(f"Recomendaciones Excel: {SALIDA_RECOMENDACIONES_EXCEL}")
    print(f"Archivo para Power BI: {SALIDA_RECOMENDACIONES_DASHBOARD}")
    print(f"Resumen TXT: {SALIDA_RESUMEN_TXT}")
    print(f"Total recomendaciones generadas: {len(recomendaciones_df)}")


if __name__ == "__main__":
    main()