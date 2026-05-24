from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_MENSUAL = BASE_DIR / "outputs" / "dashboard" / "generacion_mensual.csv"
INPUT_PREDICCION = BASE_DIR / "outputs" / "modelo" / "serie_real_y_predicha_dashboard.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "modelo"
LOG_DIR = BASE_DIR / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SALIDA_CSV = OUTPUT_DIR / "clasificacion_generacion.csv"
SALIDA_EXCEL = OUTPUT_DIR / "clasificacion_generacion.xlsx"
REPORTE_LOG = LOG_DIR / "reporte_clasificacion_semaforo.txt"


# ============================================================
# CONFIGURACIÓN DE CLASIFICACIÓN
# ============================================================

UMBRAL_BAJO = -5.0
UMBRAL_ALTO = 5.0


# ============================================================
# FUNCIONES
# ============================================================

def cargar_datos():
    """
    Carga la tabla mensual y, si existe, la serie real/predicha del modelo.
    """
    if not INPUT_MENSUAL.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{INPUT_MENSUAL}\n"
            "Verifica que el Paso 5 esté aprobado."
        )

    print(f"Cargando generación mensual desde: {INPUT_MENSUAL}")

    mensual = pd.read_csv(INPUT_MENSUAL, encoding="utf-8-sig")
    mensual["fecha_mes"] = pd.to_datetime(mensual["fecha_mes"], errors="coerce")

    if INPUT_PREDICCION.exists():
        print(f"Cargando serie real y predicha desde: {INPUT_PREDICCION}")

        prediccion = pd.read_csv(INPUT_PREDICCION, encoding="utf-8-sig")
        prediccion["fecha_mes"] = pd.to_datetime(prediccion["fecha_mes"], errors="coerce")

    else:
        print("No se encontró la serie real y predicha. Se generará clasificación solo con histórico.")
        prediccion = None

    return mensual, prediccion


def clasificar_variacion(variacion_porcentual):
    """
    Clasifica la generación según la variación porcentual contra el promedio histórico.
    """
    if pd.isna(variacion_porcentual):
        return "SIN HISTORICO"

    if variacion_porcentual <= UMBRAL_BAJO:
        return "BAJO"

    if variacion_porcentual >= UMBRAL_ALTO:
        return "ALTO"

    return "NORMAL"


def asignar_prioridad(clasificacion):
    """
    Asigna una prioridad para el dashboard.
    """
    if clasificacion == "BAJO":
        return "ALTA"

    if clasificacion == "ALTO":
        return "MEDIA"

    if clasificacion == "NORMAL":
        return "BAJA"

    return "SIN PRIORIDAD"


def generar_mensaje(clasificacion, variacion):
    """
    Genera un mensaje corto para Power BI.
    """
    if clasificacion == "BAJO":
        return (
            f"La generación mensual está {abs(variacion):.2f}% por debajo "
            "del promedio histórico para este mes."
        )

    if clasificacion == "ALTO":
        return (
            f"La generación mensual está {variacion:.2f}% por encima "
            "del promedio histórico para este mes."
        )

    if clasificacion == "NORMAL":
        return (
            f"La generación mensual está dentro del rango esperado "
            f"frente al promedio histórico. Variación: {variacion:.2f}%."
        )

    return "No existe suficiente información histórica para clasificar este mes."


def crear_clasificacion_historica(mensual):
    """
    Clasifica cada mes real comparándolo contra el promedio histórico del mismo mes.
    """
    base = mensual.copy()
    base = base.sort_values("fecha_mes").reset_index(drop=True)

    # Para no usar el dato actual dentro de su propio promedio,
    # se calcula el promedio histórico acumulado por número de mes.
    base["promedio_historico_mes_gwh"] = np.nan

    for indice, fila in base.iterrows():
        mes_actual = fila["mes"]
        fecha_actual = fila["fecha_mes"]

        historico_previo = base[
            (base["mes"] == mes_actual)
            & (base["fecha_mes"] < fecha_actual)
        ]

        if len(historico_previo) > 0:
            promedio = historico_previo["generacion_total_gwh"].mean()
            base.loc[indice, "promedio_historico_mes_gwh"] = promedio

    base["variacion_vs_historico_pct"] = (
        (
            base["generacion_total_gwh"]
            - base["promedio_historico_mes_gwh"]
        )
        / base["promedio_historico_mes_gwh"]
        * 100
    )

    base["clasificacion_generacion"] = base["variacion_vs_historico_pct"].apply(
        clasificar_variacion
    )

    base["prioridad"] = base["clasificacion_generacion"].apply(asignar_prioridad)

    base["mensaje_clasificacion"] = base.apply(
        lambda fila: generar_mensaje(
            fila["clasificacion_generacion"],
            fila["variacion_vs_historico_pct"]
        ),
        axis=1
    )

    base["tipo_registro"] = "REAL_HISTORICO"

    columnas_salida = [
        "fecha_mes",
        "anio",
        "mes",
        "generacion_total_gwh",
        "promedio_historico_mes_gwh",
        "variacion_vs_historico_pct",
        "clasificacion_generacion",
        "prioridad",
        "mensaje_clasificacion",
        "tipo_registro"
    ]

    return base[columnas_salida]


def agregar_clasificacion_predicciones(clasificacion_historica, prediccion):
    """
    Agrega clasificación para predicciones futuras, si existe archivo del modelo.
    """
    if prediccion is None:
        return clasificacion_historica

    futuras = prediccion[prediccion["tipo_dato"] == "PREDICCION_FUTURA"].copy()

    if len(futuras) == 0:
        return clasificacion_historica

    registros_futuros = []

    for _, fila in futuras.iterrows():
        mes_actual = fila["mes"]

        historico_mismo_mes = clasificacion_historica[
            (clasificacion_historica["mes"] == mes_actual)
            & (clasificacion_historica["tipo_registro"] == "REAL_HISTORICO")
        ]

        promedio_historico = historico_mismo_mes["generacion_total_gwh"].mean()

        generacion_predicha = fila["generacion_predicha_gwh"]

        if pd.isna(promedio_historico) or promedio_historico == 0:
            variacion = np.nan
        else:
            variacion = (
                (generacion_predicha - promedio_historico)
                / promedio_historico
                * 100
            )

        clasificacion = clasificar_variacion(variacion)
        prioridad = asignar_prioridad(clasificacion)
        mensaje = generar_mensaje(clasificacion, variacion)

        registros_futuros.append({
            "fecha_mes": fila["fecha_mes"],
            "anio": fila["anio"],
            "mes": fila["mes"],
            "generacion_total_gwh": generacion_predicha,
            "promedio_historico_mes_gwh": promedio_historico,
            "variacion_vs_historico_pct": variacion,
            "clasificacion_generacion": clasificacion,
            "prioridad": prioridad,
            "mensaje_clasificacion": mensaje,
            "tipo_registro": "PREDICCION_FUTURA"
        })

    clasificacion_futura = pd.DataFrame(registros_futuros)

    salida = pd.concat(
        [clasificacion_historica, clasificacion_futura],
        ignore_index=True
    )

    salida = salida.sort_values("fecha_mes").reset_index(drop=True)

    return salida


def crear_resumen(clasificacion):
    """
    Crea una tabla resumen por clasificación y tipo de registro.
    """
    resumen = (
        clasificacion
        .groupby(["tipo_registro", "clasificacion_generacion"], as_index=False)
        .agg(
            cantidad_meses=("fecha_mes", "count"),
            generacion_promedio_gwh=("generacion_total_gwh", "mean"),
            variacion_promedio_pct=("variacion_vs_historico_pct", "mean")
        )
        .sort_values(["tipo_registro", "clasificacion_generacion"])
    )

    return resumen


def crear_reporte_log(clasificacion, resumen):
    """
    Crea un reporte TXT con los resultados de la clasificación.
    """
    with open(REPORTE_LOG, "w", encoding="utf-8") as log:
        log.write("ENERGYVIEW COLOMBIA - REPORTE DE CLASIFICACIÓN TIPO SEMÁFORO\n")
        log.write("=" * 80)
        log.write("\n\n")

        log.write("OBJETIVO\n")
        log.write("-" * 80)
        log.write(
            "Clasificar la generación mensual como BAJO, NORMAL o ALTO "
            "comparándola contra el promedio histórico del mismo mes.\n\n"
        )

        log.write("CRITERIO DE CLASIFICACIÓN\n")
        log.write("-" * 80)
        log.write(f"BAJO: variación menor o igual a {UMBRAL_BAJO}%\n")
        log.write(f"NORMAL: variación entre {UMBRAL_BAJO}% y {UMBRAL_ALTO}%\n")
        log.write(f"ALTO: variación mayor o igual a {UMBRAL_ALTO}%\n\n")

        log.write("RESUMEN POR CLASIFICACIÓN\n")
        log.write("-" * 80)
        for _, fila in resumen.iterrows():
            log.write(
                f"{fila['tipo_registro']} | "
                f"{fila['clasificacion_generacion']} | "
                f"Cantidad meses: {fila['cantidad_meses']} | "
                f"Generación promedio: {fila['generacion_promedio_gwh']:.4f} GWh | "
                f"Variación promedio: {fila['variacion_promedio_pct']:.4f}%\n"
            )

        log.write("\n")

        log.write("ÚLTIMOS 12 REGISTROS CLASIFICADOS\n")
        log.write("-" * 80)
        ultimos = clasificacion.tail(12)

        for _, fila in ultimos.iterrows():
            log.write(
                f"{fila['fecha_mes'].date()} | "
                f"{fila['tipo_registro']} | "
                f"Generación: {fila['generacion_total_gwh']:.4f} GWh | "
                f"Promedio histórico: {fila['promedio_historico_mes_gwh']:.4f} GWh | "
                f"Variación: {fila['variacion_vs_historico_pct']:.4f}% | "
                f"Clasificación: {fila['clasificacion_generacion']} | "
                f"Prioridad: {fila['prioridad']}\n"
            )

        log.write("\n")

        log.write("ARCHIVOS GENERADOS\n")
        log.write("-" * 80)
        log.write(f"{SALIDA_CSV}\n")
        log.write(f"{SALIDA_EXCEL}\n")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("======================================================")
    print("ENERGYVIEW COLOMBIA - CLASIFICACIÓN TIPO SEMÁFORO")
    print("======================================================")

    mensual, prediccion = cargar_datos()

    print("Generando clasificación histórica...")
    clasificacion_historica = crear_clasificacion_historica(mensual)

    print("Agregando clasificación de predicciones futuras...")
    clasificacion = agregar_clasificacion_predicciones(
        clasificacion_historica,
        prediccion
    )

    print("Creando resumen de clasificación...")
    resumen = crear_resumen(clasificacion)

    print("Guardando archivos...")
    clasificacion.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(SALIDA_EXCEL, engine="xlsxwriter") as writer:
        clasificacion.to_excel(writer, sheet_name="Clasificacion", index=False)
        resumen.to_excel(writer, sheet_name="Resumen", index=False)

    print("Creando reporte...")
    crear_reporte_log(clasificacion, resumen)

    print("\nClasificación tipo semáforo generada correctamente.")
    print(f"Archivo CSV: {SALIDA_CSV}")
    print(f"Archivo Excel: {SALIDA_EXCEL}")
    print(f"Reporte: {REPORTE_LOG}")


if __name__ == "__main__":
    main()