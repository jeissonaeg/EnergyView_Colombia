from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "outputs" / "dashboard" / "generacion_mensual.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "modelo"
LOG_DIR = BASE_DIR / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SALIDA_BASE_MODELO = OUTPUT_DIR / "base_modelo_comparativo.csv"
SALIDA_METRICAS_DASHBOARD = OUTPUT_DIR / "comparacion_modelos_dashboard.csv"
SALIDA_VALIDACION_MODELOS = OUTPUT_DIR / "predicciones_validacion_modelos.csv"
SALIDA_FUTURAS_MODELOS = OUTPUT_DIR / "predicciones_futuras_modelos.csv"
SALIDA_SERIE_DASHBOARD = OUTPUT_DIR / "serie_modelos_dashboard.csv"
SALIDA_EXCEL = OUTPUT_DIR / "metricas_modelos_comparadas.xlsx"
SALIDA_GRAFICA_VALIDACION = OUTPUT_DIR / "grafica_comparacion_modelos_validacion.png"
SALIDA_GRAFICA_FUTURA = OUTPUT_DIR / "grafica_predicciones_futuras_modelos.png"

REPORTE_LOG = LOG_DIR / "reporte_modelos_comparativos.txt"


# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

ANIO_TEST = 2025
MESES_A_PREDECIR = 12


# ============================================================
# FUNCIONES
# ============================================================

def cargar_generacion_mensual():
    """
    Carga la tabla mensual generada para el dashboard.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{INPUT_FILE}\n"
            "Verifica que el Paso 5 esté aprobado."
        )

    print(f"Cargando generación mensual desde: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")

    df = df.sort_values("fecha_mes").reset_index(drop=True)

    return df


def crear_variables_modelo(df):
    """
    Crea variables temporales, estacionales y rezagos para entrenar los modelos.
    """
    base = df.copy()
    base = base.sort_values("fecha_mes").reset_index(drop=True)

    base["indice_tiempo"] = np.arange(len(base))

    # Variables estacionales para representar el comportamiento mensual.
    base["mes_sin"] = np.sin(2 * np.pi * base["mes"] / 12)
    base["mes_cos"] = np.cos(2 * np.pi * base["mes"] / 12)

    # Rezagos.
    base["lag_1"] = base["generacion_total_gwh"].shift(1)
    base["lag_2"] = base["generacion_total_gwh"].shift(2)
    base["lag_3"] = base["generacion_total_gwh"].shift(3)
    base["lag_12"] = base["generacion_total_gwh"].shift(12)

    # Promedios móviles.
    base["promedio_movil_3"] = (
        base["generacion_total_gwh"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    base["promedio_movil_6"] = (
        base["generacion_total_gwh"]
        .shift(1)
        .rolling(window=6)
        .mean()
    )

    # Cambio reciente.
    base["variacion_lag_1"] = base["lag_1"] - base["lag_2"]

    base_modelo = base.dropna().reset_index(drop=True)

    return base_modelo


def calcular_mape(y_real, y_predicho):
    """
    Calcula el error porcentual absoluto medio.
    """
    y_real = np.array(y_real)
    y_predicho = np.array(y_predicho)

    mascara = y_real != 0

    if mascara.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_real[mascara] - y_predicho[mascara]) / y_real[mascara])) * 100


def evaluar_modelo(nombre_modelo, tipo_modelo, modelo, X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un modelo de regresión.
    """
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    mape = calcular_mape(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    precision_aproximada = 100 - mape

    if precision_aproximada < 0:
        precision_aproximada = 0

    metricas = {
        "modelo": nombre_modelo,
        "tipo_modelo": tipo_modelo,
        "MAE_GWh": mae,
        "RMSE_GWh": rmse,
        "MAPE_porcentaje": mape,
        "R2": r2,
        "precision_aproximada_porcentaje": precision_aproximada
    }

    return metricas, predicciones


def crear_tabla_validacion(test, predicciones_por_modelo):
    """
    Crea una tabla larga con las predicciones de validación de ambos modelos.
    Esta tabla será útil para Power BI.
    """
    registros = []

    for nombre_modelo, info in predicciones_por_modelo.items():
        predicciones = info["predicciones"]
        tipo_modelo = info["tipo_modelo"]

        for indice, (_, fila) in enumerate(test.iterrows()):
            real = fila["generacion_total_gwh"]
            predicho = float(predicciones[indice])
            error = real - predicho
            error_absoluto = abs(error)

            if real != 0:
                error_porcentual = error_absoluto / real * 100
            else:
                error_porcentual = np.nan

            registros.append({
                "fecha_mes": fila["fecha_mes"],
                "anio": fila["anio"],
                "mes": fila["mes"],
                "modelo": nombre_modelo,
                "tipo_modelo": tipo_modelo,
                "periodo": "VALIDACION",
                "generacion_real_gwh": real,
                "generacion_predicha_gwh": predicho,
                "error_gwh": error,
                "error_absoluto_gwh": error_absoluto,
                "error_porcentual": error_porcentual
            })

    return pd.DataFrame(registros)


def crear_predicciones_futuras(modelo, nombre_modelo, tipo_modelo, df_mensual, columnas_modelo, meses_a_predecir):
    """
    Genera predicciones futuras recursivas para un modelo.
    """
    historial = df_mensual[["fecha_mes", "generacion_total_gwh"]].copy()
    historial = historial.sort_values("fecha_mes").reset_index(drop=True)

    registros = []

    for _ in range(meses_a_predecir):
        ultima_fecha = historial["fecha_mes"].max()
        siguiente_fecha = ultima_fecha + pd.DateOffset(months=1)

        valores = historial["generacion_total_gwh"].tolist()

        lag_1 = valores[-1]
        lag_2 = valores[-2] if len(valores) >= 2 else lag_1
        lag_3 = valores[-3] if len(valores) >= 3 else lag_2
        lag_12 = valores[-12] if len(valores) >= 12 else np.mean(valores)

        promedio_movil_3 = np.mean(valores[-3:]) if len(valores) >= 3 else np.mean(valores)
        promedio_movil_6 = np.mean(valores[-6:]) if len(valores) >= 6 else np.mean(valores)

        variacion_lag_1 = lag_1 - lag_2

        fila_futura = {
            "anio": siguiente_fecha.year,
            "mes": siguiente_fecha.month,
            "indice_tiempo": len(historial),
            "mes_sin": np.sin(2 * np.pi * siguiente_fecha.month / 12),
            "mes_cos": np.cos(2 * np.pi * siguiente_fecha.month / 12),
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_12": lag_12,
            "promedio_movil_3": promedio_movil_3,
            "promedio_movil_6": promedio_movil_6,
            "variacion_lag_1": variacion_lag_1
        }

        X_futuro = pd.DataFrame([fila_futura])[columnas_modelo]

        prediccion = float(modelo.predict(X_futuro)[0])

        if prediccion < 0:
            prediccion = 0

        registros.append({
            "fecha_mes": siguiente_fecha,
            "anio": siguiente_fecha.year,
            "mes": siguiente_fecha.month,
            "modelo": nombre_modelo,
            "tipo_modelo": tipo_modelo,
            "periodo": "PREDICCION_FUTURA",
            "generacion_predicha_gwh": prediccion
        })

        historial = pd.concat(
            [
                historial,
                pd.DataFrame([{
                    "fecha_mes": siguiente_fecha,
                    "generacion_total_gwh": prediccion
                }])
            ],
            ignore_index=True
        )

    return pd.DataFrame(registros)


def crear_serie_dashboard(df_mensual, validacion_modelos, futuras_modelos):
    """
    Crea una tabla larga para graficar en Power BI:
    - Serie real histórica.
    - Predicciones de validación por modelo.
    - Predicciones futuras por modelo.
    """
    registros = []

    for _, fila in df_mensual.iterrows():
        registros.append({
            "fecha_mes": fila["fecha_mes"],
            "anio": fila["anio"],
            "mes": fila["mes"],
            "serie": "Real",
            "modelo": "Real",
            "tipo_modelo": "Dato observado",
            "periodo": "REAL_HISTORICO",
            "generacion_gwh": fila["generacion_total_gwh"],
            "generacion_real_gwh": fila["generacion_total_gwh"],
            "generacion_predicha_gwh": np.nan,
            "error_porcentual": np.nan
        })

    for _, fila in validacion_modelos.iterrows():
        registros.append({
            "fecha_mes": fila["fecha_mes"],
            "anio": fila["anio"],
            "mes": fila["mes"],
            "serie": fila["modelo"],
            "modelo": fila["modelo"],
            "tipo_modelo": fila["tipo_modelo"],
            "periodo": "VALIDACION",
            "generacion_gwh": fila["generacion_predicha_gwh"],
            "generacion_real_gwh": fila["generacion_real_gwh"],
            "generacion_predicha_gwh": fila["generacion_predicha_gwh"],
            "error_porcentual": fila["error_porcentual"]
        })

    for _, fila in futuras_modelos.iterrows():
        registros.append({
            "fecha_mes": fila["fecha_mes"],
            "anio": fila["anio"],
            "mes": fila["mes"],
            "serie": fila["modelo"],
            "modelo": fila["modelo"],
            "tipo_modelo": fila["tipo_modelo"],
            "periodo": "PREDICCION_FUTURA",
            "generacion_gwh": fila["generacion_predicha_gwh"],
            "generacion_real_gwh": np.nan,
            "generacion_predicha_gwh": fila["generacion_predicha_gwh"],
            "error_porcentual": np.nan
        })

    serie = pd.DataFrame(registros)
    serie = serie.sort_values(["fecha_mes", "serie"]).reset_index(drop=True)

    return serie


def guardar_graficas(validacion_modelos, futuras_modelos):
    """
    Guarda gráficas comparativas para usar en informes o presentación.
    """
    plt.figure(figsize=(11, 6))

    real = (
        validacion_modelos[["fecha_mes", "generacion_real_gwh"]]
        .drop_duplicates()
        .sort_values("fecha_mes")
    )

    plt.plot(
        real["fecha_mes"],
        real["generacion_real_gwh"],
        marker="o",
        label="Real"
    )

    for modelo in validacion_modelos["modelo"].unique():
        datos_modelo = validacion_modelos[
            validacion_modelos["modelo"] == modelo
        ].sort_values("fecha_mes")

        plt.plot(
            datos_modelo["fecha_mes"],
            datos_modelo["generacion_predicha_gwh"],
            marker="o",
            label=modelo
        )

    plt.title("Comparación de modelos - Validación 2025")
    plt.xlabel("Fecha")
    plt.ylabel("Generación mensual GWh")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SALIDA_GRAFICA_VALIDACION, dpi=150)
    plt.close()

    plt.figure(figsize=(11, 6))

    for modelo in futuras_modelos["modelo"].unique():
        datos_modelo = futuras_modelos[
            futuras_modelos["modelo"] == modelo
        ].sort_values("fecha_mes")

        plt.plot(
            datos_modelo["fecha_mes"],
            datos_modelo["generacion_predicha_gwh"],
            marker="o",
            label=modelo
        )

    plt.title("Predicciones futuras por modelo")
    plt.xlabel("Fecha")
    plt.ylabel("Generación mensual GWh")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SALIDA_GRAFICA_FUTURA, dpi=150)
    plt.close()


def crear_reporte_log(metricas_df, validacion_modelos, futuras_modelos, mejor_modelo):
    """
    Crea reporte de comparación de modelos.
    """
    with open(REPORTE_LOG, "w", encoding="utf-8") as log:
        log.write("ENERGYVIEW COLOMBIA - REPORTE DE MODELOS COMPARATIVOS\n")
        log.write("=" * 80)
        log.write("\n\n")

        log.write("OBJETIVO\n")
        log.write("-" * 80)
        log.write(
            "Comparar un modelo lineal de Regresión Lineal contra un modelo no lineal "
            "Random Forest para predecir generación mensual total en GWh.\n\n"
        )

        log.write("MODELOS COMPARADOS\n")
        log.write("-" * 80)
        log.write("1. Regresión Lineal: modelo lineal e interpretable.\n")
        log.write("2. Random Forest: modelo no lineal basado en árboles de decisión.\n\n")

        log.write("MÉTRICAS DE VALIDACIÓN\n")
        log.write("-" * 80)

        for _, fila in metricas_df.iterrows():
            log.write(f"Modelo: {fila['modelo']}\n")
            log.write(f"Tipo de modelo: {fila['tipo_modelo']}\n")
            log.write(f"MAE GWh: {fila['MAE_GWh']:.4f}\n")
            log.write(f"RMSE GWh: {fila['RMSE_GWh']:.4f}\n")
            log.write(f"MAPE %: {fila['MAPE_porcentaje']:.4f}\n")
            log.write(f"R2: {fila['R2']:.4f}\n")
            log.write(f"Precisión aproximada %: {fila['precision_aproximada_porcentaje']:.4f}\n")
            log.write(f"Mejor modelo: {fila['es_mejor_modelo']}\n")
            log.write("\n")

        log.write("MODELO RECOMENDADO\n")
        log.write("-" * 80)
        log.write(f"Modelo recomendado según menor RMSE: {mejor_modelo}\n\n")

        log.write("PREDICCIONES FUTURAS GENERADAS\n")
        log.write("-" * 80)

        for _, fila in futuras_modelos.iterrows():
            log.write(
                f"{fila['fecha_mes'].date()} | "
                f"{fila['modelo']} | "
                f"{fila['generacion_predicha_gwh']:.4f} GWh\n"
            )

        log.write("\n")

        log.write("ARCHIVOS GENERADOS\n")
        log.write("-" * 80)
        log.write(f"{SALIDA_BASE_MODELO}\n")
        log.write(f"{SALIDA_METRICAS_DASHBOARD}\n")
        log.write(f"{SALIDA_VALIDACION_MODELOS}\n")
        log.write(f"{SALIDA_FUTURAS_MODELOS}\n")
        log.write(f"{SALIDA_SERIE_DASHBOARD}\n")
        log.write(f"{SALIDA_EXCEL}\n")
        log.write(f"{SALIDA_GRAFICA_VALIDACION}\n")
        log.write(f"{SALIDA_GRAFICA_FUTURA}\n")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("==============================================================")
    print("ENERGYVIEW COLOMBIA - COMPARACIÓN DE MODELOS PREDICTIVOS")
    print("==============================================================")

    df_mensual = cargar_generacion_mensual()

    print("Creando variables del modelo...")
    base_modelo = crear_variables_modelo(df_mensual)

    columnas_modelo = [
        "anio",
        "mes",
        "indice_tiempo",
        "mes_sin",
        "mes_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "promedio_movil_3",
        "promedio_movil_6",
        "variacion_lag_1"
    ]

    variable_objetivo = "generacion_total_gwh"

    train = base_modelo[base_modelo["anio"] < ANIO_TEST].copy()
    test = base_modelo[base_modelo["anio"] == ANIO_TEST].copy()

    if len(test) == 0:
        print(
            f"No se encontró el año {ANIO_TEST} para validación. "
            "Se usarán los últimos 12 meses como prueba."
        )

        train = base_modelo.iloc[:-12].copy()
        test = base_modelo.iloc[-12:].copy()

    X_train = train[columnas_modelo]
    y_train = train[variable_objetivo]

    X_test = test[columnas_modelo]
    y_test = test[variable_objetivo]

    modelos = {
        "Regresion_Lineal": {
            "tipo_modelo": "Lineal",
            "modelo": LinearRegression()
        },
        "Random_Forest": {
            "tipo_modelo": "No lineal",
            "modelo": RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                min_samples_leaf=2
            )
        }
    }

    metricas = []
    predicciones_por_modelo = {}

    print("Entrenando y evaluando Regresión Lineal y Random Forest...")

    for nombre_modelo, info in modelos.items():
        metricas_modelo, predicciones = evaluar_modelo(
            nombre_modelo=nombre_modelo,
            tipo_modelo=info["tipo_modelo"],
            modelo=info["modelo"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        metricas.append(metricas_modelo)

        predicciones_por_modelo[nombre_modelo] = {
            "tipo_modelo": info["tipo_modelo"],
            "predicciones": predicciones
        }

    metricas_df = pd.DataFrame(metricas)
    metricas_df = metricas_df.sort_values("RMSE_GWh").reset_index(drop=True)

    mejor_modelo = metricas_df.iloc[0]["modelo"]

    metricas_df["es_mejor_modelo"] = metricas_df["modelo"].apply(
        lambda modelo: "SI" if modelo == mejor_modelo else "NO"
    )

    print(f"Mejor modelo según RMSE: {mejor_modelo}")

    print("Creando tabla de validación para Power BI...")
    validacion_modelos = crear_tabla_validacion(test, predicciones_por_modelo)

    print("Entrenando modelos finales con toda la información disponible...")
    X_total = base_modelo[columnas_modelo]
    y_total = base_modelo[variable_objetivo]

    predicciones_futuras_lista = []

    for nombre_modelo, info in modelos.items():
        modelo_final = clone(info["modelo"])
        modelo_final.fit(X_total, y_total)

        futuras = crear_predicciones_futuras(
            modelo=modelo_final,
            nombre_modelo=nombre_modelo,
            tipo_modelo=info["tipo_modelo"],
            df_mensual=df_mensual,
            columnas_modelo=columnas_modelo,
            meses_a_predecir=MESES_A_PREDECIR
        )

        predicciones_futuras_lista.append(futuras)

    futuras_modelos = pd.concat(predicciones_futuras_lista, ignore_index=True)

    print("Creando serie para dashboard...")
    serie_dashboard = crear_serie_dashboard(
        df_mensual,
        validacion_modelos,
        futuras_modelos
    )

    print("Guardando archivos CSV...")
    base_modelo.to_csv(SALIDA_BASE_MODELO, index=False, encoding="utf-8-sig")
    metricas_df.to_csv(SALIDA_METRICAS_DASHBOARD, index=False, encoding="utf-8-sig")
    validacion_modelos.to_csv(SALIDA_VALIDACION_MODELOS, index=False, encoding="utf-8-sig")
    futuras_modelos.to_csv(SALIDA_FUTURAS_MODELOS, index=False, encoding="utf-8-sig")
    serie_dashboard.to_csv(SALIDA_SERIE_DASHBOARD, index=False, encoding="utf-8-sig")

    print("Guardando Excel comparativo...")
    with pd.ExcelWriter(SALIDA_EXCEL, engine="xlsxwriter") as writer:
        metricas_df.to_excel(writer, sheet_name="Metricas_Modelos", index=False)
        validacion_modelos.to_excel(writer, sheet_name="Validacion_Modelos", index=False)
        futuras_modelos.to_excel(writer, sheet_name="Predicciones_Futuras", index=False)
        serie_dashboard.to_excel(writer, sheet_name="Serie_Dashboard", index=False)
        base_modelo.to_excel(writer, sheet_name="Base_Modelo", index=False)

    print("Guardando gráficas...")
    guardar_graficas(validacion_modelos, futuras_modelos)

    print("Creando reporte...")
    crear_reporte_log(
        metricas_df=metricas_df,
        validacion_modelos=validacion_modelos,
        futuras_modelos=futuras_modelos,
        mejor_modelo=mejor_modelo
    )

    print("\nComparación de modelos generada correctamente.")
    print(f"Mejor modelo según RMSE: {mejor_modelo}")
    print(f"Métricas para dashboard: {SALIDA_METRICAS_DASHBOARD}")
    print(f"Validación modelos: {SALIDA_VALIDACION_MODELOS}")
    print(f"Predicciones futuras modelos: {SALIDA_FUTURAS_MODELOS}")
    print(f"Serie para Power BI: {SALIDA_SERIE_DASHBOARD}")
    print(f"Reporte: {REPORTE_LOG}")


if __name__ == "__main__":
    main()