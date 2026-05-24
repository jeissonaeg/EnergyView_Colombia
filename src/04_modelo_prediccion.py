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

SALIDA_BASE_MODELO = OUTPUT_DIR / "base_modelo_mensual.csv"
SALIDA_COMPARACION_MODELOS = OUTPUT_DIR / "comparacion_modelos.csv"
SALIDA_COMPARACION_REAL_PREDICHO = OUTPUT_DIR / "comparacion_real_vs_predicho.csv"
SALIDA_PREDICCIONES_FUTURAS = OUTPUT_DIR / "predicciones_generacion.csv"
SALIDA_SERIE_DASHBOARD = OUTPUT_DIR / "serie_real_y_predicha_dashboard.csv"
SALIDA_EXCEL_METRICAS = OUTPUT_DIR / "metricas_modelo.xlsx"
SALIDA_GRAFICA_TEST = OUTPUT_DIR / "grafica_real_vs_predicho.png"
SALIDA_GRAFICA_FUTURA = OUTPUT_DIR / "grafica_prediccion_futura.png"

REPORTE_LOG = LOG_DIR / "reporte_modelo_prediccion.txt"


# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

ANIO_TEST = 2025
MESES_A_PREDECIR = 12


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def cargar_generacion_mensual():
    """
    Carga la tabla mensual generada en el Paso 5.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{INPUT_FILE}\n"
            "Verifica que el Paso 5 se haya ejecutado correctamente."
        )

    print(f"Cargando generación mensual desde: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")
    df = df.sort_values("fecha_mes").reset_index(drop=True)

    return df


def crear_variables_modelo(df):
    """
    Crea variables explicativas para el modelo predictivo mensual.
    """
    base = df.copy()

    base = base.sort_values("fecha_mes").reset_index(drop=True)

    base["indice_tiempo"] = np.arange(len(base))

    base["mes_sin"] = np.sin(2 * np.pi * base["mes"] / 12)
    base["mes_cos"] = np.cos(2 * np.pi * base["mes"] / 12)

    base["lag_1"] = base["generacion_total_gwh"].shift(1)
    base["lag_2"] = base["generacion_total_gwh"].shift(2)
    base["lag_3"] = base["generacion_total_gwh"].shift(3)
    base["lag_12"] = base["generacion_total_gwh"].shift(12)

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

    mape = np.mean(np.abs((y_real[mascara] - y_predicho[mascara]) / y_real[mascara])) * 100

    return mape


def evaluar_modelo(nombre_modelo, modelo, X_train, y_train, X_test, y_test):
    """
    Entrena y evalúa un modelo.
    """
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    mape = calcular_mape(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    metricas = {
        "modelo": nombre_modelo,
        "MAE_GWh": mae,
        "RMSE_GWh": rmse,
        "MAPE_porcentaje": mape,
        "R2": r2
    }

    return metricas, predicciones


def crear_predicciones_futuras(modelo, df_mensual, columnas_modelo, meses_a_predecir):
    """
    Genera predicciones futuras de manera recursiva.
    Usa los valores históricos y luego va usando sus propias predicciones.
    """
    historial = df_mensual[["fecha_mes", "generacion_total_gwh"]].copy()
    historial = historial.sort_values("fecha_mes").reset_index(drop=True)

    predicciones_futuras = []

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

        fila = {
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

        X_futuro = pd.DataFrame([fila])[columnas_modelo]

        prediccion = float(modelo.predict(X_futuro)[0])

        if prediccion < 0:
            prediccion = 0

        predicciones_futuras.append({
            "fecha_mes": siguiente_fecha,
            "anio": siguiente_fecha.year,
            "mes": siguiente_fecha.month,
            "generacion_predicha_gwh": prediccion,
            "tipo_prediccion": "FUTURA"
        })

        nueva_fila_historial = pd.DataFrame([{
            "fecha_mes": siguiente_fecha,
            "generacion_total_gwh": prediccion
        }])

        historial = pd.concat(
            [historial, nueva_fila_historial],
            ignore_index=True
        )

    return pd.DataFrame(predicciones_futuras)


def crear_serie_dashboard(df_mensual, comparacion_test, predicciones_futuras):
    """
    Crea una tabla combinada para Power BI:
    - Serie histórica real.
    - Predicciones del periodo de prueba.
    - Predicciones futuras.
    """
    historico = df_mensual[[
        "fecha_mes",
        "anio",
        "mes",
        "generacion_total_gwh"
    ]].copy()

    historico = historico.rename(
        columns={"generacion_total_gwh": "generacion_real_gwh"}
    )

    historico["generacion_predicha_gwh"] = np.nan
    historico["tipo_dato"] = "REAL"

    comparacion = comparacion_test[[
        "fecha_mes",
        "anio",
        "mes",
        "generacion_real_gwh",
        "generacion_predicha_gwh"
    ]].copy()

    comparacion["tipo_dato"] = "VALIDACION_MODELO"

    futuras = predicciones_futuras[[
        "fecha_mes",
        "anio",
        "mes",
        "generacion_predicha_gwh"
    ]].copy()

    futuras["generacion_real_gwh"] = np.nan
    futuras["tipo_dato"] = "PREDICCION_FUTURA"

    serie = pd.concat(
        [historico, comparacion, futuras],
        ignore_index=True
    )

    serie = serie.sort_values(["fecha_mes", "tipo_dato"]).reset_index(drop=True)

    return serie


def guardar_graficas(comparacion_test, predicciones_futuras):
    """
    Guarda dos gráficas simples del modelo.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        comparacion_test["fecha_mes"],
        comparacion_test["generacion_real_gwh"],
        marker="o",
        label="Real"
    )
    plt.plot(
        comparacion_test["fecha_mes"],
        comparacion_test["generacion_predicha_gwh"],
        marker="o",
        label="Predicho"
    )
    plt.title("Generación mensual real vs predicha")
    plt.xlabel("Fecha")
    plt.ylabel("Generación GWh")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SALIDA_GRAFICA_TEST, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        predicciones_futuras["fecha_mes"],
        predicciones_futuras["generacion_predicha_gwh"],
        marker="o",
        label="Predicción futura"
    )
    plt.title("Predicción futura de generación mensual")
    plt.xlabel("Fecha")
    plt.ylabel("Generación GWh")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(SALIDA_GRAFICA_FUTURA, dpi=150)
    plt.close()


def crear_reporte_log(
    df_mensual,
    base_modelo,
    metricas_df,
    mejor_modelo_nombre,
    comparacion_test,
    predicciones_futuras
):
    """
    Crea el reporte TXT del modelo.
    """
    mejor_fila = metricas_df.loc[
        metricas_df["modelo"] == mejor_modelo_nombre
    ].iloc[0]

    with open(REPORTE_LOG, "w", encoding="utf-8") as log:
        log.write("ENERGYVIEW COLOMBIA - REPORTE DEL MODELO DE PREDICCIÓN\n")
        log.write("=" * 75)
        log.write("\n\n")

        log.write("OBJETIVO DEL MODELO\n")
        log.write("-" * 75)
        log.write("Predecir la generación mensual total de energía en GWh.\n\n")

        log.write("BASE UTILIZADA\n")
        log.write("-" * 75)
        log.write(f"Meses disponibles en la base mensual: {len(df_mensual)}\n")
        log.write(f"Fecha mínima: {df_mensual['fecha_mes'].min()}\n")
        log.write(f"Fecha máxima: {df_mensual['fecha_mes'].max()}\n")
        log.write(f"Meses útiles después de crear rezagos: {len(base_modelo)}\n")
        log.write("\n")

        log.write("DIVISIÓN ENTRENAMIENTO / PRUEBA\n")
        log.write("-" * 75)
        log.write(f"Año usado para prueba: {ANIO_TEST}\n")
        log.write("Los datos anteriores se usan para entrenamiento.\n\n")

        log.write("MODELOS COMPARADOS\n")
        log.write("-" * 75)
        for _, fila in metricas_df.iterrows():
            log.write(f"Modelo: {fila['modelo']}\n")
            log.write(f"MAE GWh: {fila['MAE_GWh']:.4f}\n")
            log.write(f"RMSE GWh: {fila['RMSE_GWh']:.4f}\n")
            log.write(f"MAPE %: {fila['MAPE_porcentaje']:.4f}\n")
            log.write(f"R2: {fila['R2']:.4f}\n")
            log.write("\n")

        log.write("MEJOR MODELO SELECCIONADO\n")
        log.write("-" * 75)
        log.write(f"Modelo seleccionado: {mejor_modelo_nombre}\n")
        log.write(f"Criterio: menor RMSE.\n")
        log.write(f"RMSE del mejor modelo: {mejor_fila['RMSE_GWh']:.4f} GWh\n")
        log.write(f"MAPE del mejor modelo: {mejor_fila['MAPE_porcentaje']:.4f} %\n")
        log.write("\n")

        log.write("PREDICCIONES DEL PERIODO DE PRUEBA\n")
        log.write("-" * 75)
        for _, fila in comparacion_test.iterrows():
            log.write(
                f"{fila['fecha_mes'].date()} | "
                f"Real: {fila['generacion_real_gwh']:.4f} GWh | "
                f"Predicho: {fila['generacion_predicha_gwh']:.4f} GWh | "
                f"Error: {fila['error_gwh']:.4f} GWh | "
                f"Error %: {fila['error_porcentual']:.4f}%\n"
            )

        log.write("\n")

        log.write("PREDICCIONES FUTURAS\n")
        log.write("-" * 75)
        for _, fila in predicciones_futuras.iterrows():
            log.write(
                f"{fila['fecha_mes'].date()} | "
                f"Predicción: {fila['generacion_predicha_gwh']:.4f} GWh\n"
            )

        log.write("\n")

        log.write("ARCHIVOS GENERADOS\n")
        log.write("-" * 75)
        log.write(f"{SALIDA_BASE_MODELO}\n")
        log.write(f"{SALIDA_COMPARACION_MODELOS}\n")
        log.write(f"{SALIDA_COMPARACION_REAL_PREDICHO}\n")
        log.write(f"{SALIDA_PREDICCIONES_FUTURAS}\n")
        log.write(f"{SALIDA_SERIE_DASHBOARD}\n")
        log.write(f"{SALIDA_EXCEL_METRICAS}\n")
        log.write(f"{SALIDA_GRAFICA_TEST}\n")
        log.write(f"{SALIDA_GRAFICA_FUTURA}\n")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("====================================================")
    print("ENERGYVIEW COLOMBIA - MODELO DE PREDICCIÓN MENSUAL")
    print("====================================================")

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
            f"No se encontró el año {ANIO_TEST} para prueba. "
            "Se usará el último bloque de 12 meses como prueba."
        )

        train = base_modelo.iloc[:-12].copy()
        test = base_modelo.iloc[-12:].copy()

    X_train = train[columnas_modelo]
    y_train = train[variable_objetivo]

    X_test = test[columnas_modelo]
    y_test = test[variable_objetivo]

    modelos = {
        "Regresion_Lineal": LinearRegression(),
        "Random_Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2
        )
    }

    metricas = []
    predicciones_por_modelo = {}

    print("Entrenando y evaluando modelos...")

    for nombre, modelo in modelos.items():
        metricas_modelo, predicciones = evaluar_modelo(
            nombre,
            modelo,
            X_train,
            y_train,
            X_test,
            y_test
        )

        metricas.append(metricas_modelo)
        predicciones_por_modelo[nombre] = predicciones

    metricas_df = pd.DataFrame(metricas)
    metricas_df = metricas_df.sort_values("RMSE_GWh").reset_index(drop=True)

    mejor_modelo_nombre = metricas_df.iloc[0]["modelo"]

    print(f"Mejor modelo seleccionado: {mejor_modelo_nombre}")

    mejores_predicciones_test = predicciones_por_modelo[mejor_modelo_nombre]

    comparacion_test = test[[
        "fecha_mes",
        "anio",
        "mes",
        "generacion_total_gwh"
    ]].copy()

    comparacion_test = comparacion_test.rename(
        columns={"generacion_total_gwh": "generacion_real_gwh"}
    )

    comparacion_test["generacion_predicha_gwh"] = mejores_predicciones_test
    comparacion_test["error_gwh"] = (
        comparacion_test["generacion_real_gwh"]
        - comparacion_test["generacion_predicha_gwh"]
    )
    comparacion_test["error_absoluto_gwh"] = comparacion_test["error_gwh"].abs()
    comparacion_test["error_porcentual"] = (
        comparacion_test["error_absoluto_gwh"]
        / comparacion_test["generacion_real_gwh"]
        * 100
    )

    print("Entrenando modelo final con toda la información disponible...")

    mejor_modelo_base = modelos[mejor_modelo_nombre]
    mejor_modelo_final = clone(mejor_modelo_base)

    X_total = base_modelo[columnas_modelo]
    y_total = base_modelo[variable_objetivo]

    mejor_modelo_final.fit(X_total, y_total)

    print("Generando predicciones futuras...")

    predicciones_futuras = crear_predicciones_futuras(
        mejor_modelo_final,
        df_mensual,
        columnas_modelo,
        MESES_A_PREDECIR
    )

    serie_dashboard = crear_serie_dashboard(
        df_mensual,
        comparacion_test,
        predicciones_futuras
    )

    print("Guardando archivos del modelo...")

    base_modelo.to_csv(SALIDA_BASE_MODELO, index=False, encoding="utf-8-sig")
    metricas_df.to_csv(SALIDA_COMPARACION_MODELOS, index=False, encoding="utf-8-sig")
    comparacion_test.to_csv(SALIDA_COMPARACION_REAL_PREDICHO, index=False, encoding="utf-8-sig")
    predicciones_futuras.to_csv(SALIDA_PREDICCIONES_FUTURAS, index=False, encoding="utf-8-sig")
    serie_dashboard.to_csv(SALIDA_SERIE_DASHBOARD, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(SALIDA_EXCEL_METRICAS, engine="xlsxwriter") as writer:
        metricas_df.to_excel(writer, sheet_name="Metricas_Modelos", index=False)
        comparacion_test.to_excel(writer, sheet_name="Real_vs_Predicho", index=False)
        predicciones_futuras.to_excel(writer, sheet_name="Predicciones_Futuras", index=False)
        serie_dashboard.to_excel(writer, sheet_name="Serie_Dashboard", index=False)

    print("Guardando gráficas del modelo...")
    guardar_graficas(comparacion_test, predicciones_futuras)

    print("Creando reporte del modelo...")
    crear_reporte_log(
        df_mensual,
        base_modelo,
        metricas_df,
        mejor_modelo_nombre,
        comparacion_test,
        predicciones_futuras
    )

    print("\nModelo de predicción generado correctamente.")
    print(f"Mejor modelo: {mejor_modelo_nombre}")
    print(f"Reporte: {REPORTE_LOG}")
    print(f"Predicciones futuras: {SALIDA_PREDICCIONES_FUTURAS}")
    print(f"Serie para Power BI: {SALIDA_SERIE_DASHBOARD}")


if __name__ == "__main__":
    main()