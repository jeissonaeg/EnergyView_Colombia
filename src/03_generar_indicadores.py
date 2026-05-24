from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "outputs" / "data_limpia" / "generacion_diaria_consolidada.csv"

OUTPUT_DIR = BASE_DIR / "outputs" / "dashboard"
LOG_DIR = BASE_DIR / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SALIDA_BASE_DASHBOARD = OUTPUT_DIR / "base_dashboard.csv"
SALIDA_GENERACION_ANUAL = OUTPUT_DIR / "generacion_anual.csv"
SALIDA_GENERACION_MENSUAL = OUTPUT_DIR / "generacion_mensual.csv"
SALIDA_GENERACION_MENSUAL_RECURSO = OUTPUT_DIR / "generacion_mensual_recurso.csv"
SALIDA_GENERACION_TECNOLOGIA = OUTPUT_DIR / "generacion_por_tecnologia.csv"
SALIDA_GENERACION_COMBUSTIBLE = OUTPUT_DIR / "generacion_por_combustible.csv"
SALIDA_RANKING_RECURSOS = OUTPUT_DIR / "ranking_recursos.csv"
SALIDA_RANKING_AGENTES = OUTPUT_DIR / "ranking_agentes.csv"
SALIDA_KPIS = OUTPUT_DIR / "kpis_generales.csv"
SALIDA_EXCEL = OUTPUT_DIR / "indicadores_generales.xlsx"

REPORTE_LOG = LOG_DIR / "reporte_indicadores_dashboard.txt"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def cargar_base_consolidada():
    """
    Carga la base diaria consolidada generada en el Paso 4.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró la base consolidada en:\n{INPUT_FILE}\n"
            "Verifica que el Paso 4 se haya ejecutado correctamente."
        )

    print(f"Cargando base consolidada desde: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["fecha_mes"] = pd.to_datetime(df["fecha_mes"], errors="coerce")

    return df


def crear_base_dashboard(df):
    """
    Crea una base reducida para Power BI sin las columnas horarias.
    Esto hace que el dashboard sea más liviano.
    """
    columnas_dashboard = [
        "fecha",
        "fecha_mes",
        "anio",
        "mes",
        "dia",
        "recurso",
        "tipo_generacion",
        "combustible",
        "codigo_agente",
        "tipo_despacho",
        "generacion_diaria_kwh",
        "generacion_diaria_gwh",
        "archivo_origen"
    ]

    columnas_existentes = [
        columna for columna in columnas_dashboard
        if columna in df.columns
    ]

    base_dashboard = df[columnas_existentes].copy()

    return base_dashboard


def crear_generacion_anual(df):
    """
    Genera la tabla de generación total por año.
    """
    tabla = (
        df
        .groupby("anio", as_index=False)
        .agg(
            generacion_total_kwh=("generacion_diaria_kwh", "sum"),
            generacion_total_gwh=("generacion_diaria_gwh", "sum"),
            cantidad_registros=("generacion_diaria_gwh", "size"),
            recursos_unicos=("recurso", "nunique"),
            agentes_unicos=("codigo_agente", "nunique")
        )
        .sort_values("anio")
    )

    return tabla


def crear_generacion_mensual(df):
    """
    Genera la tabla de generación total por mes.
    """
    tabla = (
        df
        .groupby(["fecha_mes", "anio", "mes"], as_index=False)
        .agg(
            generacion_total_kwh=("generacion_diaria_kwh", "sum"),
            generacion_total_gwh=("generacion_diaria_gwh", "sum"),
            cantidad_registros=("generacion_diaria_gwh", "size"),
            recursos_unicos=("recurso", "nunique"),
            agentes_unicos=("codigo_agente", "nunique")
        )
        .sort_values(["anio", "mes"])
    )

    return tabla


def crear_generacion_mensual_recurso(df):
    """
    Genera una tabla mensual por recurso.
    Esta tabla será útil para Power BI y también para el modelo predictivo.
    """
    tabla = (
        df
        .groupby(
            [
                "fecha_mes",
                "anio",
                "mes",
                "recurso",
                "tipo_generacion",
                "combustible",
                "codigo_agente",
                "tipo_despacho"
            ],
            as_index=False
        )
        .agg(
            generacion_mensual_kwh=("generacion_diaria_kwh", "sum"),
            generacion_mensual_gwh=("generacion_diaria_gwh", "sum"),
            dias_reportados=("fecha", "nunique")
        )
        .sort_values(["fecha_mes", "tipo_generacion", "recurso"])
    )

    return tabla


def crear_generacion_por_tecnologia(df):
    """
    Genera generación anual por tipo de generación.
    """
    tabla = (
        df
        .groupby(["anio", "tipo_generacion"], as_index=False)
        .agg(
            generacion_total_kwh=("generacion_diaria_kwh", "sum"),
            generacion_total_gwh=("generacion_diaria_gwh", "sum"),
            cantidad_registros=("generacion_diaria_gwh", "size"),
            recursos_unicos=("recurso", "nunique")
        )
    )

    total_anual = (
        tabla
        .groupby("anio")["generacion_total_gwh"]
        .transform("sum")
    )

    tabla["participacion_porcentual"] = (
        tabla["generacion_total_gwh"] / total_anual * 100
    )

    tabla = tabla.sort_values(
        ["anio", "generacion_total_gwh"],
        ascending=[True, False]
    )

    return tabla


def crear_generacion_por_combustible(df):
    """
    Genera generación anual por combustible.
    """
    tabla = (
        df
        .groupby(["anio", "combustible"], as_index=False)
        .agg(
            generacion_total_kwh=("generacion_diaria_kwh", "sum"),
            generacion_total_gwh=("generacion_diaria_gwh", "sum"),
            cantidad_registros=("generacion_diaria_gwh", "size"),
            recursos_unicos=("recurso", "nunique")
        )
    )

    total_anual = (
        tabla
        .groupby("anio")["generacion_total_gwh"]
        .transform("sum")
    )

    tabla["participacion_porcentual"] = (
        tabla["generacion_total_gwh"] / total_anual * 100
    )

    tabla = tabla.sort_values(
        ["anio", "generacion_total_gwh"],
        ascending=[True, False]
    )

    return tabla


def crear_ranking_recursos(df):
    """
    Genera ranking de recursos por generación total acumulada.
    """
    tabla = (
        df
        .groupby(
            [
                "recurso",
                "tipo_generacion",
                "combustible",
                "codigo_agente"
            ],
            as_index=False
        )
        .agg(
            generacion_total_kwh=("generacion_diaria_kwh", "sum"),
            generacion_total_gwh=("generacion_diaria_gwh", "sum"),
            primera_fecha=("fecha", "min"),
            ultima_fecha=("fecha", "max"),
            dias_reportados=("fecha", "nunique")
        )
        .sort_values("generacion_total_gwh", ascending=False)
        .reset_index(drop=True)
    )

    tabla["ranking"] = tabla.index + 1

    columnas = [
        "ranking",
        "recurso",
        "tipo_generacion",
        "combustible",
        "codigo_agente",
        "generacion_total_kwh",
        "generacion_total_gwh",
        "primera_fecha",
        "ultima_fecha",
        "dias_reportados"
    ]

    tabla = tabla[columnas]

    return tabla


def crear_ranking_agentes(df):
    """
    Genera ranking de agentes por generación total acumulada.
    """
    tabla = (
        df
        .groupby("codigo_agente", as_index=False)
        .agg(
            generacion_total_kwh=("generacion_diaria_kwh", "sum"),
            generacion_total_gwh=("generacion_diaria_gwh", "sum"),
            recursos_unicos=("recurso", "nunique"),
            primera_fecha=("fecha", "min"),
            ultima_fecha=("fecha", "max"),
            dias_reportados=("fecha", "nunique")
        )
        .sort_values("generacion_total_gwh", ascending=False)
        .reset_index(drop=True)
    )

    tabla["ranking"] = tabla.index + 1

    columnas = [
        "ranking",
        "codigo_agente",
        "generacion_total_kwh",
        "generacion_total_gwh",
        "recursos_unicos",
        "primera_fecha",
        "ultima_fecha",
        "dias_reportados"
    ]

    tabla = tabla[columnas]

    return tabla


def crear_kpis_generales(df):
    """
    Crea una tabla pequeña con indicadores generales del proyecto.
    """
    generacion_total_gwh = df["generacion_diaria_gwh"].sum()

    tecnologia_dominante = (
        df
        .groupby("tipo_generacion")["generacion_diaria_gwh"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    combustible_dominante = (
        df
        .groupby("combustible")["generacion_diaria_gwh"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    anio_mayor_generacion = (
        df
        .groupby("anio")["generacion_diaria_gwh"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    mes_mayor_generacion = (
        df
        .groupby("fecha_mes")["generacion_diaria_gwh"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    kpis = [
        {
            "indicador": "Generación total consolidada GWh",
            "valor": round(generacion_total_gwh, 4)
        },
        {
            "indicador": "Cantidad de registros diarios",
            "valor": len(df)
        },
        {
            "indicador": "Años analizados",
            "valor": f"{int(df['anio'].min())} - {int(df['anio'].max())}"
        },
        {
            "indicador": "Fecha mínima",
            "valor": str(df["fecha"].min().date())
        },
        {
            "indicador": "Fecha máxima",
            "valor": str(df["fecha"].max().date())
        },
        {
            "indicador": "Recursos únicos",
            "valor": df["recurso"].nunique()
        },
        {
            "indicador": "Agentes únicos",
            "valor": df["codigo_agente"].nunique()
        },
        {
            "indicador": "Tipos de generación",
            "valor": df["tipo_generacion"].nunique()
        },
        {
            "indicador": "Combustibles únicos",
            "valor": df["combustible"].nunique()
        },
        {
            "indicador": "Tecnología dominante",
            "valor": tecnologia_dominante
        },
        {
            "indicador": "Combustible dominante",
            "valor": combustible_dominante
        },
        {
            "indicador": "Año con mayor generación",
            "valor": int(anio_mayor_generacion)
        },
        {
            "indicador": "Mes con mayor generación",
            "valor": str(pd.to_datetime(mes_mayor_generacion).date())
        },
        {
            "indicador": "Nota sobre 2026",
            "valor": "El año 2026 está parcial hasta enero"
        }
    ]

    return pd.DataFrame(kpis)


def guardar_excel_indicadores(
    kpis,
    generacion_anual,
    generacion_mensual,
    generacion_tecnologia,
    generacion_combustible,
    ranking_recursos,
    ranking_agentes
):
    """
    Guarda varias tablas en un solo archivo Excel con diferentes hojas.
    """
    with pd.ExcelWriter(SALIDA_EXCEL, engine="xlsxwriter") as writer:
        kpis.to_excel(writer, sheet_name="KPIS", index=False)
        generacion_anual.to_excel(writer, sheet_name="Generacion_Anual", index=False)
        generacion_mensual.to_excel(writer, sheet_name="Generacion_Mensual", index=False)
        generacion_tecnologia.to_excel(writer, sheet_name="Por_Tecnologia", index=False)
        generacion_combustible.to_excel(writer, sheet_name="Por_Combustible", index=False)
        ranking_recursos.to_excel(writer, sheet_name="Ranking_Recursos", index=False)
        ranking_agentes.to_excel(writer, sheet_name="Ranking_Agentes", index=False)


def crear_reporte_log(
    df,
    base_dashboard,
    generacion_anual,
    generacion_mensual,
    generacion_mensual_recurso,
    generacion_tecnologia,
    generacion_combustible,
    ranking_recursos,
    ranking_agentes,
    kpis
):
    """
    Crea un reporte TXT con el resumen de las tablas generadas.
    """
    with open(REPORTE_LOG, "w", encoding="utf-8") as log:
        log.write("ENERGYVIEW COLOMBIA - REPORTE DE INDICADORES PARA DASHBOARD\n")
        log.write("=" * 75)
        log.write("\n\n")

        log.write("BASE CONSOLIDADA UTILIZADA\n")
        log.write("-" * 75)
        log.write(f"Filas base original: {len(df)}\n")
        log.write(f"Filas base dashboard: {len(base_dashboard)}\n")
        log.write(f"Fecha mínima: {df['fecha'].min()}\n")
        log.write(f"Fecha máxima: {df['fecha'].max()}\n")
        log.write(f"Años disponibles: {sorted(df['anio'].dropna().unique().tolist())}\n")
        log.write(f"Generación total GWh: {df['generacion_diaria_gwh'].sum():.4f}\n")
        log.write("\n")

        log.write("TABLAS GENERADAS\n")
        log.write("-" * 75)
        log.write(f"base_dashboard.csv: {len(base_dashboard)} filas\n")
        log.write(f"generacion_anual.csv: {len(generacion_anual)} filas\n")
        log.write(f"generacion_mensual.csv: {len(generacion_mensual)} filas\n")
        log.write(f"generacion_mensual_recurso.csv: {len(generacion_mensual_recurso)} filas\n")
        log.write(f"generacion_por_tecnologia.csv: {len(generacion_tecnologia)} filas\n")
        log.write(f"generacion_por_combustible.csv: {len(generacion_combustible)} filas\n")
        log.write(f"ranking_recursos.csv: {len(ranking_recursos)} filas\n")
        log.write(f"ranking_agentes.csv: {len(ranking_agentes)} filas\n")
        log.write(f"kpis_generales.csv: {len(kpis)} filas\n")
        log.write("\n")

        log.write("TOP 10 RECURSOS POR GENERACIÓN GWh\n")
        log.write("-" * 75)
        for _, fila in ranking_recursos.head(10).iterrows():
            log.write(
                f"{int(fila['ranking'])}. "
                f"{fila['recurso']} | "
                f"{fila['tipo_generacion']} | "
                f"{fila['generacion_total_gwh']:.4f} GWh\n"
            )

        log.write("\n")

        log.write("TOP 10 AGENTES POR GENERACIÓN GWh\n")
        log.write("-" * 75)
        for _, fila in ranking_agentes.head(10).iterrows():
            log.write(
                f"{int(fila['ranking'])}. "
                f"{fila['codigo_agente']} | "
                f"{fila['generacion_total_gwh']:.4f} GWh\n"
            )

        log.write("\n")

        log.write("KPIS GENERALES\n")
        log.write("-" * 75)
        for _, fila in kpis.iterrows():
            log.write(f"{fila['indicador']}: {fila['valor']}\n")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("====================================================")
    print("ENERGYVIEW COLOMBIA - GENERACIÓN DE INDICADORES")
    print("====================================================")

    df = cargar_base_consolidada()

    print("Creando base para Power BI...")
    base_dashboard = crear_base_dashboard(df)

    print("Creando generación anual...")
    generacion_anual = crear_generacion_anual(df)

    print("Creando generación mensual...")
    generacion_mensual = crear_generacion_mensual(df)

    print("Creando generación mensual por recurso...")
    generacion_mensual_recurso = crear_generacion_mensual_recurso(df)

    print("Creando generación por tecnología...")
    generacion_tecnologia = crear_generacion_por_tecnologia(df)

    print("Creando generación por combustible...")
    generacion_combustible = crear_generacion_por_combustible(df)

    print("Creando ranking de recursos...")
    ranking_recursos = crear_ranking_recursos(df)

    print("Creando ranking de agentes...")
    ranking_agentes = crear_ranking_agentes(df)

    print("Creando KPIs generales...")
    kpis = crear_kpis_generales(df)

    print("Guardando archivos CSV...")

    base_dashboard.to_csv(SALIDA_BASE_DASHBOARD, index=False, encoding="utf-8-sig")
    generacion_anual.to_csv(SALIDA_GENERACION_ANUAL, index=False, encoding="utf-8-sig")
    generacion_mensual.to_csv(SALIDA_GENERACION_MENSUAL, index=False, encoding="utf-8-sig")
    generacion_mensual_recurso.to_csv(SALIDA_GENERACION_MENSUAL_RECURSO, index=False, encoding="utf-8-sig")
    generacion_tecnologia.to_csv(SALIDA_GENERACION_TECNOLOGIA, index=False, encoding="utf-8-sig")
    generacion_combustible.to_csv(SALIDA_GENERACION_COMBUSTIBLE, index=False, encoding="utf-8-sig")
    ranking_recursos.to_csv(SALIDA_RANKING_RECURSOS, index=False, encoding="utf-8-sig")
    ranking_agentes.to_csv(SALIDA_RANKING_AGENTES, index=False, encoding="utf-8-sig")
    kpis.to_csv(SALIDA_KPIS, index=False, encoding="utf-8-sig")

    print("Guardando archivo Excel de indicadores...")
    guardar_excel_indicadores(
        kpis,
        generacion_anual,
        generacion_mensual,
        generacion_tecnologia,
        generacion_combustible,
        ranking_recursos,
        ranking_agentes
    )

    print("Creando reporte de indicadores...")
    crear_reporte_log(
        df,
        base_dashboard,
        generacion_anual,
        generacion_mensual,
        generacion_mensual_recurso,
        generacion_tecnologia,
        generacion_combustible,
        ranking_recursos,
        ranking_agentes,
        kpis
    )

    print("\nIndicadores generados correctamente.")
    print(f"Base dashboard: {SALIDA_BASE_DASHBOARD}")
    print(f"Indicadores Excel: {SALIDA_EXCEL}")
    print(f"Reporte: {REPORTE_LOG}")


if __name__ == "__main__":
    main()