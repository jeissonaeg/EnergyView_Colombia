from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "inputs" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORTE_TXT = OUTPUT_DIR / "reporte_inspeccion_datos.txt"
REPORTE_EXCEL = OUTPUT_DIR / "reporte_inspeccion_datos.xlsx"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_nombre_columna(columna):
    """
    Convierte el nombre de una columna a texto limpio.
    Esto ayuda a comparar columnas aunque vengan como números.
    """
    return str(columna).strip()


def detectar_columnas_horarias(columnas):
    """
    Detecta si existen las columnas horarias de 0 a 23.
    En algunas bases pueden venir como números enteros y en otras como texto.
    """
    columnas_texto = [normalizar_nombre_columna(col) for col in columnas]

    horas_esperadas = [str(hora) for hora in range(24)]

    horas_encontradas = [
        hora for hora in horas_esperadas
        if hora in columnas_texto
    ]

    horas_faltantes = [
        hora for hora in horas_esperadas
        if hora not in columnas_texto
    ]

    return horas_encontradas, horas_faltantes


def buscar_columna(columnas, nombre_buscado):
    """
    Busca una columna por nombre sin importar mayúsculas o minúsculas.
    """
    nombre_buscado = nombre_buscado.lower().strip()

    for col in columnas:
        if str(col).lower().strip() == nombre_buscado:
            return col

    return None


def inspeccionar_archivo(ruta_archivo):
    """
    Inspecciona un archivo Excel de generación y devuelve un diccionario
    con la información principal del archivo.
    """
    print(f"Inspeccionando archivo: {ruta_archivo.name}")

    df = pd.read_excel(ruta_archivo)

    columnas = list(df.columns)

    col_fecha = buscar_columna(columnas, "Fecha")
    col_recurso = buscar_columna(columnas, "Recurso")
    col_tipo_generacion = buscar_columna(columnas, "Tipo Generación")
    col_combustible = buscar_columna(columnas, "Combustible")
    col_codigo_agente = buscar_columna(columnas, "Código Agente")

    horas_encontradas, horas_faltantes = detectar_columnas_horarias(columnas)

    anios_detectados = ""

    if col_fecha is not None:
        fechas = pd.to_datetime(df[col_fecha], errors="coerce")
        anios = sorted(fechas.dropna().dt.year.unique().tolist())
        anios_detectados = ", ".join(str(anio) for anio in anios)

    tipos_generacion = ""
    if col_tipo_generacion is not None:
        valores = sorted(df[col_tipo_generacion].dropna().astype(str).unique().tolist())
        tipos_generacion = " | ".join(valores)

    combustibles = ""
    if col_combustible is not None:
        valores = sorted(df[col_combustible].dropna().astype(str).unique().tolist())
        combustibles = " | ".join(valores)

    cantidad_recursos = None
    if col_recurso is not None:
        cantidad_recursos = df[col_recurso].nunique(dropna=True)

    cantidad_agentes = None
    if col_codigo_agente is not None:
        cantidad_agentes = df[col_codigo_agente].nunique(dropna=True)

    total_celdas_vacias = int(df.isna().sum().sum())

    resultado = {
        "archivo": ruta_archivo.name,
        "filas": len(df),
        "columnas": len(df.columns),
        "tiene_fecha": col_fecha is not None,
        "tiene_recurso": col_recurso is not None,
        "tiene_tipo_generacion": col_tipo_generacion is not None,
        "tiene_combustible": col_combustible is not None,
        "tiene_codigo_agente": col_codigo_agente is not None,
        "horas_encontradas": len(horas_encontradas),
        "horas_faltantes": ", ".join(horas_faltantes),
        "anios_detectados": anios_detectados,
        "cantidad_recursos": cantidad_recursos,
        "cantidad_agentes": cantidad_agentes,
        "total_celdas_vacias": total_celdas_vacias,
        "tipos_generacion": tipos_generacion,
        "combustibles": combustibles,
        "lista_columnas": " | ".join(str(col) for col in columnas),
    }

    return resultado


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("==============================================")
    print("ENERGYVIEW COLOMBIA - INSPECCIÓN DE DATOS")
    print("==============================================")

    archivos_excel = sorted(INPUT_DIR.glob("*.xlsx"))

    if not archivos_excel:
        mensaje = (
            f"No se encontraron archivos Excel en la carpeta:\n"
            f"{INPUT_DIR}\n\n"
            f"Verifica que los archivos estén guardados en inputs/raw/"
        )
        print(mensaje)
        REPORTE_TXT.write_text(mensaje, encoding="utf-8")
        return

    resultados = []

    for archivo in archivos_excel:
        try:
            resultado = inspeccionar_archivo(archivo)
            resultados.append(resultado)
        except Exception as error:
            resultados.append({
                "archivo": archivo.name,
                "filas": "ERROR",
                "columnas": "ERROR",
                "tiene_fecha": "ERROR",
                "tiene_recurso": "ERROR",
                "tiene_tipo_generacion": "ERROR",
                "tiene_combustible": "ERROR",
                "tiene_codigo_agente": "ERROR",
                "horas_encontradas": "ERROR",
                "horas_faltantes": "ERROR",
                "anios_detectados": "ERROR",
                "cantidad_recursos": "ERROR",
                "cantidad_agentes": "ERROR",
                "total_celdas_vacias": "ERROR",
                "tipos_generacion": "ERROR",
                "combustibles": "ERROR",
                "lista_columnas": f"ERROR AL LEER ARCHIVO: {error}",
            })

    reporte_df = pd.DataFrame(resultados)

    reporte_df.to_excel(REPORTE_EXCEL, index=False)

    with open(REPORTE_TXT, "w", encoding="utf-8") as archivo_txt:
        archivo_txt.write("ENERGYVIEW COLOMBIA - REPORTE DE INSPECCIÓN\n")
        archivo_txt.write("=" * 60)
        archivo_txt.write("\n\n")

        for resultado in resultados:
            archivo_txt.write(f"Archivo: {resultado['archivo']}\n")
            archivo_txt.write(f"Filas: {resultado['filas']}\n")
            archivo_txt.write(f"Columnas: {resultado['columnas']}\n")
            archivo_txt.write(f"Tiene Fecha: {resultado['tiene_fecha']}\n")
            archivo_txt.write(f"Tiene Recurso: {resultado['tiene_recurso']}\n")
            archivo_txt.write(f"Tiene Tipo Generación: {resultado['tiene_tipo_generacion']}\n")
            archivo_txt.write(f"Tiene Combustible: {resultado['tiene_combustible']}\n")
            archivo_txt.write(f"Tiene Código Agente: {resultado['tiene_codigo_agente']}\n")
            archivo_txt.write(f"Horas encontradas: {resultado['horas_encontradas']} de 24\n")
            archivo_txt.write(f"Horas faltantes: {resultado['horas_faltantes']}\n")
            archivo_txt.write(f"Años detectados: {resultado['anios_detectados']}\n")
            archivo_txt.write(f"Cantidad de recursos: {resultado['cantidad_recursos']}\n")
            archivo_txt.write(f"Cantidad de agentes: {resultado['cantidad_agentes']}\n")
            archivo_txt.write(f"Total de celdas vacías: {resultado['total_celdas_vacias']}\n")
            archivo_txt.write(f"Tipos de generación: {resultado['tipos_generacion']}\n")
            archivo_txt.write(f"Combustibles: {resultado['combustibles']}\n")
            archivo_txt.write("-" * 60)
            archivo_txt.write("\n\n")

    print("\nInspección finalizada correctamente.")
    print(f"Reporte TXT generado en: {REPORTE_TXT}")
    print(f"Reporte Excel generado en: {REPORTE_EXCEL}")


if __name__ == "__main__":
    main()