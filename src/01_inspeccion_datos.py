from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata


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

def quitar_acentos(texto):
    """
    Elimina acentos para comparar nombres de columnas.
    Ejemplo: 'Código Agente' -> 'Codigo Agente'
    """
    texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(caracter for caracter in texto if not unicodedata.combining(caracter))


def normalizar_texto(valor):
    """
    Normaliza cualquier valor para poder compararlo.
    Sirve para detectar columnas aunque vengan como:
    - 0
    - 0.0
    - 0.00
    - '0'
    """
    if pd.isna(valor):
        return ""

    if isinstance(valor, (int, float, np.integer, np.floating)):
        numero = float(valor)

        if numero.is_integer():
            return str(int(numero))

        return str(valor).strip()

    texto = str(valor).strip()

    # Detectar horas escritas como texto decimal: '0.00', '1.0', etc.
    try:
        numero = float(texto.replace(",", "."))

        if numero.is_integer() and 0 <= int(numero) <= 23:
            return str(int(numero))

    except Exception:
        pass

    texto = quitar_acentos(texto)
    texto = texto.lower()
    texto = " ".join(texto.split())

    return texto


def buscar_columna(columnas, nombre_buscado):
    """
    Busca una columna por nombre, ignorando mayúsculas, minúsculas y acentos.
    """
    nombre_buscado_normalizado = normalizar_texto(nombre_buscado)

    for columna in columnas:
        if normalizar_texto(columna) == nombre_buscado_normalizado:
            return columna

    return None


def detectar_columnas_horarias(columnas):
    """
    Detecta si existen las columnas horarias de 0 a 23.
    """
    columnas_normalizadas = [normalizar_texto(columna) for columna in columnas]

    horas_esperadas = [str(hora) for hora in range(24)]

    horas_encontradas = [
        hora for hora in horas_esperadas
        if hora in columnas_normalizadas
    ]

    horas_faltantes = [
        hora for hora in horas_esperadas
        if hora not in columnas_normalizadas
    ]

    return horas_encontradas, horas_faltantes


def evaluar_fila_como_encabezado(valores_fila):
    """
    Evalúa si una fila parece ser el encabezado real de la tabla.
    Le da puntaje a una fila si contiene campos importantes y horas.
    """
    valores_normalizados = [normalizar_texto(valor) for valor in valores_fila]

    campos_importantes = [
        "fecha",
        "recurso",
        "tipo generacion",
        "combustible",
        "codigo agente",
        "tipo despacho",
        "version"
    ]

    horas_esperadas = [str(hora) for hora in range(24)]

    campos_encontrados = sum(
        1 for campo in campos_importantes
        if campo in valores_normalizados
    )

    horas_encontradas = sum(
        1 for hora in horas_esperadas
        if hora in valores_normalizados
    )

    puntaje = campos_encontrados * 10 + horas_encontradas

    return puntaje, campos_encontrados, horas_encontradas


def detectar_fila_encabezado(ruta_archivo, max_filas=40):
    """
    Lee las primeras filas del Excel y detecta cuál fila contiene los encabezados reales.
    """
    muestra = pd.read_excel(ruta_archivo, header=None, nrows=max_filas)

    candidatos = []

    for indice_fila, fila in muestra.iterrows():
        puntaje, campos_encontrados, horas_encontradas = evaluar_fila_como_encabezado(fila.tolist())

        candidatos.append({
            "indice_fila_python": indice_fila,
            "fila_excel": indice_fila + 1,
            "puntaje": puntaje,
            "campos_encontrados": campos_encontrados,
            "horas_encontradas": horas_encontradas
        })

    candidatos_ordenados = sorted(
        candidatos,
        key=lambda item: item["puntaje"],
        reverse=True
    )

    mejor_candidato = candidatos_ordenados[0]

    if mejor_candidato["puntaje"] < 20:
        raise ValueError(
            "No se pudo detectar automáticamente la fila de encabezado. "
            "Revisa manualmente dónde empieza la tabla."
        )

    return mejor_candidato


def leer_excel_con_encabezado_detectado(ruta_archivo):
    """
    Detecta la fila del encabezado y luego lee el Excel usando esa fila.
    """
    encabezado = detectar_fila_encabezado(ruta_archivo)

    df = pd.read_excel(
        ruta_archivo,
        header=encabezado["indice_fila_python"]
    )

    # Eliminar filas completamente vacías
    df = df.dropna(how="all")

    return df, encabezado


def inspeccionar_archivo(ruta_archivo):
    """
    Inspecciona un archivo Excel de generación y devuelve un diccionario
    con la información principal del archivo.
    """
    print(f"Inspeccionando archivo: {ruta_archivo.name}")

    df, encabezado = leer_excel_con_encabezado_detectado(ruta_archivo)

    columnas = list(df.columns)

    col_fecha = buscar_columna(columnas, "Fecha")
    col_recurso = buscar_columna(columnas, "Recurso")
    col_tipo_generacion = buscar_columna(columnas, "Tipo Generación")
    col_combustible = buscar_columna(columnas, "Combustible")
    col_codigo_agente = buscar_columna(columnas, "Código Agente")
    col_tipo_despacho = buscar_columna(columnas, "Tipo Despacho")

    horas_encontradas, horas_faltantes = detectar_columnas_horarias(columnas)

    anios_detectados = ""

    if col_fecha is not None:
        fechas = pd.to_datetime(df[col_fecha], errors="coerce")
        anios = sorted(fechas.dropna().dt.year.unique().tolist())
        anios_detectados = ", ".join(str(anio) for anio in anios)

    tipos_generacion = ""

    if col_tipo_generacion is not None:
        valores = sorted(
            df[col_tipo_generacion]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        tipos_generacion = " | ".join(valores)

    combustibles = ""

    if col_combustible is not None:
        valores = sorted(
            df[col_combustible]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        combustibles = " | ".join(valores)

    cantidad_recursos = None

    if col_recurso is not None:
        cantidad_recursos = df[col_recurso].nunique(dropna=True)

    cantidad_agentes = None

    if col_codigo_agente is not None:
        cantidad_agentes = df[col_codigo_agente].nunique(dropna=True)

    total_celdas_vacias = int(df.isna().sum().sum())

    estructura_basica_ok = (
        col_fecha is not None
        and col_recurso is not None
        and col_tipo_generacion is not None
        and col_combustible is not None
        and col_codigo_agente is not None
        and len(horas_encontradas) == 24
    )

    resultado = {
        "archivo": ruta_archivo.name,
        "fila_encabezado_excel": encabezado["fila_excel"],
        "puntaje_encabezado": encabezado["puntaje"],
        "filas": len(df),
        "columnas": len(df.columns),
        "tiene_fecha": col_fecha is not None,
        "tiene_recurso": col_recurso is not None,
        "tiene_tipo_generacion": col_tipo_generacion is not None,
        "tiene_combustible": col_combustible is not None,
        "tiene_codigo_agente": col_codigo_agente is not None,
        "tiene_tipo_despacho": col_tipo_despacho is not None,
        "horas_encontradas": len(horas_encontradas),
        "horas_faltantes": ", ".join(horas_faltantes),
        "anios_detectados": anios_detectados,
        "cantidad_recursos": cantidad_recursos,
        "cantidad_agentes": cantidad_agentes,
        "total_celdas_vacias": total_celdas_vacias,
        "tipos_generacion": tipos_generacion,
        "combustibles": combustibles,
        "estructura_basica_ok": estructura_basica_ok,
        "lista_columnas": " | ".join(str(columna) for columna in columnas),
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
                "fila_encabezado_excel": "ERROR",
                "puntaje_encabezado": "ERROR",
                "filas": "ERROR",
                "columnas": "ERROR",
                "tiene_fecha": "ERROR",
                "tiene_recurso": "ERROR",
                "tiene_tipo_generacion": "ERROR",
                "tiene_combustible": "ERROR",
                "tiene_codigo_agente": "ERROR",
                "tiene_tipo_despacho": "ERROR",
                "horas_encontradas": "ERROR",
                "horas_faltantes": "ERROR",
                "anios_detectados": "ERROR",
                "cantidad_recursos": "ERROR",
                "cantidad_agentes": "ERROR",
                "total_celdas_vacias": "ERROR",
                "tipos_generacion": "ERROR",
                "combustibles": "ERROR",
                "estructura_basica_ok": "ERROR",
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
            archivo_txt.write(f"Fila de encabezado detectada en Excel: {resultado['fila_encabezado_excel']}\n")
            archivo_txt.write(f"Puntaje de encabezado: {resultado['puntaje_encabezado']}\n")
            archivo_txt.write(f"Filas: {resultado['filas']}\n")
            archivo_txt.write(f"Columnas: {resultado['columnas']}\n")
            archivo_txt.write(f"Tiene Fecha: {resultado['tiene_fecha']}\n")
            archivo_txt.write(f"Tiene Recurso: {resultado['tiene_recurso']}\n")
            archivo_txt.write(f"Tiene Tipo Generación: {resultado['tiene_tipo_generacion']}\n")
            archivo_txt.write(f"Tiene Combustible: {resultado['tiene_combustible']}\n")
            archivo_txt.write(f"Tiene Código Agente: {resultado['tiene_codigo_agente']}\n")
            archivo_txt.write(f"Tiene Tipo Despacho: {resultado['tiene_tipo_despacho']}\n")
            archivo_txt.write(f"Horas encontradas: {resultado['horas_encontradas']} de 24\n")
            archivo_txt.write(f"Horas faltantes: {resultado['horas_faltantes']}\n")
            archivo_txt.write(f"Años detectados: {resultado['anios_detectados']}\n")
            archivo_txt.write(f"Cantidad de recursos: {resultado['cantidad_recursos']}\n")
            archivo_txt.write(f"Cantidad de agentes: {resultado['cantidad_agentes']}\n")
            archivo_txt.write(f"Total de celdas vacías: {resultado['total_celdas_vacias']}\n")
            archivo_txt.write(f"Tipos de generación: {resultado['tipos_generacion']}\n")
            archivo_txt.write(f"Combustibles: {resultado['combustibles']}\n")
            archivo_txt.write(f"Estructura básica OK: {resultado['estructura_basica_ok']}\n")
            archivo_txt.write("-" * 60)
            archivo_txt.write("\n\n")

    print("\nInspección finalizada correctamente.")
    print(f"Reporte TXT generado en: {REPORTE_TXT}")
    print(f"Reporte Excel generado en: {REPORTE_EXCEL}")


if __name__ == "__main__":
    main()