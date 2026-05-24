from pathlib import Path
import pandas as pd
import numpy as np
import unicodedata


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = BASE_DIR / "inputs" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs" / "data_limpia"
LOG_DIR = BASE_DIR / "outputs" / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SALIDA_CSV = OUTPUT_DIR / "generacion_diaria_consolidada.csv"
SALIDA_EXCEL = OUTPUT_DIR / "generacion_diaria_consolidada.xlsx"
REPORTE_LOG = LOG_DIR / "reporte_limpieza_consolidacion.txt"


# ============================================================
# FUNCIONES PARA NORMALIZAR TEXTO Y COLUMNAS
# ============================================================

def quitar_acentos(texto):
    """
    Elimina acentos de un texto.
    Ejemplo: Código -> Codigo
    """
    texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(caracter for caracter in texto if not unicodedata.combining(caracter))


def normalizar_texto(valor):
    """
    Normaliza un valor para poder comparar nombres de columnas.
    También ayuda a reconocer columnas horarias como 0, 1, 2, ..., 23.
    """
    if pd.isna(valor):
        return ""

    if isinstance(valor, (int, float, np.integer, np.floating)):
        numero = float(valor)

        if numero.is_integer():
            return str(int(numero))

        return str(valor).strip()

    texto = str(valor).strip()

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
    Busca una columna ignorando acentos, mayúsculas y minúsculas.
    """
    nombre_buscado_normalizado = normalizar_texto(nombre_buscado)

    for columna in columnas:
        if normalizar_texto(columna) == nombre_buscado_normalizado:
            return columna

    return None


def detectar_columna_hora(columnas, hora):
    """
    Busca una columna horaria específica.
    Ejemplo: hora = 0, busca columna 0 aunque venga como texto o número.
    """
    hora_texto = str(hora)

    for columna in columnas:
        if normalizar_texto(columna) == hora_texto:
            return columna

    return None


# ============================================================
# FUNCIONES PARA DETECTAR ENCABEZADO REAL
# ============================================================

def evaluar_fila_como_encabezado(valores_fila):
    """
    Evalúa si una fila parece ser el encabezado real de la tabla.
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

    return puntaje


def detectar_fila_encabezado(ruta_archivo, max_filas=40):
    """
    Detecta automáticamente la fila donde están los encabezados reales.
    """
    muestra = pd.read_excel(ruta_archivo, header=None, nrows=max_filas)

    mejor_indice = None
    mejor_puntaje = -1

    for indice_fila, fila in muestra.iterrows():
        puntaje = evaluar_fila_como_encabezado(fila.tolist())

        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_indice = indice_fila

    if mejor_puntaje < 20:
        raise ValueError(
            f"No se pudo detectar encabezado en el archivo {ruta_archivo.name}"
        )

    return mejor_indice, mejor_puntaje


def leer_excel_generacion(ruta_archivo):
    """
    Lee un archivo Excel usando la fila de encabezado detectada automáticamente.
    """
    indice_encabezado, puntaje = detectar_fila_encabezado(ruta_archivo)

    df = pd.read_excel(
        ruta_archivo,
        header=indice_encabezado
    )

    df = df.dropna(how="all")

    return df, indice_encabezado, puntaje


# ============================================================
# FUNCIÓN PRINCIPAL DE LIMPIEZA DE CADA ARCHIVO
# ============================================================

def limpiar_archivo(ruta_archivo):
    """
    Limpia un archivo anual de generación y devuelve una base diaria.
    """
    print(f"Procesando archivo: {ruta_archivo.name}")

    df, indice_encabezado, puntaje = leer_excel_generacion(ruta_archivo)

    columnas = list(df.columns)

    col_fecha = buscar_columna(columnas, "Fecha")
    col_recurso = buscar_columna(columnas, "Recurso")
    col_tipo_generacion = buscar_columna(columnas, "Tipo Generación")
    col_combustible = buscar_columna(columnas, "Combustible")
    col_codigo_agente = buscar_columna(columnas, "Código Agente")
    col_tipo_despacho = buscar_columna(columnas, "Tipo Despacho")

    columnas_obligatorias = {
        "Fecha": col_fecha,
        "Recurso": col_recurso,
        "Tipo Generación": col_tipo_generacion,
        "Combustible": col_combustible,
        "Código Agente": col_codigo_agente,
        "Tipo Despacho": col_tipo_despacho,
    }

    faltantes = [
        nombre for nombre, columna in columnas_obligatorias.items()
        if columna is None
    ]

    if faltantes:
        raise ValueError(
            f"En el archivo {ruta_archivo.name} faltan columnas obligatorias: {faltantes}"
        )

    columnas_horas = []

    for hora in range(24):
        columna_hora = detectar_columna_hora(columnas, hora)

        if columna_hora is None:
            raise ValueError(
                f"En el archivo {ruta_archivo.name} falta la columna horaria {hora}"
            )

        columnas_horas.append(columna_hora)

    df_limpio = pd.DataFrame()

    df_limpio["fecha"] = pd.to_datetime(df[col_fecha], errors="coerce")
    df_limpio["recurso"] = df[col_recurso].astype(str).str.strip()
    df_limpio["tipo_generacion"] = df[col_tipo_generacion].astype(str).str.strip().str.upper()
    df_limpio["combustible"] = df[col_combustible].astype(str).str.strip().str.upper()
    df_limpio["codigo_agente"] = df[col_codigo_agente].astype(str).str.strip().str.upper()
    df_limpio["tipo_despacho"] = df[col_tipo_despacho].astype(str).str.strip().str.upper()

    for hora, columna_hora in enumerate(columnas_horas):
        nombre_columna_hora = f"h{hora:02d}"

        df_limpio[nombre_columna_hora] = pd.to_numeric(
            df[columna_hora],
            errors="coerce"
        ).fillna(0)

    columnas_horas_limpias = [f"h{hora:02d}" for hora in range(24)]

    df_limpio["generacion_diaria_kwh"] = df_limpio[columnas_horas_limpias].sum(axis=1)
    df_limpio["generacion_diaria_gwh"] = df_limpio["generacion_diaria_kwh"] / 1_000_000

    df_limpio["anio"] = df_limpio["fecha"].dt.year
    df_limpio["mes"] = df_limpio["fecha"].dt.month
    df_limpio["dia"] = df_limpio["fecha"].dt.day
    df_limpio["fecha_mes"] = df_limpio["fecha"].dt.to_period("M").dt.to_timestamp()

    df_limpio["archivo_origen"] = ruta_archivo.name
    df_limpio["fila_encabezado_excel"] = indice_encabezado + 1

    filas_antes = len(df_limpio)

    df_limpio = df_limpio.dropna(subset=["fecha"])

    filas_despues = len(df_limpio)

    resumen = {
        "archivo": ruta_archivo.name,
        "fila_encabezado_excel": indice_encabezado + 1,
        "puntaje_encabezado": puntaje,
        "filas_originales": filas_antes,
        "filas_validas": filas_despues,
        "filas_eliminadas_sin_fecha": filas_antes - filas_despues,
        "anio_min": int(df_limpio["anio"].min()) if len(df_limpio) > 0 else None,
        "anio_max": int(df_limpio["anio"].max()) if len(df_limpio) > 0 else None,
        "recursos": int(df_limpio["recurso"].nunique()),
        "agentes": int(df_limpio["codigo_agente"].nunique()),
        "generacion_total_kwh": float(df_limpio["generacion_diaria_kwh"].sum()),
        "generacion_total_gwh": float(df_limpio["generacion_diaria_gwh"].sum()),
    }

    return df_limpio, resumen


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("================================================")
    print("ENERGYVIEW COLOMBIA - LIMPIEZA Y CONSOLIDACIÓN")
    print("================================================")

    archivos_excel = sorted(INPUT_DIR.glob("*.xlsx"))

    if not archivos_excel:
        mensaje = f"No se encontraron archivos Excel en {INPUT_DIR}"
        print(mensaje)
        REPORTE_LOG.write_text(mensaje, encoding="utf-8")
        return

    bases_limpias = []
    resumenes = []

    for archivo in archivos_excel:
        try:
            base_limpia, resumen = limpiar_archivo(archivo)
            bases_limpias.append(base_limpia)
            resumenes.append(resumen)

        except Exception as error:
            print(f"ERROR procesando {archivo.name}: {error}")

            resumenes.append({
                "archivo": archivo.name,
                "error": str(error)
            })

    if not bases_limpias:
        mensaje = "No se pudo consolidar ningún archivo."
        print(mensaje)
        REPORTE_LOG.write_text(mensaje, encoding="utf-8")
        return

    df_consolidado = pd.concat(bases_limpias, ignore_index=True)

    df_consolidado = df_consolidado.sort_values(
        by=["fecha", "tipo_generacion", "recurso"]
    ).reset_index(drop=True)

    # Guardar CSV
    df_consolidado.to_csv(
        SALIDA_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # Guardar Excel
    df_consolidado.to_excel(
        SALIDA_EXCEL,
        index=False
    )

    # Crear reporte de limpieza
    with open(REPORTE_LOG, "w", encoding="utf-8") as log:
        log.write("ENERGYVIEW COLOMBIA - REPORTE DE LIMPIEZA Y CONSOLIDACIÓN\n")
        log.write("=" * 70)
        log.write("\n\n")

        for resumen in resumenes:
            log.write(f"Archivo: {resumen.get('archivo')}\n")

            if "error" in resumen:
                log.write(f"ERROR: {resumen.get('error')}\n")
                log.write("-" * 70)
                log.write("\n\n")
                continue

            log.write(f"Fila encabezado Excel: {resumen.get('fila_encabezado_excel')}\n")
            log.write(f"Puntaje encabezado: {resumen.get('puntaje_encabezado')}\n")
            log.write(f"Filas originales: {resumen.get('filas_originales')}\n")
            log.write(f"Filas válidas: {resumen.get('filas_validas')}\n")
            log.write(f"Filas eliminadas sin fecha: {resumen.get('filas_eliminadas_sin_fecha')}\n")
            log.write(f"Año mínimo: {resumen.get('anio_min')}\n")
            log.write(f"Año máximo: {resumen.get('anio_max')}\n")
            log.write(f"Recursos únicos: {resumen.get('recursos')}\n")
            log.write(f"Agentes únicos: {resumen.get('agentes')}\n")
            log.write(f"Generación total kWh: {resumen.get('generacion_total_kwh'):.2f}\n")
            log.write(f"Generación total GWh: {resumen.get('generacion_total_gwh'):.4f}\n")
            log.write("-" * 70)
            log.write("\n\n")

        log.write("RESUMEN CONSOLIDADO\n")
        log.write("=" * 70)
        log.write("\n")
        log.write(f"Total filas consolidadas: {len(df_consolidado)}\n")
        log.write(f"Años consolidados: {sorted(df_consolidado['anio'].dropna().unique().tolist())}\n")
        log.write(f"Fecha mínima: {df_consolidado['fecha'].min()}\n")
        log.write(f"Fecha máxima: {df_consolidado['fecha'].max()}\n")
        log.write(f"Total recursos únicos: {df_consolidado['recurso'].nunique()}\n")
        log.write(f"Total agentes únicos: {df_consolidado['codigo_agente'].nunique()}\n")
        log.write(f"Tipos de generación: {sorted(df_consolidado['tipo_generacion'].dropna().unique().tolist())}\n")
        log.write(f"Combustibles: {sorted(df_consolidado['combustible'].dropna().unique().tolist())}\n")
        log.write(f"Generación total kWh: {df_consolidado['generacion_diaria_kwh'].sum():.2f}\n")
        log.write(f"Generación total GWh: {df_consolidado['generacion_diaria_gwh'].sum():.4f}\n")

    print("\nLimpieza y consolidación finalizada correctamente.")
    print(f"Base CSV generada en: {SALIDA_CSV}")
    print(f"Base Excel generada en: {SALIDA_EXCEL}")
    print(f"Reporte generado en: {REPORTE_LOG}")
    print(f"Total filas consolidadas: {len(df_consolidado)}")


if __name__ == "__main__":
    main()