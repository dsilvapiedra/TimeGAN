import os
import scipy.io
import pandas as pd
import datetime as dt

# Desfase entre épocas MATLAB y Unix
matlab_to_epoch_days = 719529


def matlab_enum_to_datetime(timestamps):
    """
    Convierte valores datenum de MATLAB a pandas datetime.

    Parámetros:
        timestamps (array): Array con valores datenum de MATLAB.

    Retorna:
        pd.Series: Fechas convertidas a pandas datetime.
    """
    return pd.to_datetime(timestamps - matlab_to_epoch_days, unit="D").round(freq="min")


def mat_to_csv(input_folder, output_csv, start_date, end_date, station_code):
    """
    Convierte y concatena archivos .mat en un archivo CSV basado en un rango de fechas.

    Parámetros:
        input_folder (str): Carpeta donde están los archivos .mat.
        output_csv (str): Ruta del archivo CSV de salida.
        start_date (str): Fecha inicial en formato 'YYYYMMDD'.
        end_date (str): Fecha final en formato 'YYYYMMDD'.
        station_code (str): Código de la estación (parte del nombre del archivo).
    """
    # Convertir fechas a objetos datetime
    start_date = dt.datetime.strptime(start_date, '%Y%m%d')
    end_date = dt.datetime.strptime(end_date, '%Y%m%d')

    # Lista para almacenar los DataFrames
    data_frames = []

    # Recorrer el rango de fechas
    current_date = start_date
    while current_date <= end_date:
        # Construir el nombre del archivo .mat
        file_name = f"{current_date.strftime('%Y%m%d')}{station_code}.mat"
        file_path = os.path.join(input_folder, file_name)

        if os.path.exists(file_path):
            print(f"Procesando: {file_name}")
            # Cargar archivo .mat
            mat_data = scipy.io.loadmat(file_path)

            # Extraer datos
            if "fecha" in mat_data and station_code in mat_data:
                # Convertir "fecha" de datenum a datetime
                fechas = matlab_enum_to_datetime(mat_data["fecha"].flatten())

                # Extraer los valores de la estación
                valores = mat_data[station_code].flatten()

                # Crear un DataFrame
                df = pd.DataFrame({
                    "datetime": fechas,
                    "level": valores
                })
                data_frames.append(df)
            else:
                print(f"Variables 'fecha' o '{station_code}' no encontradas en {file_name}")
        else:
            print(f"Archivo no encontrado: {file_name}")

        # Avanzar al siguiente día
        current_date += dt.timedelta(days=1)

    # Concatenar todos los DataFrames
    if data_frames:
        full_data = pd.concat(data_frames, ignore_index=True)
        # Exportar a CSV
        full_data.to_csv(output_csv, index=False)
        print(f"Archivo CSV creado: {output_csv}")
    else:
        print("No se encontraron datos para las fechas proporcionadas.")

def concatenar_csv(codigo_estacion, fecha_inicial, fecha_final, directorio):
    # Convertir las fechas de entrada a objetos datetime
    fecha_inicial = dt.datetime.strptime(fecha_inicial + " 00:00:00", "%Y%m%d %H:%M:%S")
    fecha_final = dt.datetime.strptime(fecha_final + " 23:59:59", "%Y%m%d %H:%M:%S")

    # Lista para almacenar los DataFrames
    dfs = []

    # Recorrer todos los archivos en el directorio
    for archivo in os.listdir(directorio):
        if archivo.endswith(".csv"):
            # Extraer el código de la estación y las fechas del nombre del archivo
            try:
                partes = archivo.split("_")
                codigo_archivo = partes[0]
                fecha_inicio_archivo = dt.datetime.strptime(partes[1], "%d-%m-%Y")
                fecha_fin_archivo = dt.datetime.strptime(partes[2].replace(".csv", ""), "%d-%m-%Y")
            except (IndexError, ValueError):
                # Si el archivo no tiene el formato correcto, se ignora
                continue

            # Verificar si el archivo corresponde al código de la estación y está dentro del rango de fechas
            if (codigo_archivo == codigo_estacion and
                fecha_inicio_archivo <= fecha_final and
                fecha_fin_archivo >= fecha_inicial):

                # Leer el archivo CSV
                ruta_archivo = os.path.join(directorio, archivo)
                df = pd.read_csv(ruta_archivo, parse_dates=['Fecha/Hora'])

                # Filtrar por el rango de fechas
                df_filtrado = df[(df['Fecha/Hora'] >= fecha_inicial) & (df['Fecha/Hora'] <= fecha_final)]

                # Agregar el DataFrame filtrado a la lista
                dfs.append(df_filtrado)

    # Concatenar todos los DataFrames en uno solo
    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        # Guardar el DataFrame final en un nuevo archivo CSV
        df_final.to_csv(f'{codigo_estacion}_concatenado.csv', index=False)
        print(f"Archivo concatenado guardado como {codigo_estacion}_concatenado.csv")
    else:
        print("No se encontraron archivos que coincidan con los criterios.")

# Ejemplo de uso
codigo_estacion = "Muelle Fluvial"  # Código de la estación
fecha_inicial = '20200301'
fecha_final = '20210301'
directorio = '../../../ANP/download/'

concatenar_csv(codigo_estacion, fecha_inicial, fecha_final, directorio)

# # Ejemplo de uso para .mats
# in_dir = '../GANS/data/ANP/data/'
# start_dt = '20200301'
# end_dt = '20210301'
# st_code = 'MVD'
# out_csv = f"./{start_dt}_{end_dt}_{st_code}dataset.csv"

# mat_to_csv(in_dir, out_csv, start_dt, end_dt, st_code)
