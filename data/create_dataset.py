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


# Ejemplo de uso
in_dir = '../GANS/data/ANP/data/'
start_dt = '20200301'
end_dt = '20210301'
st_code = 'MVD'
out_csv = f"./{start_dt}_{end_dt}_{st_code}dataset.csv"

mat_to_csv(in_dir, out_csv, start_dt, end_dt, st_code)
