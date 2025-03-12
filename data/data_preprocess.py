"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com


-----------------------------

(1) data_preprocess: Load the data and preprocess into a 3d numpy array
(2) imputater: Impute missing data 
"""
# Local packages
import os
from typing import Union, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def indexar_datos(ori_data, index, tipo_indexacion, n=1):
    """
    Indexa los datos según el tipo de indexación especificado.
    Además, permite dividir cada día en n partes y dropea filas de días con solo NaNs.

    Args:
        ori_data (pd.DataFrame): DataFrame con la columna 'datetime'.
        index (str): Nombre de la columna de índice.
        tipo_indexacion (str): 'dia', 'semana' o 'mes'.
        n (int): Número de subdivisiones por día (default=1, indexación por día completo).

    Returns:
        pd.DataFrame: DataFrame con la columna indexada y sin días completos con NaNs.
    """

    # Convertir a datetime si no lo es
    ori_data['datetime'] = pd.to_datetime(ori_data['datetime'])

    # Eliminar días en los que todas las columnas son NaN
    ori_data['dia'] = ori_data['datetime'].dt.strftime('%Y%m%d')  # Crear una columna auxiliar con el día
    dias_validos = ori_data.groupby('dia').apply(lambda x: not x.drop(columns=['datetime', 'dia']).isna().all().all())
    ori_data = ori_data[ori_data['dia'].isin(dias_validos[dias_validos].index)]  # Filtrar días válidos
    ori_data = ori_data.drop(columns=['dia'])  # Eliminar la columna auxiliar

    # Generar índices según el tipo de indexación
    if tipo_indexacion == 'dia':
        ori_data[index] = ori_data['datetime'].dt.strftime('%Y%m%d').astype(int)
    elif tipo_indexacion == 'semana':
        ori_data[index] = ori_data['datetime'].dt.strftime('%Y%U').astype(int)  # %U: Semana del año (domingo como primer día)
    elif tipo_indexacion == 'mes':
        ori_data[index] = ori_data['datetime'].dt.strftime('%Y%m').astype(int)
    else:
        raise ValueError("Tipo de indexación no válido. Debe ser 'dia', 'semana' o 'mes'.")

    # Dividir cada día en `n` partes (solo si tipo_indexacion es 'dia')
    if tipo_indexacion == 'dia' and n > 1:
        segmento = (ori_data['datetime'].dt.hour // (24 // n)).astype(str)
        ori_data[index] = ori_data[index].astype(str) + segmento  # Convertir ambos a str antes de concatenar

    return ori_data.drop(["datetime"], axis=1)
    
# Function to trim the DataFrame between two dates
def trim_dataframe(df, start_date, end_date):
    """Trims a DataFrame to include only rows within a specified date range.

    Args:
        df: The input DataFrame with a 'datetime' column.
        start_date: The start date of the range (inclusive).
        end_date: The end date of the range (inclusive).

    Returns:
        A new DataFrame containing only the rows within the specified date range.
        Returns the original DataFrame if invalid dates are provided.
    """
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    except ValueError:
        print("Invalid date format. Please use a valid date format.")
        return df  # Return original DataFrame if dates are invalid
    # Display min and max of the 'datetime' column
    print("Min Datetime:", df['datetime'].min())
    print("Max Datetime:", df['datetime'].max())
    trimmed_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    return trimmed_df


def data_preprocess(
    file_name: str, 
    max_seq_len: int, 
    padding_value: float=-1.0,
    impute_method: str="mode", 
    scaling_method: str="minmax", 
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Load the data and preprocess into 3d numpy array.
    Preprocessing includes:
    1. Remove outliers
    2. Extract sequence length for each patient id
    3. Impute missing data 
    4. Normalize data
    6. Sort dataset according to sequence length

    Args:
    - file_name (str): CSV file name
    - max_seq_len (int): maximum sequence length
    - impute_method (str): The imputation method ("median" or "mode") 
    - scaling_method (str): The scaler method ("standard" or "minmax")

    Returns:
    - processed_data: preprocessed data
    - time: ndarray of ints indicating the length for each data
    - params: the parameters to rescale the data 
    """

    #########################
    # Load data
    #########################
    index = 'Idx'
    # Data period:
    start_date = '2020-03-01'
    end_date = '2024-03-01'
    
    # Load csv
    print("Loading data...\n")
    ori_data = pd.read_csv(file_name)

    
    # Convertir la columna 'datetime' a tipo datetime
    ori_data["datetime"] = pd.to_datetime(ori_data["datetime"], errors='coerce')

    # Recortar al periodo deseado
    ori_data = trim_dataframe(ori_data, start_date, end_date)

    
    
    # Verificar si hay valores no convertidos (NaT)
    if ori_data["datetime"].isna().any():
        print("Advertencia: Algunas filas tienen valores inválidos en 'datetime'.")

    # Creo indice y dropeo timestamps
    # Indexar por día (n veces) y Eliminar días que constan solo de NaNs 
    ori_data = indexar_datos(ori_data, index, 'dia', n=4)
    print("Indexado por día:\n", ori_data)

    # Drop Colonia por falta de datos
    ori_data = ori_data.drop(["Col"], axis=1, errors='ignore')
    print(ori_data.shape)

    # Remove spurious column, so that column 0 is now 'admissionid'.
    if ori_data.columns[0] == "Unnamed: 0":  
        ori_data = ori_data.drop(["Unnamed: 0"], axis=1)

    # Index to 0 column
    first_column = ori_data.pop(index) 
    ori_data.insert(0, index, first_column) 
    #########################
    # Remove outliers from dataset
    #########################
    
    no = ori_data.shape[0]

    # # Seleccionar solo las columnas numéricas para calcular Z-scores
    # numeric_data = ori_data.select_dtypes(include=[np.number])

    # # Calcular Z-scores solo para las columnas numéricas
    # z_scores = stats.zscore(numeric_data, axis=0, nan_policy='omit')

    # # Filtrar filas donde el valor absoluto máximo de Z-score sea menor que 3
    # z_filter = np.nanmax(np.abs(z_scores), axis=1) < 3

    # # Capturar las filas eliminadas (outliers)
    # outliers = ori_data[~z_filter]  # Usamos ~ para invertir el filtro
    
    # # Aplicar el filtro al df original
    # ori_data = ori_data[z_filter]
    # print(f"Dropped {no - ori_data.shape[0]} rows (outliers)\n")
    

    # # Mostrar los valores eliminados
    # if not outliers.empty:
    #     print("Valores eliminados (outliers):")
    #     print(outliers)
    # else:
    #     print("No se eliminaron valores (no se encontraron outliers).")

    # Parameters
    uniq_id = np.unique(ori_data[index])
    no = len(uniq_id)
    dim = len(ori_data.columns) - 1

    #########################
    # Impute, scale and pad data
    #########################
    
    # Initialize scaler
    if scaling_method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(ori_data)
        params = [scaler.data_min_, scaler.data_max_]
    
    elif scaling_method == "standard":
        scaler = StandardScaler()
        scaler.fit(ori_data)
        params = [scaler.mean_, scaler.var_]

    # Imputation values
    if impute_method == "median":
        impute_vals = ori_data.median()
    elif impute_method == "mode":
        impute_arr = stats.mode(ori_data,axis=0, nan_policy='omit')
        impute_vals = impute_arr.mode[1]
    else:
        raise ValueError("Imputation method should be `median` or `mode`")    

    # TODO: Sanity check for padding value
    # if np.any(ori_data == padding_value):
    #     print(f"Padding value `{padding_value}` found in data")
    #     padding_value = np.nanmin(ori_data.to_numpy()) - 1
    #     print(f"Changed padding value to: {padding_value}\n")
    
    # Output initialization
    #padding_value = impute_vals
    output = np.empty([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
    output.fill(padding_value)
    time = []

    # For each uniq id
    for i in tqdm(range(no)):
        # Extract the time-series data with a certain admissionid

        curr_data = ori_data[ori_data[index] == uniq_id[i]].to_numpy()

        # Impute missing data
        curr_data = imputer(curr_data, impute_vals)

        # Normalize data
        curr_data = scaler.transform(curr_data)
        
        # Extract time and assign to the preprocessed data (Excluding ID)
        curr_no = len(curr_data)

        # Pad data to `max_seq_len`
        if curr_no >= max_seq_len:
            output[i, :, :] = curr_data[:max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
            time.append(max_seq_len)
        else:
            output[i, :curr_no, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]
            time.append(curr_no)
    return output, time, params, max_seq_len, padding_value


def imputer(
    curr_data: np.ndarray, 
    impute_vals: List, 
    zero_fill: bool = True
) -> np.ndarray:
    """Improved imputer for time-series water level data."""
    
    curr_data = pd.DataFrame(data=curr_data)
    
    # Imputation with provided values for each column
    impute_vals = pd.Series(impute_vals)
    imputed_data = curr_data.fillna(impute_vals)

    # Interpolation for missing values
    imputed_data = imputed_data.interpolate(method='linear', limit_direction='both')

    # Zero-fill only if explicitly required
    if zero_fill:
        imputed_data = imputed_data.fillna(0.0)

    # Check for any remaining NaN values
    if imputed_data.isnull().any().any():
        
        imputed_data = imputed_data.interpolate(method='pad', limit_direction='forward')
        if imputed_data.isnull().any().any():
            imputed_data = imputed_data.interpolate(method='pad', limit_direction='forward')
            if imputed_data.isnull().any().any():
                raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()

