import pandas as pd
import os


def cargar_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Carga un dataset (CSV o Excel) desde una ruta específica.
    Parámetros:
        dataset_path (str): ruta completa o relativa al archivo de datos.
    Retorna:
        pd.DataFrame: datos cargados en un DataFrame de pandas.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"El archivo no existe: {dataset_path}")

    formato = os.path.splitext(dataset_path)[1].lower()

    if formato == ".csv":
        df = pd.read_csv(dataset_path)
    elif formato in [".xls", ".xlsx"]:
        df = pd.read_excel(dataset_path)
    else:
        raise ValueError(f"Formato de dataset no soportado: {formato}")

    return df
