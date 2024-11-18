# == Archivo que contiene las funciones requeridas para el manejo y preprocesamiento de datos ==

# Jose Daniel Perez Ramirez - CIC IPN
# Maestría en Ciencias en Computación
# Trabajo de Tesis
# Julio 2024

import os
import pandas as pd
import numpy as np
from tabulate import tabulate

from tesis_experiments_utils.files_utils import (
    path_hold_out,
    hold_out_files,
    path_imputed,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ====================================================================================================
#               FUNCIÓN PARA CARGAR UN DATASET DESDE UN ARCHIVO CSV
# ====================================================================================================


def load_dataset(url, cat_atts=[]):
    """Carga un dataset desde un archivo CSV y lo devuelve como un DataFrame de Pandas
    url: str, URL del archivo CSV
    cat_atts: list, lista de atributos categóricos
    return: pd.DataFrame

    Los atributos categóricos (en la lista cat_atts) se convierten a tipo 'category' y los numéricos a 'float64'
    """
    df = pd.read_csv(url)
    num_atts = [c for c in df.columns if c not in cat_atts and c != "target"]
    df[cat_atts] = df[cat_atts].astype("category")
    df[num_atts] = df[num_atts].astype("float64")
    return df


# ====================================================================================================
#               FUNCIONES PARA LA PARTICIÓN DE DATOS UTILIZANDO PARTICION FIJA
# ====================================================================================================


# Genera particiones de datos hold-out si no existen, o las lee si ya se han generado
def fixed_partitioning(
    path_to_file,
    cat_atts=[],
    test_size=0.20,
    hold_out_dir=path_hold_out,
    overwrite=False,
):
    """Genera particiones de datos hold-out si no existen, o las lee si ya se han generado
    Internamente utiliza train_test_split de sklearn
    path_to_file: str, URL del archivo CSV,
    cat_atts: list, lista de atributos categóricos. Por defecto vacía
    test_size: float, tamaño del conjunto de prueba. Por defecto 0.20
    hold_out_dir: str, directorio donde se almacenarán o leerán las particiones. Por defecto 'particiones/hold_out'
    return: X_train, X_test, y_train, y_test"""

    training_size = 1 - test_size

    if path_to_file is None:
        raise ValueError("Debe proporcionar la ruta del archivo CSV")

    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            "El tamaño del conjunto de prueba debe ser un valor entre 0 y 1"
        )

    if not hold_out_files_exist(test_size) or overwrite:
        return split_and_save_hold_out(path_to_file, cat_atts, test_size, hold_out_dir)
    else:
        print(
            f"Partición de datos hold out {training_size:.2f}-{test_size:.2f} encontrada en '{hold_out_dir}'"
        )
        return read_hold_out(hold_out_dir, cat_atts, test_size)


# ====================================================================================================
#               FUNCIONES PARA LA PARTICIÓN DE DATOS UTILIZANDO HOLD-OUT
# ====================================================================================================


test_size = 0.20


def calculate_imbalance_ratio(y, classes):
    card_classes = [sum(1 for l in y if l == c[0]) for c in classes]

    IR = max(card_classes) / min(card_classes)

    print("Cardinalidades de las clases")
    print(
        tabulate(
            [[c[1], card_classes[i]] for i, c in enumerate(classes)],
            headers=["Clase", "Cardinalidad"],
        )
    )

    if IR <= 1.5:
        print("Dataset balanceado con IR = %.4f" % IR)
    else:
        print("Dataset desbalanceado con IR = %.4f" % IR)


# Verifica si los archivos de partición de datos existen en el directorio especificado
def hold_out_files_exist(test_size):
    """Verifica si se encuentran los archivos de partición de datos en el directorio especificado.
    Si un archivo no se encuentra, se devuelve False
    return: bool"""
    training_size = 1 - test_size
    files_path = os.path.join(path_hold_out, f"{training_size:.2f}_{test_size:.2f}")
    for file in hold_out_files:
        if not os.path.isfile(os.path.join(files_path, file)):
            print(f"Archivo '{file}' no encontrado en '{files_path}'")
            return False
    return True


# Particiona el dataset y almacena las particiones en archivos CSV
def split_and_save_hold_out(
    path_to_file, cat_atts=[], test_size=test_size, hold_out_dir=path_hold_out
):
    """Particiona el dataset y almacena las particiones en archivos CSV'
    path_to_file: str, URL del archivo CSV,
    cat_atts: list, lista de atributos categóricos. Por defecto vacía
    test_size: float, tamaño del conjunto de prueba. Por defecto 0.20
    hold_out_dir: str, directorio donde se almacenarán las particiones. Por defecto 'particiones/hold_out'
    return: X_train, X_test, y_train, y_test
    """

    training_size = 1 - test_size

    hold_out_dir = os.path.join(path_hold_out, f"{training_size:.2f}_{test_size:.2f}")

    # Valida si la ruta para almacenar las particiones existe, si no la crea
    os.makedirs(hold_out_dir, exist_ok=True)

    print(
        f"Generando una nueva partición de datos hold out {training_size:.2f}-{test_size:.2f}. Almacenando en: '{hold_out_dir}'"
    )

    # Verifica si el archivo CSV existe
    if not os.path.isfile(path_to_file):
        raise FileNotFoundError(f"El archivo {path_to_file} no se encontró")

    # Lectura del dataset
    df = load_dataset(path_to_file)

    # Partición de atributos y etiquetas de clase
    X = df.drop(df.columns[-1], axis=1)
    y = df[df.columns[-1]]

    # Atributos numéricos
    num_atts = [c for c in X.columns if c not in cat_atts and c != "target"]

    # Asignación de tipos de atributo
    X[cat_atts] = X[cat_atts].astype("category")
    X[num_atts] = X[num_atts].astype("float64")

    # Partición de datos utilizando train_test_split de sklearn
    # Hold out estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=94
    )

    # Almacenamiento de particiones
    X_train.to_csv(os.path.join(hold_out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(hold_out_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(hold_out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(hold_out_dir, "y_test.csv"), index=False)

    print(
        f"\nSe ha generado y almacenado una nueva partición de datos en '{hold_out_dir}'"
    )

    return X_train, X_test, y_train, y_test


# Lee las particiones de datos desde los archivos CSV
def read_hold_out(hold_out_dir=path_hold_out, cat_atts=[], test_size=test_size):
    """Lee las particiones de datos desde los archivos CSV. Los archivos deben estar nombrados como 'X_train.csv', 'X_test.csv', 'y_train.csv' y 'y_test.csv'
    hold_out_dir: str, directorio donde se encuentran los archivos CSV
    cat_atts: list, lista de atributos categóricos. Por defecto vacía
    return: X_train, X_test, y_train, y_test"""

    training_size = 1 - test_size
    files_path = os.path.join(hold_out_dir, f"{training_size:.2f}_{test_size:.2f}")

    X_train = pd.read_csv(os.path.join(files_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(files_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(files_path, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(files_path, "y_test.csv"))

    # Atributos numéricos
    num_atts = [c for c in X_train.columns if c not in cat_atts and c != "target"]

    # Asignación de tipos de atributo
    X_train[cat_atts] = X_train[cat_atts].astype("category")
    X_train[num_atts] = X_train[num_atts].astype("float64")
    X_test[cat_atts] = X_test[cat_atts].astype("category")
    X_test[num_atts] = X_test[num_atts].astype("float64")

    print(f"Partición de datos cargada exitosamente desde el directorio '{files_path}'")

    return X_train, X_test, y_train.to_numpy().ravel(), y_test.to_numpy().ravel()


# Genera particiones de datos hold-out si no existen, o las lee si ya se han generado
def stratified_hold_out(
    path_to_file,
    cat_atts=[],
    test_size=0.20,
    hold_out_dir=path_hold_out,
    overwrite=False,
):
    """Genera particiones de datos hold-out si no existen, o las lee si ya se han generado
    Internamente utiliza train_test_split de sklearn
    path_to_file: str, URL del archivo CSV,
    cat_atts: list, lista de atributos categóricos. Por defecto vacía
    test_size: float, tamaño del conjunto de prueba. Por defecto 0.20
    hold_out_dir: str, directorio donde se almacenarán o leerán las particiones. Por defecto 'particiones/hold_out'
    return: X_train, X_test, y_train, y_test"""

    training_size = 1 - test_size

    if path_to_file is None:
        raise ValueError("Debe proporcionar la ruta del archivo CSV")

    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            "El tamaño del conjunto de prueba debe ser un valor entre 0 y 1"
        )

    if not hold_out_files_exist(test_size) or overwrite:
        return split_and_save_hold_out(path_to_file, cat_atts, test_size, hold_out_dir)
    else:
        print(
            f"Partición de datos hold out {training_size:.2f}-{test_size:.2f} encontrada en '{hold_out_dir}'"
        )
        return read_hold_out(hold_out_dir, cat_atts, test_size)


# ====================================================================================================
#               FUNCIONES PARA LA IMPUTACIÓN DE DATOS PERDIDOS O FALTANTES
# ====================================================================================================


# Imputa valores faltantes en los conjuntos de datos de entrenamiento y prueba
def impute_missing_values(X_train, X_test, cat_atts, strategy, save_path):
    """Imputa valores faltantes en los conjuntos de datos de entrenamiento y prueba.
    La estrategia de imputación se calcula sobre el conjunto de entrenamiento y se aplica a ambos conjuntos.
    Internamente, se utiliza SimpleImputer de sklearn
    X_train: pd.DataFrame, conjunto de datos de entrenamiento
    X_test: pd.DataFrame, conjunto de datos de prueba
    cat_atts: list, lista de atributos categóricos. Por defecto vacía
    strategy: str, estrategia de imputación. Puede ser 'mean', 'median', 'most_frequent' o 'constant'. Por defecto 'mean'
    save_path: str, directorio donde se almacenarán los datos imputados. Por defecto 'imputed_data'
    return: X_train, X_test"""

    print(
        f"Imputando valores faltantes en los conjuntos de datos utilizando la estrategia '{strategy}'"
    )

    # Instancia de SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy=strategy)

    # Atributos numéricos
    num_atts = [c for c in X_train.columns if c not in cat_atts and c != "target"]

    # Entrenamiento del imputador y transformación de los datos de entrenamiento
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)

    # Imputación en el conjunto de prueba
    X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

    # Asignación de tipo de atributo categorico
    X_train[cat_atts] = X_train[cat_atts].astype("category")
    X_test[cat_atts] = X_test[cat_atts].astype("category")

    # Asignación de tipo de atributo numérico
    X_train[num_atts] = X_train[num_atts].astype("float64")
    X_test[num_atts] = X_test[num_atts].astype("float64")

    # Almacenamiento de dataset con valores imputados
    os.makedirs(save_path, exist_ok=True)
    X_train.to_csv(
        os.path.join(save_path, f"X_train_imputed_{strategy}.csv"), index=False
    )
    X_test.to_csv(
        os.path.join(save_path, f"X_test_imputed_{strategy}.csv"), index=False
    )

    print(f"\nSe han imputado y almacenado los conjuntos de datos en '{save_path}'")

    return X_train, X_test


# Lee los conjuntos de datos imputados desde los archivos CSV
def read_imputed_data(data_path=path_imputed, strategy="mean"):
    """Lee los conjuntos de datos imputados desde los archivos CSV
    data_path: str, directorio donde se encuentran los archivos CSV
    strategy: str, estrategia de imputación. Por defecto 'mean'
    return: X_train, X_test"""

    X_train = pd.read_csv(os.path.join(data_path, f"X_train_imputed_{strategy}.csv"))
    X_test = pd.read_csv(os.path.join(data_path, f"X_test_imputed_{strategy}.csv"))

    print(f"Datos imputados cargados exitosamente desde '{data_path}'")

    return X_train, X_test


# Verifica si los archivos de datos imputados existen en el directorio especificado
def imputed_files_exist(strategy, data_path):
    """Verifica si se encuentran los archivos de datos imputados en el directorio especificado.
    Si un archivo no se encuentra, se devuelve False
    return: bool"""
    imputed_files = [
        f"X_train_imputed_{strategy}.csv",
        f"X_test_imputed_{strategy}.csv",
    ]
    for file in imputed_files:
        if not os.path.isfile(os.path.join(data_path, file)):
            print(f"Archivo '{file}' no encontrado en '{data_path}'")
            return False
    return True


# Genera conjuntos de datos imputados si no existen, o los lee si ya se han generado
def impute_data(
    X_train=None,
    X_test=None,
    cat_atts=[],
    strategy="mean",
    data_path=path_imputed,
    overwrite=False,
):
    """Genera conjuntos de datos imputados si no existen, o los lee si ya se han generado
    Internamente, se utiliza SimpleImputer de sklearn
    path_to_file: str, URL del archivo CSV,
    cat_atts: list, lista de atributos categóricos. Por defecto vacía
    test_size: float, tamaño del conjunto de prueba. Por defecto 0.20
    hold_out_dir: str, directorio donde se almacenarán o leerán las particiones. Por defecto 'datos_imputados'
    return: X_train, X_test, y_train, y_test"""
    if data_path is None:
        raise ValueError("Debe proporcionar la ruta de los datos imputados")

    if X_train is None or X_test is None:
        print("No se han proporcionado alguno de los conjuntos de datos.")
        print("Se intentará leer los datos imputados desde el directorio especificado")
        if not imputed_files_exist(strategy, data_path):
            raise FileNotFoundError(
                f"Es necesario proporcionar los conjuntos de datos de entrenamiento y prueba para generar los datos imputados."
            )
        else:
            return read_imputed_data(data_path, strategy)

    if (not imputed_files_exist(strategy, data_path)) or overwrite:
        return impute_missing_values(X_train, X_test, cat_atts, strategy, data_path)
    else:
        print(
            f"Datos imputados con estrategia '{strategy}' encontrados en '{data_path}'"
        )
        return read_imputed_data(data_path, strategy)
