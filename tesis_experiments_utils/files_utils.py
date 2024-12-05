import os
import pandas as pd

# Rutas para almacenar particiones de datos
path_hold_out = os.path.join("particiones", "hold_out")

# Nombre de los archivos de particiones
hold_out_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]

# Ruta para almacenar los datos imputados
path_imputed = "datos_imputados"

# Carpeta para almacenar las matrices de confusión
conf_matrices_path = "matrices_de_confusion"

# Carpeta para almacenar learning cuves
learning_curves_path = "curvas_de_aprendizaje"

# Carpeta para almacenar resultados
results_path = "resultados"

# Ruta para el archivo de resultados
results_file = os.path.join(results_path, "resultados.csv")

# Ruta para el archivo de predicciones
predictions_file = os.path.join(results_path, "predicciones.csv")

# Ruta para el archivo de curvas de aprendizaje
learning_curves_file = os.path.join(learning_curves_path, "curvas_de_aprendizaje.csv")

# Ruta para el archivo de tiempos de ejecución
execution_times_file = os.path.join(results_path, "execution_times.csv")

# Ruta para el archivo de tiempos de validación cruzada
cross_val_times_file = os.path.join(results_path, "cross_val_times.csv")

# Ruta para el archivo de tiempos de curvas de aprendizaje
learning_curves_times_file = os.path.join(
    learning_curves_path, "learning_curves_times.csv"
)


# Función para configurar las rutas de los archivos


def setup_folders():
    mkdirs()


# Función para crear las carpetas necesarias
def mkdirs():
    """Función para crear las carpetas necesarias"""
    folders = [
        path_hold_out,
        path_imputed,
        conf_matrices_path,
        learning_curves_path,
        results_path,
    ]
    for f in folders:
        try:
            os.makedirs(f)
            print(f"La carpeta '{f}' ha sido creada")
        except FileExistsError:
            print(f"La carpeta '{f}' ya existe")
            pass
        except Exception as e:
            print(f"Error al crear la carpeta '{f}': {e}")
            pass


def folder_exists(folder):
    """Función para verificar si una carpeta existe"""
    return os.path.exists(folder)


# ====================================================================================================
#               FUNCIONES PARA CREAR ARCHIVOS DE RESULTADOS Y CURVAS DE APRENDIZAJE
# ====================================================================================================


def create_files():
    """Función para crear los archivos de resultados y curvas de aprendizaje"""
    create_results_file()
    create_predictions_file()
    create_learning_curves_file()


def create_results_file(multiclass=False):
    """Función para crear el archivo de resultados"""
    if not os.path.exists(results_file):
        with open(results_file, "w") as file:
            if multiclass:
                file.write(
                    "Classifier,clase,accuracy,balanced_accuracy,sensitivity,specificity,f1,mcc\n"
                )
            else:
                file.write(
                    "Classifier,Accuracy,Balanced_accuracy,Recall,Specificity,AUC,MCC,Precision,F1-score\n"
                )
    print(f"Archivo de resultados creado en {results_file}")
    return results_file  # Devuelve la ruta del archivo de resultados


def create_predictions_file():
    """Función para crear el archivo de predicciones"""
    if not os.path.exists(predictions_file):
        with open(predictions_file, "w") as file:
            file.write("Classifier,y_true,y_pred\n")
    print(f"Archivo de predicciones creado en {predictions_file}")
    return predictions_file  # Devuelve la ruta del archivo de predicciones


def create_learning_curves_file():
    """Función para crear el archivo de curvas de aprendizaje"""
    if not os.path.exists(learning_curves_file):
        with open(learning_curves_file, "w") as file:
            file.write("Classifier,Train_sizes,Train_scores,Validation_scores\n")
    print(f"Archivo de curvas de aprendizaje creado en {learning_curves_file}")
    return learning_curves_file  # Devuelve la ruta del archivo de curvas de aprendizaje


def create_execution_times_file():
    """Función para crear el archivo de tiempos de ejecución"""
    if not os.path.exists(execution_times_file):
        with open(execution_times_file, "w") as file:
            file.write("Classifier,training_s,testing_s\n")
    print(f"Archivo de tiempos de ejecución creado en {execution_times_file}")
    return execution_times_file  # Devuelve la ruta del archivo de tiempos de ejecución


def create_cross_val_times_file():
    """Función para crear el archivo de tiempos de validación cruzada"""
    if not os.path.exists(cross_val_times_file):
        with open(cross_val_times_file, "w") as file:
            file.write("Classifier,time_s\n")
    print(f"Archivo de tiempos de validación cruzada creado en {cross_val_times_file}")
    return cross_val_times_file  # Devuelve la ruta del archivo de tiempos de validación cruzada


def create_learning_curves_times_file():
    """Función para crear el archivo de tiempos de curvas de aprendizaje"""
    if not os.path.exists(learning_curves_times_file):
        with open(learning_curves_times_file, "w") as file:
            file.write("Classifier,time_s\n")
    print(
        f"Archivo de tiempos de curvas de aprendizaje creado en {learning_curves_times_file}"
    )
    return learning_curves_times_file  # Devuelve la ruta del archivo de tiempos de curvas de aprendizaje


# ====================================================================================================
#    FUNCIONES PARA ESCRIBIR LOS RESULTADOS Y CURVAS DE APRENDIZAJE DE UN CLASIFICADOR EN ARCHIVOS
# ====================================================================================================


def write_results_to_file(results, results_file=results_file):
    """Función para escribir los resultados de un clasificador en el archivo de resultados"""
    if not os.path.exists(results_file):
        create_results_file()

    try:
        with open(results_file, "a") as file:
            file.write(
                f"{results['Classifier']},{results['Accuracy']},{results['Balanced_accuracy']},{results['Recall']},{results['Specificity']},{results['AUC']},{results['MCC']},{results['Precision']},{results['F1-score']}\n"
            )
        print(
            f"Resultados del clasificador {results['Classifier']} almacenados en {results_file}"
        )
    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")


def write_results_to_file_multiclass(results, results_file=results_file):
    """Función para escribir los resultados de un clasificador en el archivo de resultados para clasificación multiclase"""
    if not os.path.exists(results_file):
        create_results_file(multiclass=True)

    try:
        with open(results_file, "a") as file:
            file.write(
                f"{results['Classifier']},{results['clase']},{results['accuracy']},{results['balanced_accuracy']},{results['sensitivity']},{results['specificity']},{results['f1']},{results['mcc']}\n"
            )
        print(
            f"Resultados del clasificador {results['Classifier']} almacenados en {results_file}"
        )
    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")


def write_predictions_to_file(predictions, predictions_file=predictions_file):
    """Función para escribir las predicciones de un clasificador en el archivo de predicciones"""

    if not os.path.exists(predictions_file):
        create_predictions_file()

    try:
        #        with open(predictions_file, "a") as file:
        #            file.write(
        #                f"{predictions['Classifier']},{predictions['y_true']},{predictions['y_pred']}\n"
        #            )
        #        print(
        #            f"Predicciones del clasificador {predictions['Classifier']} almacenadas en {predictions_file}"
        #        )
        df = pd.DataFrame(predictions)

        df.to_csv(predictions_file, mode="a", header=False, index=False)
        print(
            f"Predicciones del clasificador {predictions['Classifier']} almacenadas en {predictions_file}."
        )

    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")


def write_learning_curves_to_file(
    learning_curve, learning_curves_file=learning_curves_file
):
    """Función para escribir las curvas de aprendizaje de un clasificador en el archivo de curvas de aprendizaje"""

    if not os.path.exists(learning_curves_file):
        create_learning_curves_file()

    try:
        #        with open(learning_curves_file, "a") as file:
        #            file.write(
        #                f"{learning_curve['Classifier']},{learning_curve['Train_sizes']},{learning_curve['Train_scores']},{learning_curve['Validation_scores']}\n"
        #            )
        #        print(
        #            f"Curvas de aprendizaje del clasificador {learning_curve['Classifier']} almacenadas en {learning_curves_file}"
        #        )
        learning_curve["Train_sizes"] = learning_curve["Train_sizes"].tolist()
        learning_curve["Train_scores"] = learning_curve["Train_scores"].tolist()
        learning_curve["Validation_scores"] = learning_curve[
            "Validation_scores"
        ].tolist()
        df = pd.DataFrame(learning_curve)
        df.to_csv(learning_curves_file, mode="a", header=False, index=False)
        print(
            f"Curvas de aprendizaje del clasificador {learning_curve['Classifier']} almacenadas en {learning_curves_file}."
        )

    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")


def write_execution_times_to_file(
    execution_time, execution_times_file=execution_times_file
):
    """Función para escribir los tiempos de ejecución de un clasificador en el archivo de tiempos de ejecución"""

    if not os.path.exists(execution_times_file):
        create_execution_times_file()

    try:
        with open(execution_times_file, "a") as file:
            file.write(
                f"{execution_time['Classifier']},{execution_time['training_s']},{execution_time['testing_s']}\n"
            )
        print(
            f"Tiempo de ejecución del clasificador {execution_time['Classifier']} almacenado en {execution_times_file}"
        )
    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")


def write_cross_val_times_to_file(
    cross_val_time, cross_val_times_file=cross_val_times_file
):
    """Función para escribir los tiempos de validación cruzada de un clasificador en el archivo de tiempos de validación cruzada"""

    if not os.path.exists(cross_val_times_file):
        create_cross_val_times_file()

    try:
        with open(cross_val_times_file, "a") as file:
            file.write(f"{cross_val_time['Classifier']},{cross_val_time['time_s']}\n")
        print(
            f"Tiempo de validación cruzada del clasificador {cross_val_time['Classifier']} almacenado en {cross_val_times_file}"
        )
    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")


def write_learning_curves_times_to_file(
    learning_curves_time, learning_curves_times_file=learning_curves_times_file
):
    """Función para escribir los tiempos de curvas de aprendizaje de un clasificador en el archivo de tiempos de curvas de aprendizaje"""

    if not os.path.exists(learning_curves_times_file):
        create_learning_curves_times_file()

    try:
        with open(learning_curves_times_file, "a") as file:
            file.write(
                f"{learning_curves_time['Classifier']},{learning_curves_time['time_s']}\n"
            )
        print(
            f"Tiempo de curvas de aprendizaje del clasificador {learning_curves_time['Classifier']} almacenado en {learning_curves_times_file}"
        )
    except Exception as e:
        print(f"Error al escribir en el archivo: {e}")
