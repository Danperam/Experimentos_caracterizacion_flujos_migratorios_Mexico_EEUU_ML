# == Archivo que contiene las funciones requeridas para l ==

# Jose Daniel Perez Ramirez - CIC IPN
# Maestría en Ciencias en Computación
# Trabajo de Tesis
# Julio 2024

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scienceplots
from tesis_experiments_utils.confusion_matrices_utils import plot_save_conf_matrix
from tesis_experiments_utils.files_utils import (
    results_path,
    write_results_to_file,
    write_results_to_file_multiclass,
    write_predictions_to_file,
)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_auc_score,
)


# Diccionario para almacenar los resultados de los clasificadores
resultados = {
    "Classifier": [],
    "Accuracy": [],
    "Balanced_accuracy": [],
    "Recall": [],
    "Specificity": [],
    "AUC": [],
    "MCC": [],
    "Precision": [],
    "F1-score": [],
}

# Diccionario para almacenar las predicciones de los clasificadores
predicciones = {"Classifier": [], "y_true": [], "y_pred": []}

# Diccionario para almacenar las curvas de aprendizaje
learning_curves = {
    "Classifier": [],
    "Train_sizes": [],
    "Train_scores": [],
    "Validation_scores": [],
}


# ====================================================================================================
#               FUNCIONES PARA ENTRENAR Y EVALUAR LOS MODELOS DE CLASIFICACIÓN
# ====================================================================================================


def train_and_evaluate_model(
    modelo, X_train, y_train, X_test, y_test, nombre, pos_class, neg_class
):

    y_true, y_pred = list(), list()

    # Entrenar modelo
    modelo.fit(X_train, y_train)

    # Evaluar modelo
    y_eval = modelo.predict(X_test)

    # store
    y_true.extend(y_test)
    y_pred.extend(y_eval)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=pos_class[0])
    spec = tn / (tn + fp)
    prec = precision_score(y_true, y_pred, pos_label=pos_class[0])
    f1 = f1_score(y_true, y_pred, pos_label=pos_class[0])
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    # resultados["Classifier"].append(nombre)
    # resultados["Accuracy"].append(acc)
    # resultados["Balanced_accuracy"].append(b_acc)
    # resultados["Recall"].append(rec)
    # resultados["Specificity"].append(spec)
    # resultados["AUC"].append(auc)
    # resultados["MCC"].append(mcc)
    # resultados["Precision"].append(prec)
    # resultados["F1-score"].append(f1)

    write_results_to_file(
        {
            "Classifier": nombre,
            "Accuracy": acc,
            "Balanced_accuracy": b_acc,
            "Recall": rec,
            "Specificity": spec,
            "AUC": auc,
            "MCC": mcc,
            "Precision": prec,
            "F1-score": f1,
        }
    )

    write_predictions_to_file(
        {"Classifier": nombre, "y_true": y_true, "y_pred": y_pred}
    )

    # predicciones["Classifier"].append(nombre)
    # predicciones["y_true"].append(y_true)
    # predicciones["y_pred"].append(y_pred)

    print("Accuracy: %.4f" % acc)
    print("Balanced accuracy: %.4f" % b_acc)
    print("Recall: %.4f" % rec)
    print("Specificity: %.4f" % spec)
    print("AUC: %.4f" % rec)
    print("MCC: %.4f" % mcc)
    print("Precision: %.4f" % prec)
    print("F1-score: %.4f" % f1)

    plot_save_conf_matrix(
        y_true,
        y_pred,
        [pos_class[0], neg_class[0]],
        [pos_class[1], neg_class[1]],
        nombre,
    )

def train_and_evaluate_model_multiclass(
            modelo, X_train, y_train, X_test, y_test, nombre, classes
):
    """Función para entrenar y evaluar un clasificador en un problema multiclase"""
    #cu.train_and_evaluate_model(knn, X_train_scaled, y_train, X_test_scaled, y_test, f'{best_k}-NN',pos_class,neg_classes)
    y_true, y_pred = list(), list()

        # Entrenar modelo
    modelo.fit(X_train, y_train)

    # Evaluar modelo
    y_eval = modelo.predict(X_test)

    # store
    y_true.extend(y_test)
    y_pred.extend(y_eval)

    #Matriz de confusión
    #Columnas: Etiquetas predichas
    #Filas: Etiquetas reales
    #cm[etiqueta_real][etiqueta_predicha]

    cm = confusion_matrix(y_true, y_pred, labels=[c[0] for c in classes].sort())

    cm = pd.DataFrame(cm, columns=[c[0] for c in classes].sort(), index=[c[0] for c in classes].sort())

    performance = {}
    performancedf = pd.DataFrame()

    #Cálculo de medidas de desempeño
    for c in classes: #Para cada clase
        pos_class = c #Clase positive
        performance['clasificador'] = nombre
        neg_classes = [c for c in classes if c != pos_class] #Clases negative
        
        performance['clase'] = pos_class[1]
        
        #True Positives: Patrones de la clase positive clasificados como positive
        #cm[positive_class][positive_class]
        tp = cm[pos_class[0]][pos_class[0]]

        #True Negatives: Suma de los atrones de las clases negative clasificados como negative
        #cm[negative_class][negative_class]
        tn = sum([cm[c[0]][c[0]] for c in neg_classes])

        #False Positives: Patrones de las clases negative clasificados como positive
        #cm[negative_class][positive_class]
        fp = sum([cm[pos_class[0]][c[0]] for c in neg_classes])

        #False Negatives: Patrones de la clase positive clasificados como negative
        #cm[positive_class][negative_class]
        fn = sum([cm[c[0]][pos_class[0]] for c in neg_classes])

        acc = accuracy_score(y_true, y_pred)
        b_acc = balanced_accuracy_score(y_true, y_pred)
        #recall_score devuelve un array cuando average=None
        rec = recall_score(y_true, y_pred, labels=[pos_class[0]], average=None)[0]
        #f1_score devuelve un array cuando average=None
        f1 = f1_score(y_true, y_pred, labels=[pos_class[0]], average=None)[0]
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        performance['accuracy'] = acc
        performance['balanced_accuracy'] = b_acc
        performance['sensitivity'] = rec
        performance['specificity'] = tn / (fp + tn)
        performance['f1'] = f1
        performance['mcc'] = mcc

        performancedf = pd.concat([performancedf, pd.DataFrame(performance, index=[0])])

        write_results_to_file_multiclass(
            {
                "clasificador": nombre,
                "clase": pos_class[1],
                "accuracy": acc,
                "balanced_accuracy": b_acc,
                "sensitivity": rec,
                "specificity": tn / (fp + tn),
                "f1": f1,
                "mcc": mcc
            }
        )

    #Cálculo de macro avg para cada medida de desempeño
    macros = {}
    macros['clasificador'] = nombre
    macros['clase'] = 'macro_avg'
    macros['accuracy'] = sum(performancedf['accuracy']) / len(classes)
    macros['balanced_accuracy'] = sum(performancedf['balanced_accuracy']) / len(classes)
    macros['sensitivity'] = sum(performancedf['sensitivity']) / len(classes)
    macros['specificity'] = sum(performancedf['specificity']) / len(classes)
    macros['f1'] = sum(performancedf['f1']) / len(classes)
    macros['mcc'] = sum(performancedf['mcc']) / len(classes)

    #clasificador,clase,accuracy,balanced_accuracy,sensitivity,specificity,f1,mcc
    write_results_to_file_multiclass(
         {
            "clasificador": nombre,
            "clase": 'macro_avg',
            "accuracy": macros['accuracy'],
            "balanced_accuracy": macros['balanced_accuracy'],
            "sensitivity": macros['sensitivity'],
            "specificity": macros['specificity'],
            "f1": macros['f1'],
            "mcc": macros['mcc']
        }
    )

    #Cálculo de weighted avg para cada medida de desempeño
    weighted = {}
    proporciones = [sum(1 for l in y_true if l == c[0])/len(y_true) for c in classes] #Proporción de cada clase en el dataset de prueba
    weighted['clasificador'] = nombre
    weighted['clase'] = 'weighted_avg'
    weighted['accuracy'] = sum(performancedf['accuracy'] * proporciones)
    weighted['balanced_accuracy'] = sum(performancedf['balanced_accuracy'] * proporciones)
    weighted['sensitivity'] = sum(performancedf['sensitivity'] * proporciones)
    weighted['specificity'] = sum(performancedf['specificity'] * proporciones)
    weighted['f1'] = sum(performancedf['f1'] * proporciones)
    weighted['mcc'] = sum(performancedf['mcc'] * proporciones)

    write_results_to_file_multiclass(
         {
            "clasificador": nombre,
            "clase": 'weighted_avg',
            "accuracy": weighted['accuracy'],
            "balanced_accuracy": weighted['balanced_accuracy'],
            "sensitivity": weighted['sensitivity'],
            "specificity": weighted['specificity'],
            "f1": weighted['f1'],
            "mcc": weighted['mcc']
        }
    )

    performancedf = pd.concat([performancedf, pd.DataFrame(macros,index=[0]) ,pd.DataFrame(weighted, index=[0])])
    print(performancedf.drop(columns=['clasificador']))
    
    plot_save_conf_matrix(
        y_true,
        y_pred,
        [c[0] for c in classes],
        [c[0] for c in classes],
        nombre,
    )


def plot_scores(df, scores, legends, sort_by, title):
    with plt.style.context(["science", "grid"]):
        ax = df.sort_values(by=sort_by).plot(kind="barh", x="Classifier", y=scores)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.legend(legends, loc="center left", bbox_to_anchor=(1.0, 0.15))
        ax.tick_params(axis="y", which="both", length=0)
    fig = ax.get_figure()
    fig.savefig(os.path.join(results_path, "scores.png"), dpi=300, bbox_inches="tight")
    plt.show()
