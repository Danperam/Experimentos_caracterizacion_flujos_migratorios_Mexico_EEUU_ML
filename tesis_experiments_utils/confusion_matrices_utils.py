import os
import numpy as np
from tesis_experiments_utils.files_utils import conf_matrices_path
import matplotlib.pyplot as plt
import scienceplots
from sklearn.metrics import ConfusionMatrixDisplay

# ====================================================================================================
#               FUNCIONES PARA GENERAR Y ALMACENAR LAS MATRICES DE CONFUSIÓN
# ====================================================================================================


# Función para mostrar y almacenar las matrices de confusión por separado
def plot_save_conf_matrix(y_true, y_pred, labels, display_labels, title):
    """Función para mostrar y almacenar las matrices de confusión por separado
    y_true: etiquetas reales del conjunto de datos
    y_pred: etiquetas predichas por el clasificador
    labels: etiquetas de las clases
    display_labels: etiquetas a mostrar en la matriz de confusión
    title: título de la matriz de confusión"""
    with plt.style.context("science"):
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            normalize=None,
            labels=labels,
            display_labels=display_labels,
            colorbar=False,
            cmap="binary",
        )
        disp.ax_.set_title(title)
        disp.ax_.set_xlabel("Etiqueta predicha")
        disp.ax_.set_ylabel("Etiqueta real")
        disp.ax_.tick_params(axis="both", which="both", length=0)
    plt.savefig(
        os.path.join(conf_matrices_path, title + ".png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


# Funcion para mostrar las matrices de confusión juntas en una sola figura
def plot_save_all_conf_matrices(predicciones, labels, display_labels):
    """Función para mostrar las matrices de confusión juntas en una sola figura
    predicciones: diccionario con las predicciones de los clasificadores
    labels: etiquetas de las clases
    display_labels: etiquetas a mostrar en la matriz de confusión"""
    with plt.style.context("science"):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 6))
        ax = axes.flatten()
        for i in range(len(predicciones["Classifier"])):
            disp = ConfusionMatrixDisplay.from_predictions(
                predicciones["y_true"][i],
                predicciones["y_pred"][i],
                normalize=None,
                labels=labels,
                display_labels=display_labels,
                colorbar=False,
                cmap="binary",
                ax=ax[i],
            )
            disp.ax_.set_title(predicciones["Classifier"][i])
            disp.ax_.set_xlabel("Etiqueta predicha")
            disp.ax_.set_ylabel("Etiqueta real")
            disp.ax_.tick_params(axis="both", which="both", length=0)

        fig.suptitle("Confusion Matrices", fontweight="bold")
        plt.tight_layout()
    plt.savefig(
        os.path.join(conf_matrices_path, "all_confusion_matrices.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
