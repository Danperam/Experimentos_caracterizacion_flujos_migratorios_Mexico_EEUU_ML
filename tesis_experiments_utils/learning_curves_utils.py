import os
import numpy as np
import pandas as pd
import ast
from tesis_experiments_utils.files_utils import (
    learning_curves_path,
    write_learning_curves_to_file,
)
import matplotlib.pyplot as plt
import scienceplots
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold


def plot_and_save_learning_curve(
    clf,
    name,
    X,
    y,
    scoring,
    train_sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=94),
    n_jobs=-1,
    pre_dispatch="all",
    x_label="Instancias de entrenamiento",
    y_label="Coeficiente de Correlación de Matthews",
):
    """Función para graficar y almacenar las curvas de aprendizaje de un clasificador
    clf: classifier, clasificador a evaluar
    name: str, nombre del clasificador
    X: dataframe, matriz de atributos
    y: dataframe, vector de etiquetas
    scoring: str, métrica de evaluación
    train_sizes: list, proporciones de instancias de entrenamiento, por defecto [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cv: int o cross-validation generator, por defecto StratifiedKFold(n_splits=10, shuffle=True, random_state=94)
    n_jobs: int, número de trabajos en paralelo, por defecto -1
    pre_dispatch: int o str, número de trabajos generados para ser procesados paralelamente, por defecto 'all'
    """
    train_size_abs, train_scores, validation_scores = learning_curve(
        clf,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        pre_dispatch=pre_dispatch,
    )  # type: ignore

    train_stds = []
    validation_stds = []

    for cv_train_scores, cv_validation_scores in zip(train_scores, validation_scores):
        train_stds.append(cv_train_scores.std())
        validation_stds.append(cv_validation_scores.std())

    write_learning_curves_to_file(
        {
            "Classifier": name,
            "Train_sizes": train_size_abs,
            "Train_scores": train_scores,
            "Validation_scores": validation_scores,
        }
    )

    training_color = "firebrick"
    validation_color = "darkslategrey"

    with plt.style.context(["science", "grid"]):
        fig, ax = plt.subplots()
        ax.plot(
            train_size_abs,
            train_scores.mean(axis=1),
            label="Entrenamiento",
            marker="o",
            color=training_color,
        )
        ax.plot(
            train_size_abs,
            validation_scores.mean(axis=1),
            label="Validación",
            marker="o",
            color=validation_color,
            linestyle="dashed",
        )
        ax.fill_between(
            train_size_abs,
            train_scores.mean(axis=1) - np.array(train_stds),
            train_scores.mean(axis=1) + np.array(train_stds),
            color=training_color,
            alpha=0.2,
        )
        ax.fill_between(
            train_size_abs,
            validation_scores.mean(axis=1) - np.array(validation_stds),
            validation_scores.mean(axis=1) + np.array(validation_stds),
            color=validation_color,
            alpha=0.2,
        )
        ax.set_title(f"Curva de aprendizaje, {name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
        # print(fig.get_size_inches())
        # ax.grid()

    plt.savefig(
        os.path.join(learning_curves_path, name + ".png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_and_save_all_learning_curves(output_file_name="curvas_de_aprendizaje"):
    pd.options.display.max_colwidth = (
        1024  # Ajustar el ancho de las columnas para no truncar los datos del DataFrame
    )
    lc = pd.read_csv(os.path.join(learning_curves_path, "curvas_de_aprendizaje.csv"))
    classifiers = lc["Classifier"].unique()
    lc_obj = {}
    lc_obj["Classifier"] = []
    lc_obj["Train_sizes"] = []
    lc_obj["Train_scores"] = []
    lc_obj["Validation_scores"] = []
    for c in classifiers:
        lc_obj["Classifier"].append(c)
        lc_obj["Train_sizes"].append(lc[lc["Classifier"] == c]["Train_sizes"].to_list())
        ts = lc[lc["Classifier"] == c]["Train_scores"].to_list()
        lc_obj["Train_scores"].append(np.array([ast.literal_eval(t) for t in ts]))
        vs = lc[lc["Classifier"] == c]["Validation_scores"].to_list()
        lc_obj["Validation_scores"].append(np.array([ast.literal_eval(v) for v in vs]))

    with plt.style.context(["science", "grid"]):
        fig, ax = plt.subplots(figsize=(4.5, 3.2625))  # Default: (3.5, 2.625)
        for r in lc_obj["Classifier"]:
            ax.plot(
                lc_obj["Train_sizes"][lc_obj["Classifier"].index(r)],
                lc_obj["Validation_scores"][lc_obj["Classifier"].index(r)].mean(axis=1),
                label=f"{r}",
                marker="o",
                linestyle="solid",
            )
            ax.fill_between(
                lc_obj["Train_sizes"][lc_obj["Classifier"].index(r)],
                lc_obj["Validation_scores"][lc_obj["Classifier"].index(r)].mean(axis=1)
                - lc_obj["Validation_scores"][lc_obj["Classifier"].index(r)].std(
                    axis=1
                ),
                lc_obj["Validation_scores"][lc_obj["Classifier"].index(r)].mean(axis=1)
                + lc_obj["Validation_scores"][lc_obj["Classifier"].index(r)].std(
                    axis=1
                ),
                alpha=0.2,
            )
            ax.set_title("Curvas de aprendizaje, validación")
            ax.set_xlabel("Instancias de entrenamiento")
            # ax.grid()
            ax.set_ylabel("Coeficiente de Correlación de Matthews")
            ax.legend()
    plt.savefig(
        os.path.join(learning_curves_path, f"{output_file_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
