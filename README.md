# Caracterización de flujos migratorios utilizando análisis espacio-temporal y aprendizaje de máquina

Este repositorio contiene los archivos y el código necesarios para la implementación de los experimentos incluidos en el proyecto de tesis titulado **"Caracterización de flujos migratorios utilizando análisis espacio-temporal y aprendizaje de máquina"**.
El proyecto emplea técnicas de aprendizaje supervisado para modelar y predecir diversas características de flujos migratorios entre México y Estados Unidos utilizando datos recopilados por el [Mexican Migration Project (MMP)](https://mmp.opr.princeton.edu/).

## Contenidos del repositorio

Se incluye una carpeta por experimento, dentro de las cuales se encuentran los Jupyter Notebooks para cada uno:
- **`nombre_experimento_clasificadores.ipynb`**: Incluye la implementación de imputación, normalización y codificación de datos, así como de validación cruzada, entrenamiento y prueba de los algoritmos de aprendizaje de máquina. Este archivo genera las subcarpetas dentro de la carpeta del experimento:
  - **`curvas_aprendizaje/`**: Carpeta para almacenar las curvas de aprendizaje obtenidas.
  - **`datos_imputados/`**: Conjuntos de datos con datos faltantes imputados.
  - **`matrices_confusion/`**: Carpeta para almacenar las matrices de confusión de cada modelo.
  - **`particiones/`**: Particiones de datos generadas. Si la partición a utilizar ya se ha realizado, se obtiene de los archivos generados previamente. Se incluyen las particiones de datos con las que se obtuvieron los resultados reportados en el documento de tesis.
  - **`resultados/`**: Carpeta para almacenar los resultados de prueba de cada modelo.

- **`nombre_experimento_limpieza_datos.ipynb`**: Incluye el código necesario para la selección de información relevante y la construcción del conjunto de datos para el experimento. El conjunto de datos generado se almacena en la carpeta **`datasets`**.
  - **NOTA**: para generar los conjuntos de datos, debe contar con los archivos fuente del [Mexican Migration Project (MMP)](https://mmp.opr.princeton.edu/). Póngase en contacto con el equipo del MMP para obtener más información.

- **`datasets`**: Carpeta en la que se almacenan los conjuntos de datos a utilizar. Se incluyen los conjuntos con los que se obtuvieron los resultados reportados.
- **`tesis_experiments_utils`**: Scripts que contienen funciones que implementan diversas tareas para el entrenamiento y prueba de modelos además de la generación y almacenamiento de archivos, curvas de aprendizaje y matrices de confusión.
- **`environment.yml`**: Archivo para reproducir el entorno experimental de Conda.
- **`README.md`**: Archivo de documentación del repositorio.

## Requisitos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/).
- Python 3.9 o superior.

## Instalación

1. Clonar este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/nombre-del-repositorio.git
   cd nombre-del-repositorio
   ```

2. Crear el entorno Conda utilizando el archivo `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```

3. Activar el entorno creado:

   ```bash
   conda activate nombre-del-entorno
   ```

   *(Reemplace `nombre-del-entorno` con el nombre definido en el archivo `environment.yml`.)*

4. Verificar que las dependencias se instalaron correctamente:

   ```bash
   conda list
   ```

**Nota**: El proyecto fue implementado originalmente en Ubuntu Linux 24.04 y se recomienda seguir las instrucciones de instalación en este sistema operativo para evitar posibles problemas de compatibilidad.

## Uso

1. Asegúrese de que el entorno Conda esté activo:

   ```bash
   conda activate nombre-del-entorno
   ```

2. Ejecute los Jupyter Notebooks o scripts según sea necesario:

   ```bash
   jupyter notebook
   ```

   *(Abra el archivo deseado en la interfaz del navegador.)*

3. Como prueba inicial, ejecute el notebook de clasificadores en la carpeta del primer experimento.

## Cómo citar
Si utiliza información, código o figuras tomadas de este repositorio, cite la fuente de la siguiente manera:
``` 
Pérez Ramírez, J. D. (2024). *Caracterización de flujos migratorios utilizando análisis espacio-temporal y aprendizaje de máquina*. Instituto Politécnico Nacional. Recuperado de https://github.com/Danperam/Experimentos_caracterizacion_flujos_migratorios_Mexico_EEUU_ML
```

## Declaración de Derechos de Autor

El código su implementación y la información presentada en este repositorio son propiedad del Instituto Politécnico Nacional.

Los usuarios de la información no deben reproducir el contenido textual, gráficas o datos del trabajo sin el permiso expreso del autor y/o directores. Este puede ser obtenido escribiendo a las siguientes direcciones de correo: danprjs@gmail.com, mtorres@cic.ipn.mx, quintero@cic.ipn.mx. Si el permiso se otorga, el usuario deberá dar agradecimiento correspondiente y citar la fuente de este.
