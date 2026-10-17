# Códigos y resultados de la tesis

## 1. Descripción general

Este repositorio contiene los códigos, conjuntos de datos y resultados correspondientes al trabajo de tesis dedicado a la modelación mensual del Índice de Estado Trófico basado en fósforo total, IET(PT), en la cuenca del río Santa Lucía Chico, Uruguay, mediante algoritmos de aprendizaje automático.

El trabajo comprende la preparación y el análisis de datos ambientales, así como la implementación, optimización y evaluación de modelos basados en XGBoost y redes neuronales multicapa (MLP).

## 2. Estructura del proyecto

El repositorio se organiza en tres carpetas principales:

- `1_Conformacion_Dataset`
- `2_Modelaciones_XGBoost`
- `3_Modelaciones_MLP`

La siguiente estructura muestra los directorios y archivos principales del repositorio, omitiendo los archivos individuales de datos y resultados, con excepción del conjunto de datos definitivo utilizado en las modelaciones:

```text
.
├── 1_Conformacion_Dataset/
│   ├── 0-Datasets/
│   │   └── dataset_feature_eng.parquet
│   ├── C1.ipynb
│   ├── C2.ipynb
│   └── C3.ipynb
│
├── 2_Modelaciones_XGBoost/
│   ├── linea_de_base/
│   │   ├── xgb_LB.ipynb
│   │   ├── pipeline_modelado_baseline.py
│   │   ├── my_metrics.py
│   │   ├── visualization.py
│   │   └── reportes_html_baseline.py
│   │
│   └── modelos_optimizados/
│       ├── xgb_modelos_completos.ipynb
│       ├── xgb_modelos_reducidos.ipynb
│       ├── pipeline_modelado.py
│       ├── my_metrics.py
│       ├── visualization.py
│       └── reportes_html.py
│
├── 3_Modelaciones_MLP/
│   ├── mlp_modelos_completos.ipynb
│   ├── mlp_modelos_reducidos.ipynb
│   ├── pipeline_mlp.py
│   ├── my_metrics_mlp.py
│   ├── visualization_mlp.py
│   └── reportes_html_mlp.py
│
├── requirements.txt
└── README.md
```

### 2.1. Carpeta `1_Conformacion_Dataset`

Contiene los cuadernos de Jupyter utilizados para la integración, preparación y análisis de los datos ambientales empleados en las modelaciones.

**Subcarpeta `0-Datasets`**

Contiene los registros originales de calidad de agua, hidrología y meteorología, junto con los datos integrados e imputados provenientes del trabajo de Pastorini et al. (2024):

Pastorini, M., Rodríguez, R., Etcheverry, L., Castro, A., & Gorgoglione, A. (2024). *Enhancing environmental data imputation: A physically-constrained machine learning framework*. Science of the Total Environment, 926, 171773.

En esta subcarpeta también se almacenan los conjuntos de datos generados durante las distintas etapas de procesamiento, incluido `dataset_feature_eng.parquet`, el conjunto definitivo utilizado en las modelaciones (ver sección 3.1).

**Cuadernos de Jupyter**

- `C1.ipynb`: integra los datos imputados en un único conjunto de datos mensual, considerando el período de estudio 2014–2022, e incorpora variables adicionales derivadas de los registros hidrometeorológicos.
- `C2.ipynb`: integra los registros originales, sin imputación, de las distintas fuentes de información, para disponer de un conjunto de datos de referencia utilizado en el análisis posterior.
- `C3.ipynb`: realiza el análisis exploratorio de los datos, incorpora las variables objetivo IET(PT), construye variables con rezagos temporales y desarrolla los análisis de correlación. También incluye ajustes puntuales en la preparación del conjunto de datos utilizado para la modelación.

### 2.2. Carpeta `2_Modelaciones_XGBoost`

Contiene los códigos y resultados de las modelaciones con XGBoost, con Optuna para la optimización de hiperparámetros y SHAP para la interpretabilidad.

**Subcarpeta `linea_de_base`**: modelos de referencia para comparar con los modelos optimizados.

- `xgb_LB.ipynb`: cuaderno desde el que se ejecutan las modelaciones.
- `pipeline_modelado_baseline.py`: procedimiento de modelación de línea de base.

**Subcarpeta `modelos_optimizados`**

- `xgb_modelos_completos.ipynb`: modelaciones con los conjuntos de variables predictoras completos.
- `xgb_modelos_reducidos.ipynb`: modelaciones con configuraciones reducidas, orientadas a disminuir la cantidad de variables predictoras y el número efectivo de árboles del ensamble.
- `pipeline_modelado.py`: entrenamiento, validación temporal y optimización de hiperparámetros.

Ambas subcarpetas incluyen módulos auxiliares para el cálculo de métricas (`my_metrics.py`), la generación de gráficos (`visualization.py`) y de reportes HTML (`reportes_html.py`).

### 2.3. Carpeta `3_Modelaciones_MLP`

Contiene los modelos de perceptrón multicapa (MLP), con la misma lógica que `modelos_optimizados`. Utiliza PyTorch para las redes, Optuna para la optimización de hiperparámetros y Captum (Integrated Gradients) para la interpretabilidad.

- `mlp_modelos_completos.ipynb`: modelaciones con los conjuntos de variables predictoras completos.
- `mlp_modelos_reducidos.ipynb`: modelaciones con configuraciones reducidas.
- `pipeline_mlp.py`: redes neuronales, entrenamiento, validación temporal y optimización de hiperparámetros.

Incluye también los módulos auxiliares `my_metrics_mlp.py`, `visualization_mlp.py` y `reportes_html_mlp.py`.

## 3. Funcionamiento general de los códigos

El procesamiento se organiza en dos etapas principales: preparación de los conjuntos de datos y ejecución de las modelaciones.

El flujo general de trabajo es el siguiente:

**C1 / C2 → C3 → Modelaciones XGBoost y MLP**

### 3.1. Preparación de los datos

La preparación de los datos se realiza mediante los cuadernos `C1.ipynb`, `C2.ipynb` y `C3.ipynb`. Como resultado de este proceso, en el cuaderno `C3.ipynb` se genera el archivo `dataset_feature_eng.parquet`, que constituye el conjunto de datos definitivo utilizado en las modelaciones y contiene las variables objetivo IET(PT), las variables predictoras originales y las variables adicionales generadas durante el procesamiento.

Este archivo ya se encuentra incluido en el repositorio (carpeta `0-Datasets`), por lo que no es necesario volver a ejecutar los cuadernos de preparación para disponer del conjunto de datos utilizado en las modelaciones.

### 3.2. Ejecución de las modelaciones

Las modelaciones XGBoost y MLP se ejecutan desde los respectivos cuadernos de Jupyter.

Estos cuadernos importan las funciones y clases definidas en los módulos `.py` correspondientes, desde los que se implementan los procedimientos de entrenamiento, validación, optimización, evaluación y generación de resultados, por lo que no es necesario ejecutar los módulos individualmente.

Los reportes HTML generados durante las modelaciones fueron excluidos del repositorio para reducir su tamaño. Estos pueden generarse nuevamente mediante la ejecución de los cuadernos de Jupyter correspondientes.

## 4. Requisitos y ejecución

Los códigos fueron desarrollados en Python 3.12, utilizando cuadernos de Jupyter y módulos Python.

Las dependencias necesarias se encuentran especificadas en el archivo `requirements.txt`.

Para reproducir el entorno de ejecución, se recomienda crear un entorno virtual con esa versión de Python e instalar las dependencias mediante:

```bash
python -m pip install -r requirements.txt
```
Una vez instalado el entorno, los procedimientos pueden ejecutarse desde los cuadernos de Jupyter correspondientes, conservando la estructura de carpetas del proyecto y las rutas utilizadas para acceder a los conjuntos de datos. Ejecutar los cuadernos regenera los archivos de salida; los resultados de la tesis ya están incluidos en las carpetas correspondientes y pueden consultarse sin reejecutar.

### 4.1. Reproducibilidad de los resultados

Los cuadernos y módulos Python contienen las configuraciones utilizadas en las ejecuciones, incluyendo los procedimientos de entrenamiento, la optimización de hiperparámetros y las semillas aleatorias. Con esas semillas, las ejecuciones repetidas en el entorno de desarrollo (Python 3.12.12, Windows 11 AMD64, solo CPU: AMD Ryzen 7 7730U, 2,00 GHz, 16 GB de RAM) produjeron resultados idénticos. No se verificó la reproducibilidad en otros equipos, donde diferencias en las versiones de las bibliotecas, en el hardware o en el sistema operativo podrían generar variaciones numéricas menores.
