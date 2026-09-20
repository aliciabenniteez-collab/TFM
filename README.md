# Modelado de Machine Learning para la clasificación predictiva y selección de características en la enfermedad de Parkinson mediante algoritmos de aprendizaje supervisado.
# Autor: Alicia Benítez López. Trabajo Final de Máster en Bioinformática.

![Snakemake](https://img.shields.io/badge/Snakemake-v9.x-blue?style=flat-square&logo=snakemake)
![R / Bioconductor](https://img.shields.io/badge/R-v4.3%20%7C%20DESeq2-blue?style=flat-square&logo=R)
![Python / ML](https://img.shields.io/badge/Python-v3.12%20%7C%20SHAP-green?style=flat-square&logo=python)
![Conda / Mamba](https://img.shields.io/badge/Dependencies-Conda%2FMamba-success?style=flat-square&logo=anaconda)

Este repositorio contiene el código fuente, la canalización bioinformática e instrucciones de replicación correspondientes al Trabajo Final de Máster (TFM) titulado Modelado de Machine Learning para
la clasificación predictiva y selección de características en la enfermedad de Parkinson mediante algoritmos de aprendizaje supervisado.


# Descripción del Proyecto
El objetivo principal de este trabajo implementar y evaluar un flujo de trabajo (pipeline) bioinformático y de Machine Learning para clasificar muestras de pacientes con la enfermedad de Parkinson frente a controles sano.

# Estructura del Flujo de Trabajo (Snakemake)
El pipeline automatizado abarca el preprocesamiento, análisis de componentes principales (PCA),  el entrenamiento de los modelos y el SHAP.

[DAG Pipeline](Results/Imagenes/DAG.png)

# Arquitectura de Entornos Modulares
Con el fin de garantizar la reproducibilidad del proyecto y evitar conflictos en las dependencias, se implementó una arquitectura de tres entornos gestionados por Snakemake y Mamba.

1. Entorno Base ("snakemake_env"): Entorno base (Snakemake).
2. Entornos ("r_deseq2.yaml"): Entorno de análisis diferencial en R (DESeq2, Bioconductor).
3. (python_ml.yaml): Entorno de procesamiento de datos, modelos de Machine Learning e interpretabilidad con SHAP (Scikit-Learn, SHAP, Pandas).
# Estructura del Repositorio
TFM_BIOINFORMATICA
├── .git
├── .gitignore
├── .snakemake
│   ├── auxiliary
│   ├── conda
│   ├── conda-archive
│   ├── incomplete
│   ├── iocache
│   ├── log 
│   ├── metadata
│   ├── scripts
│   │   └── tmplgak1pcc.import_data.py
│   ├── shadow
│   └── singularity
├── Data
│   ├── Processed
│   │   ├── GSE205450_metadata.csv
│   │   ├── datos_estriado_limpios.tsv
│   │   ├── datos_test_completos.tsv
│   │   ├── datos_train_completos.tsv
│   │   ├── matriz_confusion_test.csv
│   │   ├── reporte_clasificacion_test.txt
│   │   ├── test_top100_genes.tsv
│   │   ├── train_top100_genes.tsv
│   │   └── train_top100_genes_ttest.tsv
│   └── Raw
│       ├── GSE205450_counts.table.txt.gz
│       ├── GSE205450_family.soft
│       │   └── GSE205450_family.soft
│       ├── GSE205450_family.soft.gz
│       └── GSE205450_lcpm.table.txt.gz
├── Environment
│   ├── python_ml.yaml
│   ├── r_deseq2.yaml
│   └── snakemake_env.yaml
├── Logs
├── Manuscript
├── Notebooks
├── Parkinson.bib
├── README.md
├── Results
│   └── Imagenes
│       ├── DAG.png
│       ├── PCA_inicial.png
│       ├── curvas_rendimiento_ml.png
│       ├── figura_curvas_test.png
│       ├── heatmap_deseq2.png
│       ├── heatmap_top100.png
│       ├── pca_top100.png
│       ├── shap_impacto_beeswarm.png
│       ├── shap_importancia_barras.png
│       └── volcano_deseq2.png
├── Snakefile
├── Src
│   ├── 1_import_data.py
│   ├── 2_PCA_inicial.py
│   ├── 3_train_test_split.py
│   ├── 4_run_deseq2.R
│   ├── 5_PCA_train.py
│   ├── 6_ml_modeling.py
│   ├── 7_evaluacion_test.py
│   ├── 8_SVM_RBF_shap.py
│   ├── __pycache__
│   ├── feature_selection.py
│   ├── graphs_functions.py
│   └── verific_split.py
└── pipeline.dot

# Requisitos e Instalación

Tener instalado Miniforge o Mamba.

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/aliciabenniteez-collab/TFM.git 
   cd TFM_BIOINFORMATICA

2. Crear y activar el entorno
mamba env -n create -f Environment/snakemake_env.yaml
conda activate snakemake_env

3. Ejecutar el pipeline automatizado.
snakemake --use-conda --conda-frontend mamba --cores 4
Nota: La primera vez que se ejecute, Snakemake creará automáticamente los entornos r_deseq2 y python_ml en la carpeta .snakemake/conda/.


  