# Snakefile 
import os
os.environ["PATH"] += r";C:\Program Files\R\R-4.6.1\bin"

rule all:
    input:
        "Data/Processed/datos_estriado_limpios.tsv",
        "Results/Imagenes/PCA_inicial.png",
        "Data/Processed/datos_train_completos.tsv",
        "Data/Processed/datos_test_completos.tsv",
        "Data/Processed/train_top100_genes.tsv",
        "Data/Processed/test_top100_genes.tsv",
        "Results/Imagenes/volcano_deseq2.png",
        "Results/Imagenes/heatmap_deseq2.png",
        "Results/Imagenes/pca_top100.png",
        "Results/Imagenes/curvas_rendimiento_ml.png",
        "Data/Processed/matriz_confusion_test.csv",
        "Data/Processed/reporte_clasificacion_test.txt",
        "Results/Imagenes/figura_curvas_test.png",
        "Results/Imagenes/shap_importancia_barras.png",
        "Results/Imagenes/shap_impacto_beeswarm.png"

# Import data
rule import_data:
    input:
        counts = "Data/Raw/GSE205450_counts.table.txt.gz",
        soft = "Data/Raw/GSE205450_family.soft/GSE205450_family.soft"
    output:
        matriz_limpia = "Data/Processed/datos_estriado_limpios.tsv"
    script:
        "Src/1_import_data.py"

# Control de Calidad: PCA Global de datos limpios
rule run_pca_inicial:
    input:
        matriz_limpia = "Data/Processed/datos_estriado_limpios.tsv"
    output:
        grafico_pca_inicial = "Results/Imagenes/PCA_inicial.png"
    script:
        "Src/2_PCA_inicial.py"

# Train/Test split agrupado por Donor_ID
rule train_test_split:
    input:
        matriz_limpia = "Data/Processed/datos_estriado_limpios.tsv"
    output:
        train_completos = "Data/Processed/datos_train_completos.tsv",
        test_completos = "Data/Processed/datos_test_completos.tsv"
    script:
        "Src/3_train_test_split.py"

# DESeq2 (Selección principal de características)
rule deseq2_feature_selection:
    input:
        counts = "Data/Processed/datos_train_completos.tsv",
        test = "Data/Processed/datos_test_completos.tsv"
    output:
        train_top100 = "Data/Processed/train_top100_genes.tsv",
        test_top100 = "Data/Processed/test_top100_genes.tsv",
        volcano_deseq2 = "Results/Imagenes/volcano_deseq2.png",
        heatmap_deseq2 = "Results/Imagenes/heatmap_deseq2.png"
    script:
        "Src/4_run_deseq2.R"

# PCA_2: train data
rule run_pca_train:
    input:
        train_genes = "Data/Processed/train_top100_genes.tsv"
    output:
        grafico_pca = "Results/Imagenes/pca_top100.png"
    script:
        "Src/5_PCA_train.py"

# Comparativa modelos ML
rule ml_modeling:
    input:
        train_top100 = "Data/Processed/train_top100_genes.tsv"
    output:
        grafico_curvas = "Results/Imagenes/curvas_rendimiento_ml.png"
    script:
        "Src/6_ml_modeling.py"

# Entrenamiento y evaluación en test (Lasso + SVM-RBF)
rule svm_rbf_test:
    input:
        train_top100 = "Data/Processed/train_top100_genes.tsv",
        test_top100 = "Data/Processed/test_top100_genes.tsv"
    output:
        matriz_confusion = "Data/Processed/matriz_confusion_test.csv",
        reporte_txt = "Data/Processed/reporte_clasificacion_test.txt",
        grafico_test = "Results/Imagenes/figura_curvas_test.png"
    script:
        "Src/7_evaluacion_test.py"

# SHAP: Interpretabilidad biológica
rule svm_rbf_shap:
    input:
        train_top100 = "Data/Processed/train_top100_genes.tsv",
        test_top100 = "Data/Processed/test_top100_genes.tsv"
    output:
        shap_barras = "Results/Imagenes/shap_importancia_barras.png",
        shap_beeswarm = "Results/Imagenes/shap_impacto_beeswarm.png"
    script:
        "Src/8_SVM_RBF_shap.py"