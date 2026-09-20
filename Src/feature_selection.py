import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from graphs_functions import gen_Volcano_plot, heatmap_train

# Soporte para entorno Snakemake y Standalone
if 'snakemake' not in locals():
    class MockSub:
        pass
    class Mock:
        def __init__(self):
            self.input = MockSub()
            self.input.train_completos = (
                r"Data/Processed/datos_train_completos.tsv"
            )
            self.input.test_completos = (
                r"Data/Processed/datos_test_completos.tsv"
            )
            self.output = MockSub()
            self.output.train_top100_ttest = (
                r"Data/Processed/train_top100_genes_ttest.tsv"
            )
            self.output.volcano = (
                r"Results/Imagenes/volcano_ttest.png"
            )
            self.output.heatmap = (
                r"Results/Imagenes/heatmap_ttest.png"
            )
    snakemake = Mock()

#Cargar archivos
archivo_cargado = pd.read_csv(
    snakemake.input.train_completos, 
    sep="\t", 
    index_col=0
)
test_data = pd.read_csv(
    snakemake.input.test_completos, 
    sep="\t", 
    index_col=0
)

#Separar por grupo de estudio
sub_df_ctrl = archivo_cargado[archivo_cargado["Health_Label"] == "Control"]
sub_df_pd = archivo_cargado[archivo_cargado["Health_Label"] == "Parkinson"]

#Eliminar todas las columnas no genéticas de forma segura
cols_excluir = [
    "Health_Label", "Donor_ID", "RIN", "PMI", 
    "Age", "Gender", "Region", "Disease"
]

sub_df_ctrl = sub_df_ctrl.drop(columns=cols_excluir, errors="ignore")
sub_df_pd = sub_df_pd.drop(columns=cols_excluir, errors="ignore")

#Convertir solo columnas con datos numéricos limpios
sub_df_ctrl = sub_df_ctrl.apply(pd.to_numeric, errors='coerce').fillna(0)
sub_df_pd = sub_df_pd.apply(pd.to_numeric, errors='coerce').fillna(0)

#T-Test gen por gen
lista_genes = sub_df_ctrl.columns.tolist()
resultados = []

for gen in lista_genes:
    estadistico, p_valor = ttest_ind(
        sub_df_ctrl[gen], 
        sub_df_pd[gen], 
        equal_var=False,
        nan_policy='omit'
    )
    resultados.append({"Gene": gen, "p_value": p_valor})

df_resultados = pd.DataFrame(resultados)

#Filtrar Top 100 genes por p-valor
orden_pvalue = df_resultados.sort_values(by="p_value", ascending=True)
filtro_pvalue = orden_pvalue.head(100)
genes_top = filtro_pvalue["Gene"].tolist()

#Generar conjuntos reducidos
train_top100_ttest = archivo_cargado[genes_top + ["Health_Label"]]
test_top100_ttest = test_data[genes_top + ["Health_Label"]]

#Asegurar directorios
os.makedirs(
    os.path.dirname(snakemake.output.train_top100_ttest), 
    exist_ok=True
)
os.makedirs(
    os.path.dirname(snakemake.output.volcano), 
    exist_ok=True
)

# Exportar matriz filtrada
train_top100_ttest.to_csv(snakemake.output.train_top100_ttest, sep="\t")

print(f"Train shape (t-test): {train_top100_ttest.shape}")
print(f"Test shape  (t-test): {test_top100_ttest.shape}")

#Volcano plot y heatmap
gen_Volcano_plot(
    df_resultados, sub_df_ctrl, sub_df_pd, snakemake.output.volcano
)

train_no_Label = train_top100_ttest.drop(
    columns=["Health_Label"]
).apply(
    pd.to_numeric,
    errors="coerce"
)
print(
    f"Genes antes del filtrado del heatmap: "
    f"{train_no_Label.shape[1]}"
)
print(
    f"Genes después del filtrado del heatmap: "
    f"{train_no_Label.shape[1]}"
)
heatmap_train(
    train_no_Label,
    train_top100_ttest,
    snakemake.output.heatmap
)
