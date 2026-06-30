#carga
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from graphs_functions import pca_plot

#Ruta datos_train_completos.tsv
ruta= r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
#Importamos train data
datos_train= pd.read_csv(ruta, sep="\t", index_col=0)
#Eliminamos Health_Label
train_no_Label= datos_train.drop(columns=["Health_Label"])


#Aplicamos PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(train_no_Label)

#Construimos el df para el gráfico de PCA
df_graph= pd.DataFrame({
    "PC1" : df_pca[:, 0],
    "PC2" : df_pca[:, 1],
    "Label": datos_train["Health_Label"]
})
print(df_graph.head())

#Llamamos a la función pca_plot
ruta_grafico = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\pca_top100.png"
pca_plot(
    datos_reducido=df_graph,
    ruta_salida=ruta_grafico,
    title="PCA- Top 100 Genes Seleccionados")

