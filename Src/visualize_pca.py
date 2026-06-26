
#Cargar paquetes:
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#Ruta de los datos
ruta_actual = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\datos_estriado_limpios.tsv"

#Cargar archivos limpios:
archivo_cargado= pd.read_csv(ruta_actual, sep="\t", index_col=0)
print(archivo_cargado[:10])

#Crear lista con las etiquetas de salud
Health_label= []
for muestra in archivo_cargado.index:
    if "CTRL" in muestra:
        Health_label.append("Control")
    elif "PD" in muestra:
        Health_label.append("Parkinson")
        
#Aplicamos PCA
pca = PCA(n_components=2)
datos_reducido = pca.fit_transform(archivo_cargado)

#Construimos el dataframe para el gráfico
df_graph= pd.DataFrame({
    "PC1" : datos_reducido[:, 0],
    "PC2" : datos_reducido[:, 1],
    "Label": Health_label
})
print(df_graph.head())

#Construimos el gráfico con seaborn
sns.scatterplot(
    data= df_graph,
    x= "PC1",
    y= "PC2",
    hue= "Label"
)
plt.title("PCA - Expresión Génica en Tejido Estriado (Parkinson vs Control)")
plt.show()