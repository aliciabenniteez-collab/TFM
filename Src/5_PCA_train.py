import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from graphs_functions import pca_plot

# Soporte para entorno Snakemake y Standalone
if 'snakemake' not in locals():
    class MockSub:
        pass
    class Mock:
        def __init__(self):
            self.input = MockSub()
            self.input.train_genes = (
                r"Data/Processed/train_top100_genes.tsv"
            )
            self.output = MockSub()
            self.output.grafico_pca = (
                r"Results/Imagenes/pca_top100.png"
            )
    snakemake = Mock()

# 1. Cargar datos de entrenamiento reducidos (Top 100 genes)
datos_train = pd.read_csv(
    snakemake.input.train_genes, 
    sep="\t", 
    index_col=0
)

# 2. Separar la matriz de conteos de las etiquetas de salud
X_raw = datos_train.select_dtypes(include=[np.number])
# 3. Transformación Log2 y Escalado Z-Score
X_log = np.log2(X_raw + 1)
X_scaled = StandardScaler().fit_transform(X_log)

# 4. Ajustar y transformar PCA a 2 componentes
pca = PCA(n_components=2)
df_pca = pca.fit_transform(X_scaled)

var_exp = pca.explained_variance_ratio_ * 100
print(f"[INFO] PC1 explicada: {var_exp[0]:.2f}%")
print(f"[INFO] PC2 explicada: {var_exp[1]:.2f}%")

# 5. Construir DataFrame final para la visualización
df_graph = pd.DataFrame({
    "PC1": df_pca[:, 0],
    "PC2": df_pca[:, 1],
    "Label": datos_train["Health_Label"]
})

# 6. Asegurar directorio de salida y generar el gráfico
os.makedirs(
    os.path.dirname(snakemake.output.grafico_pca), 
    exist_ok=True
)

pca_plot(
    datos_reducido=df_graph,
    ruta_salida=snakemake.output.grafico_pca,
    title=f"PCA - Top 100 Genes (PC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%)"
)

print(f"[EXITO] Gráfico PCA guardado en: {snakemake.output.grafico_pca}")