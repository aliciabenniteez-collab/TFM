#Cargar librerías
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Entorno de fallback / standalone
if 'snakemake' not in locals():
    class MockSub:
        pass
    class Mock:
        def __init__(self):
            self.input = MockSub()
            self.input.matriz_limpia = (
                r"Data/Processed/datos_estriado_limpios.tsv"
            )
            self.output = MockSub()
            self.output.grafico_pca_inicial = (
                r"Results/Imagenes/PCA_inicial.png"
            )
    snakemake = Mock()

#Cargar archivo
df = pd.read_csv(snakemake.input.matriz_limpia, sep="\t", index_col=0)

#Descartar columnas metadatos clinicos
cols_clinicas = ["Health_Label", "Donor_ID", "Age", "PMI", "RIN", "Gender", "Region", "Disease"]
solo_genes = df.drop(columns=cols_clinicas, errors="ignore")

#Convertir todas las columnas restantes a numerico e imputar NaNs con 0
solo_genes = solo_genes.apply(pd.to_numeric, errors='coerce').fillna(0)

#Eliminar genes sin varianza (std == 0) para evitar divisiones por cero
solo_genes = solo_genes.loc[:, solo_genes.std(axis=0) > 0]

print(f"[INFO] Genes validos identificados para PCA: {solo_genes.shape[1]}")

#Transformacion Log2 y Escalado
X_log = np.log2(solo_genes + 1)
X_scaled = StandardScaler().fit_transform(X_log)

#PCA 
pca = PCA(n_components=2)
datos_reducido = pca.fit_transform(X_scaled)

var_exp = pca.explained_variance_ratio_ * 100
print(f"[INFO] PC1 explicada: {var_exp[0]:.2f}%")
print(f"[INFO] PC2 explicada: {var_exp[1]:.2f}%")

#Construir DataFrame para el grafico
df_graph = pd.DataFrame({
    "PC1": datos_reducido[:, 0],
    "PC2": datos_reducido[:, 1],
    "Label": df["Health_Label"]
})

#grafico
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_graph,
    x="PC1",
    y="PC2",
    hue="Label",
    palette={"Control": "#2b5c8f", "Parkinson": "#d95f02"},
    alpha=0.8,
    s=70
)

plt.title(
    f"PCA Inicial - Expresión Génica Global ({solo_genes.shape[1]} genes)\n"
    f"PC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%"
)
plt.xlabel(f"Componente Principal 1 ({var_exp[0]:.1f}%)")
plt.ylabel(f"Componente Principal 2 ({var_exp[1]:.1f}%)")
plt.grid(True, linestyle="--", alpha=0.5)

#Guardar imaagen
os.makedirs(os.path.dirname(snakemake.output.grafico_pca_inicial), exist_ok=True)
plt.savefig(snakemake.output.grafico_pca_inicial, dpi=300, bbox_inches="tight")
plt.close()

print(f"Grafico guardado en: {snakemake.output.grafico_pca_inicial}")