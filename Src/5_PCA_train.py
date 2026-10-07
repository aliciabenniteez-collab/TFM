# 1. Cargar Paquetes
import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from graphs_functions import pca_plot

# Soporte para entorno Snakemake / Standalone
if 'snakemake' not in locals():
    class MockSub:
        pass
    class Mock:
        def __init__(self):
            self.input = MockSub()
            self.input.train_genes = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
            self.output = MockSub()
            self.output.grafico_pca = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\pca_top100.png"
            self.output.tabla_loadings = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\tabla_top_genes_pca_loadings.csv"
    snakemake = Mock()

print("[INFO] Cargando datos completos de expresión para PCA...")

# 2. Cargar datos de entrenamiento
datos_train = pd.read_csv(
    snakemake.input.train_genes, 
    sep="\t", 
    index_col=0
)

# 3. Excluir variables clínicas para aislar ÚNICAMENTE la expresión génica
cols_clinical = ["Health_Label", "Donor_ID", "RIN", "PMI", "Age", "Gender", "Region", "Disease"]
gene_cols = [c for c in datos_train.columns if c not in cols_clinical]

# 4. Preparar matrices X e y
X_raw = datos_train[gene_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
y = datos_train["Health_Label"]

# Transformación logarítmica segura: solo si los valores son estrictamente no negativos
if (X_raw.values < 0).any():
    X_log = X_raw.copy()
else:
    X_log = np.log2(X_raw + 1)

# Ajustar K según la cantidad de genes disponibles
k_features = min(100, X_log.shape[1])

# 5. Pipeline con Selección de Características y PCA
pipeline_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('select_genes', SelectKBest(score_func=f_classif, k=k_features)),
    ('pca', PCA(n_components=2))
])

# Ajustar y proyectar a 2 componentes principales
df_pca = pipeline_pca.fit_transform(X_log, y)

# Extraer información del modelo PCA dentro del Pipeline
pca_model = pipeline_pca.named_steps['pca']
kbest_model = pipeline_pca.named_steps['select_genes']

var_exp = pca_model.explained_variance_ratio_ * 100
print(f"[INFO] Varianza explicada - PC1: {var_exp[0]:.2f}% | PC2: {var_exp[1]:.2f}%")

# 6. Construir DataFrame con componentes para el gráfico
df_graph = pd.DataFrame({
    "PC1": df_pca[:, 0],
    "PC2": df_pca[:, 1],
    "Label": y
})

# 7. Generar y exportar la gráfica de PCA
ruta_grafico = getattr(snakemake.output, 'grafico_pca', 'Results/Imagenes/pca_top100.png')
os.makedirs(os.path.dirname(ruta_grafico), exist_ok=True)

pca_plot(
    datos_reducido=df_graph,
    ruta_salida=ruta_grafico,
    title=f"PCA - Top 100 Genes (PC1: {var_exp[0]:.1f}% | PC2: {var_exp[1]:.1f}%)"
)
print(f"[ÉXITO] Gráfico PCA guardado en: {ruta_grafico}")

# 8. EXPORTACIÓN DE TABLA: Loadings (Pesos de los genes en PC1 y PC2)
genes_elegidos = np.array(gene_cols)[kbest_model.get_support()]
loadings = pca_model.components_

df_loadings = pd.DataFrame({
    'Gene_ID': genes_elegidos,
    'Peso_PC1': loadings[0],
    'Peso_PC2': loadings[1],
    'Valor_Absoluto_PC1': np.abs(loadings[0])
}).sort_values(by='Valor_Absoluto_PC1', ascending=False)

# Guardar la tabla completa de loadings ordenada por peso en PC1
ruta_tabla_loadings = getattr(snakemake.output, 'tabla_loadings', 'Results/tabla_top_genes_pca_loadings.csv')
os.makedirs(os.path.dirname(ruta_tabla_loadings), exist_ok=True)

# Guardamos los 15 genes con mayor carga explicativa
df_loadings.head(15).drop(columns=['Valor_Absoluto_PC1']).to_csv(ruta_tabla_loadings, index=False)
print(f"Tabla de Loadings de PCA guardada en: {ruta_tabla_loadings}")