import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC

# Mock para Snakemake / Standalone
if 'snakemake' not in locals():
    class MockSub: pass
    class Mock:
        def __init__(self):
            self.input = MockSub()
            self.input.train_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
            self.input.test_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\test_top100_genes.tsv"
            
            self.output = MockSub()
            self.output.shap_barras = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\shap_importancia_barras.png"
            self.output.shap_beeswarm = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\shap_impacto_beeswarm.png"
            self.output.tabla_shap = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\tabla_top15_shap_genes.csv"
    snakemake = Mock()

print("[INFO] Cargando datos para el análisis SHAP aislando estrictamente la EXPRESIÓN GÉNICA...")

datos_entrenamiento = pd.read_csv(snakemake.input.train_top100, sep='\t', index_col=0)
datos_test = pd.read_csv(snakemake.input.test_top100, sep='\t', index_col=0)

# 1. EXCLUSIÓN MÁS ESTRICTA DE METADATOS Y VARIABLES CLÍNICAS
cols_exclude = [
    "Health_Label", "Donor_ID", "RIN", "PMI", "Age", "Gender", "Region", "Disease",
    "disease duration", "age at disease onset", "disease_duration", "age_at_disease_onset"
]

# Filtrar cualquier columna que empiece por 'Unnamed' o esté en la lista clínica
gene_cols_all = [
    c for c in datos_entrenamiento.columns 
    if c not in cols_exclude and not str(c).startswith("Unnamed")
]

X_train_full = datos_entrenamiento[gene_cols_all].apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = datos_entrenamiento["Health_Label"]

X_test_full = datos_test[gene_cols_all].apply(pd.to_numeric, errors='coerce').fillna(0)
y_test = datos_test["Health_Label"]

# 2. Selección de los 100 genes con más señal en train
selector = SelectKBest(score_func=f_classif, k=min(100, X_train_full.shape[1]))
selector.fit(X_train_full, y_train)

top_genes_mask = selector.get_support()
selected_gene_names = np.array(gene_cols_all)[top_genes_mask]

X_train = pd.DataFrame(selector.transform(X_train_full), columns=selected_gene_names, index=X_train_full.index)
X_test = pd.DataFrame(selector.transform(X_test_full), columns=selected_gene_names, index=X_test_full.index)

# 3. Entrenamiento del Pipeline SVM-RBF
mi_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
])
mi_pipeline.fit(X_train, y_train)

# 4. SHAP KernelExplainer
background = shap.sample(X_train, 15, random_state=42)

def predict_proba_parkinson(x):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=selected_gene_names)
    return mi_pipeline.predict_proba(x)[:, 1]

explainer = shap.KernelExplainer(predict_proba_parkinson, background)
value_shap = explainer.shap_values(X_test, nsamples=100)

if isinstance(value_shap, list):
    value_shap = value_shap[1]

# 5. Generar y guardar la TABLA COMPLETA de SHAP (Todos los genes / Top 100)
mean_shap_values = np.abs(value_shap).mean(axis=0)

# Crear DataFrame con el ranking completo
df_shap_ranking = pd.DataFrame(
    {"Gene_Symbol": selected_gene_names, "Mean_Absolute_SHAP": mean_shap_values}
).sort_values(by="Mean_Absolute_SHAP", ascending=False)

# Añadir columna de posición (Rank)
df_shap_ranking.insert(0, "Rank", range(1, len(df_shap_ranking) + 1))

# Guardar el CSV completo (sin el .head(15))
ruta_tabla = getattr(
    snakemake.output, "tabla_shap", "Results/tabla_completa_shap_genes.csv"
)
os.makedirs(os.path.dirname(ruta_tabla), exist_ok=True)

df_shap_ranking.to_csv(ruta_tabla, index=False)
print(
    f"Tabla con el ranking completo ({len(df_shap_ranking)} genes) guardada en: {ruta_tabla}"
)

# -------------------------------------------------------------------------
# Exportar la tabla completa a LaTeX (Formato multipágina con longtable)
# -------------------------------------------------------------------------
ruta_tex = ruta_tabla.replace(".csv", "_longtable.tex")

# Copia para formatear los nombres de columna en LaTeX
df_tex = df_shap_ranking.copy()
df_tex.columns = [
    "\\textbf{Rank}",
    "\\textbf{Gene Symbol}",
    "\\textbf{Mean |SHAP|}",
]

# Generar el código longtable de LaTeX
latex_longtable = df_tex.to_latex(
    index=False,
    float_format="%.5f",
    column_format="crr",
    longtable=True,  # Activa el soporte para múltiples páginas
    caption="Tabla Suplementaria S3: Ranking completo de importancia de genes según los valores medios absolutos de SHAP.",
    label="tab:shap_full_ranking",
    escape=False,
)

with open(ruta_tex, "w") as f:
    f.write(latex_longtable)

print(f"Archivo .tex multipágina guardado en: {ruta_tex}")


# 6. Exportar Gráficas Limpias
os.makedirs(os.path.dirname(snakemake.output.shap_barras), exist_ok=True)

plt.close('all')
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(value_shap, X_test, plot_type="bar", show=False, max_display=15)
plt.title("Importancia Transcriptómica Global (SVM-RBF)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(snakemake.output.shap_barras, dpi=300, bbox_inches='tight')
plt.close('all')

plt.close('all')
fig, ax = plt.subplots(figsize=(11, 7))
shap.summary_plot(value_shap, X_test, show=False, max_display=15)
plt.title("Impacto de la Expresión Génica en la Predicción (SHAP)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(snakemake.output.shap_beeswarm, dpi=300, bbox_inches='tight')
plt.close('all')