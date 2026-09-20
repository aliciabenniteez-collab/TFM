import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Mock para Snakemake
if 'snakemake' not in locals():
    class Mock:
        def __init__(self):
            class SubMock: pass
            self.input = SubMock()
            self.input.train_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
            self.input.test_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\test_top100_genes.tsv"
            
            self.output = SubMock()
            self.output.shap_barras = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\shap_importancia_barras.png"
            self.output.shap_beeswarm = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\shap_impacto_beeswarm.png"
    snakemake = Mock()

# 1. Importamos los datos
datos_entrenamiento = pd.read_csv(snakemake.input.train_top100, sep='\t', index_col=0)
datos_test = pd.read_csv(snakemake.input.test_top100, sep='\t', index_col=0)

# 2. Garantizamos solo columnas numéricas para X
y_train = datos_entrenamiento["Health_Label"]
X_train_dea = datos_entrenamiento.select_dtypes(include=[np.number]).copy()

y_test = datos_test["Health_Label"]
X_test = datos_test.select_dtypes(include=[np.number]).copy()

# 3. Entrenamiento del Pipeline SVM-RBF
mi_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
])
mi_pipeline.fit(X_train_dea, y_train)

# 4. Cálculo de SHAP mediante KernelExplainer
# Resumimos el entrenamiento con k-means
background_summary = shap.kmeans(X_train_dea, k=10)

# Función de predicción hacia la probabilidad de Parkinson (Clase 1)
f_predict_parkinson = lambda x: mi_pipeline.predict_proba(x)[:, 1]

explainer = shap.KernelExplainer(f_predict_parkinson, background_summary)
value_shap = explainer.shap_values(X_test)

# 5. Gráfico 1: Importancia de características (Barras)
plt.figure(figsize=(10, 6))
shap.summary_plot(value_shap, X_test, plot_type="bar", show=False, max_display=15)
plt.title("Importancia Global de los Genes Biomarcadores (SVM-RBF)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(snakemake.output.shap_barras, dpi=300, bbox_inches='tight')
plt.close()
plt.clf()

# 6. Gráfico 2: Impacto Biológico (Beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(value_shap, X_test, show=False, max_display=15)
plt.title("Impacto de la Expresión Génica en el Diagnóstico de Parkinson (SHAP Beeswarm)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(snakemake.output.shap_beeswarm, dpi=300, bbox_inches='tight')
plt.close()