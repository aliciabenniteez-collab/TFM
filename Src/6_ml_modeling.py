# Cargar Paquetes:
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold  
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, 
    average_precision_score, f1_score, accuracy_score, recall_score
)

# -----------------------------------------------------------------------------
# 1. Compatibilidad con Snakemake y MOCK para ejecución manual
# -----------------------------------------------------------------------------
if 'snakemake' not in locals():
    class Mock:
        def __init__(self):
            class SubMock: pass
            self.input = SubMock()
            self.input.train_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
            self.output = SubMock()
            self.output.grafico_curvas = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\curvas_rendimiento_ml.png"
            self.output.tabla_metricas = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\tabla_resumen_metricas_ml.csv"
            self.output.tabla_genes_cv = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\tabla_estabilidad_genes_cv.csv"
    snakemake = Mock()

def ensure_dir(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Cargar e Identificar Datos
# -----------------------------------------------------------------------------
datos_entrenamiento = pd.read_csv(snakemake.input.train_top100, sep='\t', index_col=0)

cols_clinicas = ["Health_Label", "Donor_ID", "RIN", "PMI", "Age", "Gender", "Region", "Disease"]

# Extraemos la columna de grupos (Donor_ID)
if "Donor_ID" in datos_entrenamiento.columns:
    groups_train = datos_entrenamiento["Donor_ID"]
else:
    cols_no_num = datos_entrenamiento.select_dtypes(include=['object', 'category']).columns
    cols_donante = [c for c in cols_no_num if c != "Health_Label"]
    groups_train = datos_entrenamiento[cols_donante[0]] if cols_donante else datos_entrenamiento.index

y_train = datos_entrenamiento["Health_Label"]

# Nos aseguramos de seleccionar ÚNICAMENTE variables numéricas correspondientes a genes
genes_cols = [col for col in datos_entrenamiento.columns if col not in cols_clinicas]
X_train_dea = datos_entrenamiento[genes_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# -----------------------------------------------------------------------------
# 3. Diccionario con los 5 Modelos Especificados
# -----------------------------------------------------------------------------
k_features = min(100, X_train_dea.shape[1])

modelos = {
    'Regresión Logística': LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
    'Lasso': LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000, random_state=42),
    'SVM Lineal': SVC(kernel='linear', probability=True, random_state=42),
    'SVM RBF': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

resultado_modelos = {}
probabilidades_modelos = {}
metricas_resumen = []
gene_selection_counts = {gene: 0 for gene in genes_cols}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# -----------------------------------------------------------------------------
# 4. Evaluación en CV Agrupada (con Selección Anidada)
# -----------------------------------------------------------------------------
for nombre_modelo, algoritmo in modelos.items():
    print(f"[INFO] Evaluando modelo: {nombre_modelo}...")
    
    # Pipeline con escalado y selección anidada dentro del fold (Anti-leakage)
    mi_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=k_features)),
        ('classifier', algoritmo) 
    ])
   
    lista_modelo = []
    lista_probabilidades = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_dea, y_train, groups=groups_train)):
        X_fold_train = X_train_dea.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
    
        X_fold_val = X_train_dea.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        mi_pipeline.fit(X_fold_train, y_fold_train)
        
        # Registrar genes seleccionados en este fold
        selected_mask = mi_pipeline.named_steps['selector'].get_support()
        selected_genes = np.array(genes_cols)[selected_mask]
        for g in selected_genes:
            gene_selection_counts[g] += 1
        
        # Probabilidad de la clase positiva "Parkinson"
        clases = list(mi_pipeline.classes_)
        idx_parkinson = clases.index('Parkinson') if 'Parkinson' in clases else 1
        y_prob_val = mi_pipeline.predict_proba(X_fold_val)[:, idx_parkinson]
        
        lista_modelo.extend(y_fold_val)
        lista_probabilidades.extend(y_prob_val)
        
    resultado_modelos[nombre_modelo] = lista_modelo
    probabilidades_modelos[nombre_modelo] = lista_probabilidades

# -----------------------------------------------------------------------------
# 5. Generación de Gráficas (ROC + PR) y Guardado de Tablas
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for nombre_modelo, reales in resultado_modelos.items():
    probabilidades = probabilidades_modelos[nombre_modelo]
    reales_binario = [1 if str(r).lower() in ["parkinson", "pd", "enfermo", "case"] else 0 for r in reales]
    
    fpr, tpr, _ = roc_curve(reales_binario, probabilidades)
    precision, recall, _ = precision_recall_curve(reales_binario, probabilidades)
    
    auc = roc_auc_score(reales_binario, probabilidades)
    ap = average_precision_score(reales_binario, probabilidades)
    
    predicciones_binarias = [1 if p >= 0.5 else 0 for p in probabilidades]
    f1 = f1_score(reales_binario, predicciones_binarias)
    acc = accuracy_score(reales_binario, predicciones_binarias)
    sens = recall_score(reales_binario, predicciones_binarias)
    spec = recall_score(reales_binario, predicciones_binarias, pos_label=0)
    
    # Guardar métricas completas para la tabla resumen
    metricas_resumen.append({
        "Modelo": nombre_modelo,
        "Accuracy": acc,
        "Sensibilidad": sens,
        "Especificidad": spec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Average Precision": ap
    })
    
    print(f"{nombre_modelo} -> F1-Score: {f1:.3f} | AUC: {auc:.3f} | AP: {ap:.3f}")
    
    ax1.plot(fpr, tpr, label=f"{nombre_modelo} (AUC = {auc:.3f})")
    ax2.plot(recall, precision, label=f"{nombre_modelo} (AP = {ap:.3f})")

# Estilo Curva ROC (ax1)
ax1.plot([0, 1], [0, 1], color="black", linestyle="--")
ax1.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax1.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
ax1.set_title("Curvas ROC (5-Fold CV)")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Estilo Curva PR (ax2)
ax2.set_xlabel("Recall (Sensibilidad)")
ax2.set_ylabel("Precision (Valor Predictivo Positivo)")
ax2.set_title("Curvas Precision-Recall")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Exportar Gráfica de Curvas
ensure_dir(snakemake.output.grafico_curvas)
plt.savefig(snakemake.output.grafico_curvas, dpi=300, bbox_inches="tight")
plt.close()

# Exportar Tabla Resumen de Métricas
if hasattr(snakemake.output, 'tabla_metricas'):
    ensure_dir(snakemake.output.tabla_metricas)
    pd.DataFrame(metricas_resumen).to_csv(snakemake.output.tabla_metricas, index=False)

# Exportar Estabilidad de Genes Seleccionados
if hasattr(snakemake.output, 'tabla_genes_cv'):
    ensure_dir(snakemake.output.tabla_genes_cv)
    df_genes = pd.DataFrame(list(gene_selection_counts.items()), columns=["Gen", "Frecuencia_CV"])
    df_genes.sort_values(by="Frecuencia_CV", ascending=False).to_csv(snakemake.output.tabla_genes_cv, index=False)

print("[ÉXITO] Ejecución de ml_modeling completada.")