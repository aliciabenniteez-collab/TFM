# Importar paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, 
    average_precision_score, f1_score, confusion_matrix, classification_report
)

if 'snakemake' not in locals():
    class Mock:
        def __init__(self):
            class SubMock: pass
            self.input = SubMock()
            self.input.train_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
            self.input.test_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\test_top100_genes.tsv"
            
            self.output = SubMock()
            self.output.matriz_confusion = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\matriz_confusion_test.csv"
            self.output.reporte_txt = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\reporte_clasificacion_test.txt"
            self.output.grafico_test = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\figura_curvas_test.png"
    snakemake = Mock()

# 1. Importamos los datos
datos_entrenamiento = pd.read_csv(snakemake.input.train_top100, sep='\t', index_col=0)
datos_test = pd.read_csv(snakemake.input.test_top100, sep='\t', index_col=0)

# 2. Separación limpia (Garantizamos solo columnas numéricas para X)
y_train = datos_entrenamiento["Health_Label"]
X_train_dea = datos_entrenamiento.select_dtypes(include=[np.number]).copy()

y_test = datos_test["Health_Label"]
X_test = datos_test.select_dtypes(include=[np.number]).copy()

# 3. Definimos los pipelines de los dos modelos seleccionados
modelos = {
    'Lasso (L1)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42))
    ]),
    'SVM RBF': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ])
}

# Configuración de figura
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
colores = {'Lasso (L1)': 'tab:purple', 'SVM RBF': 'tab:orange'}
reporte_completo = ""

# 4. Evaluación de cada modelo
for nombre, pipeline in modelos.items():
    pipeline.fit(X_train_dea, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Matriz de Confusión y Reporte
    matriz_confusion = confusion_matrix(y_test, y_pred)
    matriz_df = pd.DataFrame(
        matriz_confusion, 
        index=['Real_Control', 'Real_Parkinson'], 
        columns=['Predicho_Control', 'Predicho_Parkinson']
    )
    
    # Guardamos la matriz del primer modelo o combinada
    if nombre == 'Lasso (L1)':
        matriz_df.to_csv(snakemake.output.matriz_confusion)

    reporte = classification_report(y_test, y_pred)
    reporte_completo += f"=== {nombre} ===\n{reporte}\n\n"
    print(f"--- {nombre} ---")
    print(matriz_confusion)
    print(reporte)

    # Bootstrap para Intervalo de Confianza del AUC
    n_bootstraps = 1000
    bootstrapped_scores = []
    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        indices_boot = rng.choice(len(y_test), size=len(y_test), replace=True)
        X_test_boot = X_test.iloc[indices_boot]
        y_test_boot = y_test.iloc[indices_boot]
        
        if len(np.unique(y_test_boot)) < 2:
            continue
        
        probabilidades_boot = pipeline.predict_proba(X_test_boot)[:, 1]
        bootstrapped_scores.append(roc_auc_score(y_test_boot, probabilidades_boot))
        
    auc_arrays = np.array(bootstrapped_scores)
    ic_inferior = np.percentile(auc_arrays, 2.5)
    ic_superior = np.percentile(auc_arrays, 97.5)

    # Curvas ROC y PR
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label="Parkinson")
    precision, recall, _ = precision_recall_curve(y_test, y_prob, pos_label="Parkinson")
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob, pos_label="Parkinson")

    # Plot ROC
    ax1.plot(fpr, tpr, color=colores[nombre], label=f"{nombre} (AUC = {auc:.2f})")
    
    # Sombra del IC del 95% basada en el Bootstrap real
    error_inferior = np.clip(tpr - (auc - ic_inferior), 0, 1)
    error_superior = np.clip(tpr + (ic_superior - auc), 0, 1)
    ax1.fill_between(fpr, error_inferior, error_superior, color=colores[nombre], alpha=0.15,
                     label=f"{nombre} 95% IC [{ic_inferior:.2f} - {ic_superior:.2f}]")

    # Plot PR
    ax2.plot(recall, precision, color=colores[nombre], label=f"{nombre} (AP = {ap:.2f})")

# Guardar reporte escrito
with open(snakemake.output.reporte_txt, "w") as archivo:
    archivo.write(reporte_completo)

# Formato visual final
ax1.plot([0, 1], [0, 1], color="black", linestyle="--")
ax1.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax1.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
ax1.set_title("Curvas ROC (Test Set)")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Recall (Sensibilidad)")
ax2.set_ylabel("Precision (Valor Predictivo Positivo)")
ax2.set_title("Curvas Precision-Recall (Test Set)")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(snakemake.output.grafico_test, dpi=300, bbox_inches='tight')

