
#Cargar Paquetes:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold  
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score

if 'snakemake' not in locals():
    class Mock:
        def __init__(self):
            class SubMock: pass
            self.input = SubMock()
            self.input.train_top100 = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
            self.output = SubMock()
            self.output.grafico_curvas = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\curvas_rendimiento_ml.png"
    snakemake = Mock()
    
#Importamos los datos
datos_entrenamiento = pd.read_csv(snakemake.input.train_top100, sep='\t', index_col=0)

# Extraemos la columna de grupos (Donor_ID) de forma segura
if "Donor_ID" in datos_entrenamiento.columns:
    groups_train = datos_entrenamiento["Donor_ID"]
else:
    # Si la columna tiene otro nombre (ej. "Donor"), seleccionamos la columna no numérica distinta de Health_Label
    cols_no_num = datos_entrenamiento.select_dtypes(include=['object', 'category']).columns
    cols_donante = [c for c in cols_no_num if c != "Health_Label"]
    groups_train = datos_entrenamiento[cols_donante[0]] if cols_donante else datos_entrenamiento.index

# Separamos X (solo genes/columnas numéricas) y y (etiquetas)
y_train = datos_entrenamiento["Health_Label"]

# Nos aseguramos de que X contenga ÚNICAMENTE variables numéricas (los genes)
X_train_dea = datos_entrenamiento.select_dtypes(include=[np.number]).copy()

#Creamos un diccionario con los modelos de ML a probar
modelos = {
    'Regresión Logística': LogisticRegression(l1_ratio=0, penalty='elasticnet', solver='saga', random_state=42, max_iter=10000),
    'SVM Lineal': SVC(kernel='linear', probability=True, random_state=42),
    'SVM RBF': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Lasso': LogisticRegression(l1_ratio=1, penalty='elasticnet', solver='saga', random_state=42, max_iter=10000)
}

#para guardar los datos en condiciones
resultado_modelos ={}
probabilidades_modelos= {}


cv = StratifiedGroupKFold(
    n_splits=5, 
    shuffle=True, 
    random_state=42
)

for nombre_modelo, algoritmo in modelos.items():
#Limpia/Escala los datos
    mi_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', algoritmo) 
    ])
   
    lista_modelo=[]
    lista_probabilidades= []
    
    #genera los índices de los pacientes para cada ronda. Pasamos groups=groups_train en el split
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_dea, y_train, groups=groups_train)):
    
        X_fold_train = X_train_dea.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
    
        X_fold_val = X_train_dea.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        mi_pipeline.fit(X_fold_train, y_fold_train)
        
        # Obtenemos la probabilidad de la clase positiva "Parkinson"
        clases = list(mi_pipeline.classes_)
        idx_parkinson = clases.index('Parkinson')
        y_prob_val = mi_pipeline.predict_proba(X_fold_val)[:, idx_parkinson]
        
        lista_modelo.extend(y_fold_val)
        lista_probabilidades.extend(y_prob_val)
        
    resultado_modelos[nombre_modelo] = lista_modelo
    probabilidades_modelos[nombre_modelo] = lista_probabilidades
    
#Curvas ROC
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
for nombre_modelo, reales in resultado_modelos.items():
    probabilidades = probabilidades_modelos[nombre_modelo]
    
    # <--- CORREGIDO: Convertimos la clase real a booleano/binario para roc_auc_score
    reales_binario = [1 if r == "Parkinson" else 0 for r in reales]
    
    # ROC
    fpr, tpr, thresholds = roc_curve(reales, probabilidades, pos_label="Parkinson")
    
    # Precision_recall_curve
    precision, recall, _ = precision_recall_curve(reales, probabilidades, pos_label='Parkinson')
    
    # AUC
    auc = roc_auc_score(reales_binario, probabilidades)
    
    # AP (Average Precision)
    ap = average_precision_score(reales_binario, probabilidades)
    
    # F1
    predicciones_binarias = ['Parkinson' if p >= 0.5 else 'Control' for p in probabilidades]
    f1 = f1_score(reales, predicciones_binarias, pos_label='Parkinson')
    
    # Mostramos los números por pantalla
    print(f"{nombre_modelo} -> F1-Score: {f1:.2f} | AUC: {auc:.2f} | AP: {ap:.2f}")
    
    # Gráficas
    ax1.plot(fpr, tpr, label=f"{nombre_modelo} (AUC = {auc:.2f})")
    ax2.plot(recall, precision, label=f"{nombre_modelo} (AP = {ap:.2f})")
    
#ROC (ax1)
ax1.plot([0, 1], [0, 1], color="black", linestyle="--")
ax1.set_xlabel("Tasa de Falsos Positivos (FPR)")
ax1.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
ax1.set_title("Curvas ROC")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

#PR (ax2) ---
ax2.set_xlabel("Recall (Sensibilidad)")
ax2.set_ylabel("Precision (Valor Predictivo Positivo)")
ax2.set_title("Curvas Precision-Recall")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)
plt.tight_layout()


#Exportamos las gráficas
plt.savefig(snakemake.output.grafico_curvas, dpi=300, bbox_inches="tight")