
#Cargar Paquetes:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score


#Importamos los datos
datos_entrenamiento = pd.read_csv(r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv", sep='\t', index_col=0)

#Separamos los datos
X_train_dea= datos_entrenamiento.drop(columns=["Health_Label"])
y_train= datos_entrenamiento["Health_Label"]

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


cv= StratifiedKFold(
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
    #genera los índices de los pacientes para cada ronda
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_dea, y_train)):
    
        #Separamos los genes (X) y el diagnóstico (y) para entrenar en esta ronda
        X_fold_train = X_train_dea.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
    
        #Separamos los genes (X) y el diagnóstico (y) para el examen de esta ronda
        X_fold_val = X_train_dea.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        #Entrenar y calcular probabilidades de cada modelo
        mi_pipeline.fit(X_fold_train, y_fold_train)
        y_prob_val = mi_pipeline.predict_proba(X_fold_val)[:, 1]
        
        #Añadimos a las listas los resultados
        lista_modelo.extend(y_fold_val)
        lista_probabilidades.extend(y_prob_val)
        
    resultado_modelos[nombre_modelo]=lista_modelo
    probabilidades_modelos[nombre_modelo]= lista_probabilidades
    
#Curvas ROC
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
for nombre_modelo, reales in resultado_modelos.items():
    probabilidades = probabilidades_modelos[nombre_modelo]
    
    #ROC
    fpr, tpr, thresholds = roc_curve(reales, probabilidades, pos_label="Parkinson")
    
    #Precision_recall_curve
    precision, recall, _ = precision_recall_curve(reales, probabilidades, pos_label='Parkinson')
    
    #AUC
    auc= roc_auc_score(reales, probabilidades)
    
    #AP (Average Precision)
    ap = average_precision_score(reales, probabilidades, pos_label='Parkinson')
    
    #F1
    probabilidades_modelo = probabilidades_modelos[nombre_modelo] 
    # Las convertimos en etiquetas de texto sobre la marcha usando el umbral de 0.5
    predicciones_binarias = ['Parkinson' if p >= 0.5 else 'Control' for p in probabilidades_modelo]
    f1 = f1_score(reales, predicciones_binarias, pos_label='Parkinson')
    
    #Mostramos los nº por pantalla
    print(f"{nombre_modelo} -> F1-Score: {f1:.2f} | AUC: {auc:.2f} | AP: {ap:.2f}")
    
    #Gráficas
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

# Ajusta el espacio para que no se solapen los títulos
plt.tight_layout()

plt.show()
