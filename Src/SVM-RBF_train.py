#Importar paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score, confusion_matrix, classification_report


#Importamos los datos
datos_entrenamiento = pd.read_csv(r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv", sep='\t', index_col=0)
datos_test=pd.read_csv(r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\test_top100_genes.tsv", sep='\t', index_col=0)

#Separamos los datos
#Entrenamiento
X_train_dea= datos_entrenamiento.drop(columns=["Health_Label"])
y_train= datos_entrenamiento["Health_Label"]
#Test
X_test=datos_test.drop(columns=["Health_Label"])
y_test=datos_test["Health_Label"]

#Definimos Pipeline
mi_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))])

mi_pipeline.fit(X_train_dea, y_train)

#Calculamos predicciones
y_pred = mi_pipeline.predict(X_test)
y_prob = mi_pipeline.predict_proba(X_test)[:, 1]

### Evaluamos resultado y sacamos la matriz de confusión ###

matriz_confusion= confusion_matrix(y_test, y_pred)
#La converitmos en DF
matriz_df = pd.DataFrame(
    matriz_confusion, 
    index=['Real_Control', 'Real_Parkinson'], 
    columns=['Predicho_Control', 'Predicho_Parkinson'])
    
matriz_df.to_csv(r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\matriz_confusion_test.csv")
reporte= classification_report(y_test, y_pred)

print(matriz_confusion)
print(reporte)

with open (r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\reporte_clasificacion_test.txt", "w") as archivo:
    archivo.write(reporte)

#Bootstraping
n_bootstraps = 1000
bootstrapped_scores = []

rng = np.random.RandomState(42)
for i in range(n_bootstraps):
    #Generamos los índices
    indices_boot = rng.choice(len(y_test), size=len(y_test), replace=True)
    
    X_test_boot = X_test.iloc[indices_boot]
    y_test_boot = y_test.iloc[indices_boot]
    
    if len(np.unique(y_test_boot)) < 2:
        continue
    
    probabilidades_boot=mi_pipeline.predict_proba(X_test_boot)[:,1]
    auc= roc_auc_score(y_test_boot, probabilidades_boot)
    
    bootstrapped_scores.append(auc)
    
auc_arrays = np.array(bootstrapped_scores)

# Calculamos los percentiles para el IC del 95%
ic_inferior = np.percentile(auc_arrays, 2.5)
ic_superior = np.percentile(auc_arrays, 97.5)
auc_mediano = np.percentile(auc_arrays, 50)
print(f"Rendimiento final del SVM-RBF en el Test Set:")
print(f"AUC Mediano: {auc_mediano:.2f}")
print(f"Intervalo de Confianza al 95%: [{ic_inferior:.2f} - {ic_superior:.2f}]")
    
