#Separamos el total de datos ya normalizados en datos de Train y datos de Test:
#datos_estriado_limpios.tsv

#Cargar lbrerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ruta_archivo_tsv= r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\datos_estriado_limpios.tsv"

#Cargar archivo:
datos= pd.read_csv(ruta_archivo_tsv, sep="\t", index_col=0)

#Definimos las variables x e y.
x= datos.drop(columns=["Health_Label"])

y= datos["Health_Label"]

#Dividir datos_estriados_limpios.tsv
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y) 

#Mostramos en pantalla las dimensiones de los grupos de datos

print(f"La dimensión de tu matriz de entrenamiento es {X_train.shape}")
print(f"La dimensión de tu matriz de test es {X_test.shape}")


train=pd.concat([X_train, Y_train], axis=1)
test=pd.concat([X_test, Y_test], axis=1) 


#Exportamos la matriz a la carpeta data/processed.
train.to_csv("data/processed/datos_train_completos.tsv", sep="\t")
test.to_csv("data/processed/datos_test_completos.tsv", sep="\t")
