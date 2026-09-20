

#Cargar lbrerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
import os


# Soporte para entorno Snakemake y Standalone
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
            self.output.train_completos = (
                r"Data/Processed/datos_train_completos.tsv"
            )
            self.output.test_completos = (
                r"Data/Processed/datos_test_completos.tsv"
            )
    snakemake = Mock()

#Cargar la matriz
datos = pd.read_csv(
    snakemake.input.matriz_limpia, 
    sep="\t", 
    index_col=0
)

#Definir variables y agrupacion
X = datos.drop(columns=['Health_Label'], errors="ignore")
y = datos["Health_Label"]

# Garantizar agrupacion por Donor_ID
if 'Donor_ID' in datos.columns:
    groups = datos['Donor_ID']
else:
    groups = datos.index

#Configuracion y ejecucion de la division agrupada (80/20)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

#Generar DataFrames conservando conteos y metadatos clinicos
train_df = datos.iloc[train_idx]
test_df = datos.iloc[test_idx]

print(f"[INFO] Train shape: {train_df.shape}")
print(f"[INFO] Test shape:  {test_df.shape}")

# Verificar ausencia de Data Leakage entre donantes
if 'Donor_ID' in datos.columns:
    donores_train = set(train_df['Donor_ID'])
    donores_test = set(test_df['Donor_ID'])
    solapamiento = donores_train.intersection(donores_test)
    print(f"[INFO] Donantes unicos en Train: {len(donores_train)}")
    print(f"[INFO] Donantes unicos en Test:  {len(donores_test)}")
    print(f"[CHECK] Solapamiento de donantes: {len(solapamiento)}")

#Guardar archivos de salida respetando nombres de Snakemake
out_train = snakemake.output.train_completos
out_test = snakemake.output.test_completos

os.makedirs(os.path.dirname(out_train), exist_ok=True)
os.makedirs(os.path.dirname(out_test), exist_ok=True)

train_df.to_csv(out_train, sep="\t")
test_df.to_csv(out_test, sep="\t")

print(f"Archivos de Train y Test guardados correctamente.")








 


