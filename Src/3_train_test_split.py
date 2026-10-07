

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Soporte para entorno Snakemake y Standalone
if 'snakemake' not in locals():
    class MockSub:
        pass
    class Mock:
        def __init__(self):
            self.input = MockSub()
            self.input.matriz_limpia = r"Data/Processed/datos_estriado_limpios.tsv"
            
            self.output = MockSub()
            self.output.train_completos = r"Data/Processed/datos_train_completos.tsv"
            self.output.test_completos = r"Data/Processed/datos_test_completos.tsv"
            self.output.tabla_csv = r"Results/Tablas/Tabla_Caracterizacion_Split.csv"
            self.output.tabla_md = r"Results/Tablas/Tabla_Caracterizacion_Split.md"
    snakemake = Mock()

# Cargar la matriz limpia
datos = pd.read_csv(
    snakemake.input.matriz_limpia, 
    sep="\t", 
    index_col=0
)

# Definir variables y agrupación
X = datos.drop(columns=['Health_Label'], errors="ignore")
y = datos["Health_Label"]

# Garantizar agrupación por Donor_ID
if 'Donor_ID' in datos.columns:
    groups = datos['Donor_ID']
else:
    groups = datos.index

# Configuración y ejecución de la división agrupada (80/20)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

# Generar DataFrames conservando conteos y metadatos clínicos
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

# Guardar archivos de Train y Test
out_train = snakemake.output.train_completos
out_test = snakemake.output.test_completos

os.makedirs(os.path.dirname(out_train), exist_ok=True)
os.makedirs(os.path.dirname(out_test), exist_ok=True)

train_df.to_csv(out_train, sep="\t")
test_df.to_csv(out_test, sep="\t")

print(f"Archivos de Train y Test guardados correctamente.")


# --- FUNCIÓN GENERADORA DE TABLAS ---
def generar_tabla_split(df_total, df_train, df_test, out_csv, out_md):
    """
    Calcula y guarda una tabla comparativa de covariables entre Total, Train y Test.
    """
    def obtener_resumen_grupo(df):
        resumen = {}
        n_total = len(df)
        resumen["Muestras Totales, n (%)"] = f"{n_total} (100%)"
        
        # Donantes Únicos
        if "Donor_ID" in df.columns:
            resumen["Donantes Únicos, n"] = f"{df['Donor_ID'].nunique()}"
        else:
            resumen["Donantes Únicos, n"] = "N/A"
            
        # Diagnóstico (Health_Label)
        if "Health_Label" in df.columns:
            counts = df["Health_Label"].value_counts()
            for label in ["Control", "Parkinson"]:
                cnt = counts.get(label, 0)
                pct = (cnt / n_total) * 100 if n_total > 0 else 0
                resumen[f"Diagnóstico: {label}, n (%)"] = f"{cnt} ({pct:.1f}%)"
                
        # Género
        if "Gender" in df.columns:
            m = (df["Gender"].astype(str).str.upper().str.startswith("M")).sum()
            f = (df["Gender"].astype(str).str.upper().str.startswith("F")).sum()
            resumen["Género (Masculino / Femenino)"] = f"{m} / {f}"
            
        # Variables continuas (Media ± DE)
        for var in ["Age", "RIN", "PMI"]:
            if var in df.columns:
                val = pd.to_numeric(df[var], errors="coerce").dropna()
                if len(val) > 0:
                    resumen[f"{var} (Media ± DE)"] = f"{val.mean():.1f} ± {val.std():.1f}"
                else:
                    resumen[f"{var} (Media ± DE)"] = "N/A"
                    
        return resumen

    # Calcular resúmenes
    dict_total = obtener_resumen_grupo(df_total)
    dict_train = obtener_resumen_grupo(df_train)
    dict_test = obtener_resumen_grupo(df_test)
    
    # Construir DataFrame final
    tabla = pd.DataFrame({
        "Variable / Característica": dict_total.keys(),
        "Conjunto Total": dict_total.values(),
        "Entrenamiento (Train)": dict_train.values(),
        "Evaluación (Test)": dict_test.values()
    })
    
    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    
    # 1. Guardar CSV
    tabla.to_csv(out_csv, index=False)
    
    # 2. Guardar Markdown físicamente en disco
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(tabla.to_markdown(index=False))
    
    print(f"[OK] Tablas de caracterización creadas correctamente:\n - {out_csv}\n - {out_md}")
    return tabla


# --- EJECUCIÓN CON LAS RUTAS DE SNAKEMAKE ---
out_csv = snakemake.output.tabla_csv
out_md = snakemake.output.tabla_md

generar_tabla_split(datos, train_df, test_df, out_csv, out_md)