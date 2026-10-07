#Librerías
import pandas as pd
import numpy as np
import os

# Configuración del entorno Snakemake / Standalone
if 'snakemake' not in locals():
    class MockInput:
        def __init__(self):
            self.raw_counts = (
                "Data/Raw/GSE205450_counts.table.txt.gz"
            )
            self.soft_file = (
                "Data/Raw/GSE205450_family.soft/GSE205450_family.soft"
            )
        def __getitem__(self, item):
            return self.raw_counts if item == 0 else self.soft_file
        def __len__(self):
            return 2

    class MockOutput:
        def __init__(self):
            self.matriz_limpia = (
                "Data/Processed/datos_estriado_limpios.tsv"
            )

    class Mock:
        def __init__(self):
            self.input = MockInput()
            self.output = MockOutput()

    snakemake = Mock()

def extraer_metadatos_soft(ruta_soft):
    metadata = []
    
    with open(ruta_soft, "r", encoding="utf-8") as f:
        bloque_actual = None
        for linea in f:
            linea_str = linea.strip()
            
            if linea_str.startswith("^SAMPLE ="):
                if bloque_actual:
                    metadata.append(bloque_actual)
                gsm = linea_str.split(" = ", 1)[1]
                bloque_actual = {"GSM": gsm}
                
            elif bloque_actual is not None:
                if linea_str.startswith("!Sample_title"):
                    bloque_actual["Sample_ID"] = (
                        linea_str.split(" = ", 1)[1]
                    )
                elif linea_str.startswith("!Sample_characteristics_ch1"):
                    caract = linea_str.split(" = ", 1)[1]
                    if ": " in caract:
                        var, val = caract.rsplit(": ", 1)
                        bloque_actual[var] = val

        if bloque_actual:
            metadata.append(bloque_actual)

    df_meta = pd.DataFrame(metadata)
    
    # Mapeo de columnas a nombres estandarizados
    column_mapping = {
        "patient": "Donor_ID",
        "disease": "Disease",
        "region": "Region",
        "age at death": "Age",
        "post mortem index (pmi)": "PMI",
        "gender (m: male, f: female)": "Gender",
        "rna integrity number (rin)": "RIN"
    }
    df_meta = df_meta.rename(columns=column_mapping)
    
    # Correcciones especificas del dataset
    if "Disease" in df_meta.columns and "GSM" in df_meta.columns:
        df_meta.loc[df_meta["GSM"] == "GSM6212998", "Disease"] = "PD"
    
    # Estandarizar etiqueta principal Health_Label
    df_meta["Health_Label"] = np.where(
        df_meta["Disease"].astype(str).str.upper().str.contains("CTRL|CONTROL"),
        "Control",
        "Parkinson"
    )
    
    # Conversión a variables numericas
    cols_num = ["Age", "PMI", "RIN"]
    for col in cols_num:
        if col in df_meta.columns:
            df_meta[col] = pd.to_numeric(df_meta[col], errors="coerce")
            
    return df_meta


def cargar_y_fusionar_datos(ruta_counts, df_meta):
    counts = pd.read_csv(ruta_counts, sep='\t', index_col=0)
    counts_T = counts.T
    
    # Limpiar ceros absolutos y asegurar enteros brutos
    activos = counts_T.sum(axis=0) > 0
    counts_enteros = counts_T.loc[:, activos].round().astype(int)
    
    # Fusion por index (Sample_ID)
    matriz_final = counts_enteros.merge(
        df_meta,
        left_index=True,
        right_on="Sample_ID",
        how="inner"
    )
    matriz_final = matriz_final.set_index("Sample_ID")
    activos = counts_T.sum(axis=0) > 0

    genes_totales = counts_T.shape[1]
    genes_retenidos = activos.sum()
    genes_filtrados = genes_totales - genes_retenidos
    print(f"[INFO] Muestras procesadas: {matriz_final.shape[0]}")
    print(f"[INFO] Genes retenidos: {counts_enteros.shape[1]}")
    print(f"[INFO] Genes totales antes del filtro: {genes_totales}")
    print(f"[INFO] Genes retenidos: {genes_retenidos}")
    print(f"[INFO] Genes filtrados: {genes_filtrados}")
    print(f"[INFO] Porcentaje filtrado: {genes_filtrados / genes_totales * 100:.2f}%")
    return matriz_final
    
    

    


if __name__ == "__main__":
    ruta_counts = snakemake.input[0]
    ruta_soft = (
        snakemake.input[1] 
        if len(snakemake.input) > 1 
        else "Data/Raw/GSE205450_family.soft/GSE205450_family.soft"
    )
    ruta_out = snakemake.output.matriz_limpia
    
    os.makedirs(os.path.dirname(ruta_out), exist_ok=True)
    
    df_metadata = extraer_metadatos_soft(ruta_soft)
    df_completo = cargar_y_fusionar_datos(ruta_counts, df_metadata)
    
    df_completo.to_csv(ruta_out, sep="\t")
    print(df_completo.head(10))
    print(f"Matriz guardada en: {ruta_out}")
    
    
