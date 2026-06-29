
#Cargando paquetes:
import pandas as pd
import numpy as np
import os


#Ruta de los datos
ruta_actual = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Raw\GSE205450_counts.table.txt.gz"


#Función para Importar los datos
def cargar_datos(ruta_archivo):
    d_estriado = pd.read_csv(ruta_archivo, sep='\t', index_col=0)
    print(f"Se han cargado los datos correctamente desde la ruta {ruta_archivo}")
    #Transponemos la matriz
    d_estriado_T = d_estriado.T
    print(f"Las dimensiones de tu matriz son {d_estriado_T.shape}")
    print(d_estriado_T)
    return(d_estriado_T)


#Función para filtrar genes de baja expresión.
def filtrar_genes(matriz_T):
    #Definimos el umbral en 50% de las muestras
    umbral = matriz_T.shape[0] * 0.5
    #Definimos el filtro
    genes_a_mantener = (matriz_T >= 10).sum(axis=0) >= umbral
    df_filtrado = matriz_T.loc[:, genes_a_mantener]
    
    #Mostramos el resultado de aplicar el filtro
    print(f"Genes antes del filtro: {matriz_T.shape[1]}")
    print(f"Genes después del filtro: {df_filtrado.shape[1]}")
    return(df_filtrado) 


#Función para normalizar:
def normalizar(matriz_filtrar_genes):
    #Calculamos las lecturas totales
    lecturas_total = matriz_filtrar_genes.sum(axis=1) 
    #Calculamos CPM
    matriz_CPM = matriz_filtrar_genes.div(lecturas_total, axis=0) * 1e6
    #Calculamos log2 + 1
    matriz_log = np.log2(matriz_CPM + 1)
    return(matriz_log)

#Ejecución del código
if __name__ == "__main__":
    matriz_cargada = cargar_datos(ruta_actual)
    matriz_filtrar_genes = filtrar_genes(matriz_cargada)
    matriz_final = normalizar(matriz_filtrar_genes)

#Creamos una columna llamada Health_label para guardar si las muestras pertenecen a control o parkinson
matriz_final['Health_Label'] = np.where(matriz_final.index.str.contains("CTRL"), "Control", "Parkinson")

#Exportamos la matriz a la carpeta data/processed.
matriz_final.to_csv("data/processed/datos_estriado_limpios.tsv", sep="\t")




