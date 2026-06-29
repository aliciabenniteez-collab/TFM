#Importar paquetes
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


#Ruta de los datos
ruta_actual = r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\datos_estriado_limpios.tsv"

#Cargar archivo:
archivo_cargado= pd.read_csv(ruta_actual, sep="\t", index_col=0)

#Hacemos dos dataframes, una para controles y otra para PD
sub_df_ctrl= archivo_cargado[archivo_cargado["Health_Label"] == "Control"]
sub_df_pd= archivo_cargado[archivo_cargado["Health_Label"] == "Parkinson"]

#Eliminamos la columna Health_Label para ambos dataframes
sub_df_ctrl = sub_df_ctrl.drop(columns=["Health_Label"])
sub_df_pd = sub_df_pd.drop(columns=["Health_Label"])

#Guardamos los nombres de los genes en una lista
lista_genes = sub_df_ctrl.columns.tolist()

#T-Test
#Lista vacía
resultados= []

#Bucle
for gen in lista_genes:
    estadistico, p_valor = ttest_ind(sub_df_ctrl[gen], sub_df_pd[gen])
    resultados.append({"Gene": gen, "p_value": p_valor})
    
#Convertir la lista resultados a un dataframe para luego poder filtrar
df_resultados= pd.DataFrame(resultados)


#Ordenar dataframe por sus p_valores
orden_pvalue=df_resultados.sort_values(by= "p_value", ascending=True) #Ascending true pq buscamos los menores p-values

#Filtramos las primeras 100 filas
filtro_pvalue = orden_pvalue.head(100)

#Extraemos los nombre de los 100 top genes
genes_top= filtro_pvalue["Gene"]
matriz_top100g= archivo_cargado[genes_top]

#Exportamos la nueva matriz filtrada
matriz_top100g.to_csv("data/processed/datos_top100_genes.tsv", sep="\t")

print(matriz_top100g.isna().any().any())