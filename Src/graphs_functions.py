
#carga
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

#Ruta datos_train_completos.tsv
ruta= r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Data\Processed\train_top100_genes.tsv"
#Importamos train data
datos_train= pd.read_csv(ruta, sep="\t", index_col=0)
#Eliminamos Health_Label
train_no_Label= datos_train.drop(columns=["Health_Label"])

#Función pca scatter_plot
def pca_plot(datos_reducido, ruta_salida, title="PCA Analysis"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=datos_reducido,
        x= "PC1",
        y= "PC2",
        hue= "Label"
    )
    plt.title(title)
    plt.xlabel("Componente principal 1 (PC1)")
    plt.ylabel("Componente principal 2 (PC2)")
    
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    
    
#Función Volcano plot
def gen_Volcano_plot(df_resultados, sub_df_ctrl, sub_df_pd, ruta_salida):
    #Calculamos la media de expresión de cada grupo
    media_df_CTRL=sub_df_ctrl.mean()
    media_df_PD=sub_df_pd.mean()

    #Cálculo de Log2FC eje x volcano plot
    fold_change_div= media_df_PD.div(media_df_CTRL)
    fold_change= np.log2(fold_change_div) 
    
    #Copia de df_resultados para el gráfico
    df_volcano = df_resultados.copy()
    
    #Cálculo del eje Y (-log10 del p_value) dentro de df_resultados
    df_volcano["minus_log10_p"] = -np.log10(df_volcano["p_value"])
    
    #Pasar "Gene" al índice para que Pandas alinee los datos sólidamente
    df_volcano = df_volcano.set_index("Gene")

    #Añadir la columna de Fold Change al dataframe mapeando por índice
    df_volcano["log2FC"] = fold_change

    # El dibujo con Seaborn (sns.scatterplot)
    sns.scatterplot(
    data= df_volcano,
    x= "log2FC",
    y= "minus_log10_p",
    )
    plt.title("Volcano Plot - Expresión Diferencial de Genes")
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-Log10 p-value")
    plt.show()
    
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    
#Función Heatmap plot
def heatmap_train(train_no_Label, datos_train, ruta_salida):
    train_T= train_no_Label.T
    colores_dict = {"Control": "blue", "Parkinson": "orange"}
    colores_pacientes = datos_train["Health_Label"].map(colores_dict)
    sns.clustermap(
        train_T,
        z_score=0,
        col_colors=colores_pacientes,
        cmap="vlag",
        figsize=(10, 8) 
    )
    
    
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()
    
ruta_salida=r"C:\Users\Propietario\Documents\TFM_BIOINFORMATICA\Results\Imagenes\heatmap1"
heatmap_train(train_no_Label, datos_train, ruta_salida)