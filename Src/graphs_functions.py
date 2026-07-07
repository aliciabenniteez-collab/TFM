
#carga
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

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
    
    
#Benjamini-Hochberg (FDR)
#df_volcano["p_value"]
#reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    
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
    
    #Ajuste Benjamini-Hochberg 
    reject, p_adjusted, _, _ = multipletests(df_volcano["p_value"], alpha=0.05, method='fdr_bh')
    df_volcano["q_value"] = p_adjusted
    df_volcano["minus_log10_p"] = -np.log10(df_volcano["q_value"])
    
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
    
    #Asignamos los colores del gráfico
    rojos = df_volcano[(df_volcano["q_value"] < 0.05) & (df_volcano["log2FC"] > 0.5)]
    azules = df_volcano[(df_volcano["q_value"] < 0.05) & (df_volcano["log2FC"] < -0.5)]
    grises = df_volcano[~df_volcano.index.isin(rojos.index) & ~df_volcano.index.isin(azules.index)]
    #Pintamos
    plt.scatter(grises["log2FC"], grises["minus_log10_p"], color="#d3d3d3", s=15, alpha=0.5, label="No signif.")
    plt.scatter(azules["log2FC"], azules["minus_log10_p"], color="#1c00ff", s=25, alpha=0.8, label="Downregulated")
    plt.scatter(rojos["log2FC"], rojos["minus_log10_p"], color="#ff001c", s=25, alpha=0.8, label="Upregulated")
    
    plt.title("Volcano Plot - Expresión Diferencial de Genes")
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-Log10 q-value (FDR)")
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