# Carga de librerías
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

#PCA
def pca_plot(datos_reducido, ruta_salida, title="PCA Analysis"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=datos_reducido,
        x="PC1",
        y="PC2",
        hue="Label"
    )
    plt.title(title)
    plt.xlabel("Componente principal 1 (PC1)")
    plt.ylabel("Componente principal 2 (PC2)")
    
    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()

# Volcano Plot
def gen_Volcano_plot(df_resultados, sub_df_ctrl, sub_df_pd, ruta_salida):
    
    media_ctrl = sub_df_ctrl.mean(axis=0)
    media_pd = sub_df_pd.mean(axis=0)

    # Evitar divisiones por cero
    media_ctrl = media_ctrl + 1e-6
    media_pd = media_pd + 1e-6

    fold_change = np.log2(media_pd / media_ctrl)

    df_volcano = df_resultados.copy()

    reject, p_adjusted, _, _ = multipletests(
        df_volcano["p_value"],
        alpha=0.05,
        method="fdr_bh"
    )

    df_volcano["q_value"] = p_adjusted

    df_volcano["minus_log10_p"] = (
        -np.log10(df_volcano["q_value"] + 1e-300)
    )

   

    df_volcano["log2FC"] = df_volcano["Gene"].map(fold_change)



    df_volcano_clean = (
        df_volcano
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["log2FC", "minus_log10_p"])
    )

    # --------------------------------------------------------
    # 6. Clasificación
    # --------------------------------------------------------

    fdr_threshold = 0.05
    log2fc_threshold = 0.5

    rojos = df_volcano_clean[
        (df_volcano_clean["q_value"] < fdr_threshold) &
        (df_volcano_clean["log2FC"] > log2fc_threshold)
    ]

    azules = df_volcano_clean[
        (df_volcano_clean["q_value"] < fdr_threshold) &
        (df_volcano_clean["log2FC"] < -log2fc_threshold)
    ]

    grises = df_volcano_clean[
        ~df_volcano_clean["Gene"].isin(rojos["Gene"]) &
        ~df_volcano_clean["Gene"].isin(azules["Gene"])
    ]

    # --------------------------------------------------------
    # 7. Crear gráfico
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        grises["log2FC"],
        grises["minus_log10_p"],
        color="#3A3A3A",
        s=15,
        alpha=0.25,
        edgecolor="none",
        label="No signif."
    )

    plt.scatter(
        azules["log2FC"],
        azules["minus_log10_p"],
        color="#112CC2",
        s=25,
        alpha=0.6,
        edgecolor="none",
        label="Downregulated"
    )

    plt.scatter(
        rojos["log2FC"],
        rojos["minus_log10_p"],
        color="#FF6B00",
        s=30,
        alpha=0.9,
        edgecolor="none",
        label="Upregulated"
    )

    # Líneas de referencia

    y_line_threshold = -np.log10(fdr_threshold)

    plt.axhline(
        y=y_line_threshold,
        color="#3A3A3A",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )

    plt.axvline(
        x=log2fc_threshold,
        color="#3A3A3A",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )

    plt.axvline(
        x=-log2fc_threshold,
        color="#3A3A3A",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )

    # Títulos

    plt.title(
        "Análisis de Expresión Diferencial (Volcano Plot)",
        fontsize=13,
        pad=15,
        fontweight="bold",
        color="#3A3A3A"
    )

    plt.xlabel(
        r"$\log_2$ Fold Change",
        fontsize=11
    )

    plt.ylabel(
        r"$-\log_{10}$ q-value (FDR)",
        fontsize=11
    )

    plt.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none"
    )

    plt.grid(
        True,
        linestyle=":",
        alpha=0.4,
        color="#A0A0A0"
    )

    # --------------------------------------------------------
    # 8. Guardar
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(ruta_salida),
        exist_ok=True
    )

    plt.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Heatmap
# ============================================================

def heatmap_train(train_no_Label, datos_train, ruta_salida):

    # --------------------------------------------------------
    # 1. Asegurar que todos los datos sean numéricos
    # --------------------------------------------------------

    train_no_Label = train_no_Label.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Sustituir infinitos por NaN
    train_no_Label = train_no_Label.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # 2. Eliminar genes completamente vacíos
    # --------------------------------------------------------

    train_no_Label = train_no_Label.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # 3. Eliminar genes con algún NaN
    # --------------------------------------------------------
    # Para el clustering necesitamos una matriz completa.

    train_no_Label = train_no_Label.dropna(
        axis=1,
        how="any"
    )

    # --------------------------------------------------------
    # 4. Eliminar genes sin variabilidad
    # --------------------------------------------------------
    # Si un gen tiene exactamente el mismo valor en todos
    # los pacientes, z_score=0 produce división por cero.

    var_genes = train_no_Label.var(axis=0)

    train_no_Label = train_no_Label.loc[
        :,
        var_genes > 0
    ]

    # --------------------------------------------------------
    # 5. Comprobar que quedan genes
    # --------------------------------------------------------

    if train_no_Label.shape[1] == 0:
        raise ValueError(
            "No quedan genes válidos para generar el heatmap. "
            "Todos contienen NaN/inf o tienen varianza cero."
        )

    # --------------------------------------------------------
    # 6. Genes en filas y pacientes en columnas
    # --------------------------------------------------------

    train_T = train_no_Label.T

    # --------------------------------------------------------
    # 7. Colores de los pacientes
    # --------------------------------------------------------

    colores_dict = {
        "Control": "#4D4D4D",
        "Parkinson": "#FF7400"
    }

    colores_pacientes = (
        datos_train.loc[train_no_Label.index, "Health_Label"]
        .map(colores_dict)
    )

    # --------------------------------------------------------
    # 8. Comprobación final de valores
    # --------------------------------------------------------

    if not np.isfinite(train_T.to_numpy()).all():
        raise ValueError(
            "La matriz del heatmap todavía contiene "
            "valores no finitos."
        )

    # --------------------------------------------------------
    # 9. Generar clustermap
    # --------------------------------------------------------

    g = sns.clustermap(
        train_T,
        z_score=0,
        col_colors=colores_pacientes,
        cmap="vlag",
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # 10. Guardar
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(ruta_salida),
        exist_ok=True
    )

    g.savefig(
        ruta_salida,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

