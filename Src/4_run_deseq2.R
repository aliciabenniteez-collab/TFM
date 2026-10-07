
suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
  library(ggplot2)
  library(pheatmap)
  library(writexl)
})

# Función auxiliar para crear directorios automáticamente
save_file <- function(path) {
  if (!is.null(path) && path != "") {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  }
}

# 1. Definición global de columnas clínicas a excluir de los conteos
cols_clinical <- c(
  "Health_Label", "Donor_ID", "RIN", "PMI", "Age", "Gender",
  "Region", "Disease", "disease duration", "age at disease onset",
  "disease_duration", "age_at_disease_onset"
)

# 2. Gestionar entrada/salida para Snakemake / Standalone
if (exists("snakemake")) {
  in_c <- if (!is.null(snakemake@input[["counts"]])) snakemake@input[["counts"]] else snakemake@input[[1]]
  in_t <- if (!is.null(snakemake@input[["test"]])) snakemake@input[["test"]] else snakemake@input[[2]]
  out_tr  <- if (!is.null(snakemake@output[["train_top100"]])) snakemake@output[["train_top100"]] else snakemake@output[[1]]
  out_te  <- if (!is.null(snakemake@output[["test_top100"]])) snakemake@output[["test_top100"]] else snakemake@output[[2]]
  out_v   <- if (!is.null(snakemake@output[["volcano_deseq2"]])) snakemake@output[["volcano_deseq2"]] else snakemake@output[[3]]
  out_h   <- if (!is.null(snakemake@output[["heatmap_deseq2"]])) snakemake@output[["heatmap_deseq2"]] else snakemake@output[[4]]
  out_res <- if (!is.null(snakemake@output[["deseq2_results"]])) snakemake@output[["deseq2_results"]] else snakemake@output[[5]]
} else {
  in_c <- "Data/Processed/datos_train_completos.tsv"
  in_t <- "Data/Processed/datos_test_completos.tsv"
  out_tr <- "Data/Processed/train_top100_genes.tsv"
  out_te <- "Data/Processed/test_top100_genes.tsv"
  out_v <- "Results/Imagenes/volcano_deseq2.png"
  out_h <- "Results/Imagenes/heatmap_deseq2.png"
  out_res <- "Results/deseq2_results.csv"
}

# 3. Cargar datos
df_train <- read.table(
  in_c,
  header = TRUE,
  sep = "\t",
  row.names = 1,
  check.names = FALSE
)
df_test <- read.table(
  in_t,
  header = TRUE,
  sep = "\t",
  row.names = 1,
  check.names = FALSE
)

# 4. Extraer y preparar Metadatos
covs <- c("Health_Label", "Donor_ID", "RIN", "PMI", "Age", "Gender")
meta <- df_train[, colnames(df_train) %in% covs, drop = FALSE]
meta$Health_Label <- as.factor(meta$Health_Label)

if ("Gender" %in% colnames(meta)) {
  meta$Gender <- as.factor(meta$Gender)
}

# Imputación y escalado de variables continuas
for (col in intersect(c("RIN", "PMI", "Age"), colnames(meta))) {
  v <- as.numeric(meta[[col]])
  v[is.na(v)] <- median(v, na.rm = TRUE)
  meta[[col]] <- if (sd(v) > 0) scale(v) else v
}

# 5. Matriz de Conteos e Integración DESeq2
gene_cols <- setdiff(colnames(df_train), cols_clinical)
df_expr <- df_train[, gene_cols]

# Conversión a matriz numérica
counts_num <- data.matrix(df_expr)

# Transponer a formato (Genes x Muestras)
counts_raw <- t(counts_num)

# Limpieza de valores nulos o negativos
counts_raw[is.na(counts_raw)] <- 0
counts_raw[counts_raw < 0] <- 0
counts_clean <- round(counts_raw)

# Construir la fórmula del diseño
terms <- intersect(c("RIN", "PMI", "Age", "Gender"), colnames(meta))
design_formula <- as.formula(
  paste("~", paste(c(terms, "Health_Label"), collapse = " + "))
)

# Crear el objeto DESeq2
dds <- DESeqDataSetFromMatrix(
  countData = counts_clean,
  colData = meta,
  design = design_formula
)

dds <- dds[rowSums(counts(dds)) >= 10, ]
dds$Health_Label <- relevel(dds$Health_Label, ref = "Control")
dds <- DESeq(dds)

# 6. Extracción de Resultados y Exportación de Matrices
res <- results(dds, contrast = c("Health_Label", "Parkinson", "Control"))
res_df <- as.data.frame(res)

res_df$gene <- rownames(res_df)
save_file(out_res)
write.csv(res_df, out_res, row.names = FALSE)

genes_retenidos <- rownames(dds)
keep_cols <- c(genes_retenidos, "Health_Label", "Donor_ID")

save_file(out_tr)
write.table(
  df_train[, intersect(colnames(df_train), keep_cols)],
  out_tr,
  sep = "\t",
  quote = FALSE,
  col.names = NA
)

save_file(out_te)
write.table(
  df_test[, intersect(colnames(df_test), keep_cols)],
  out_te,
  sep = "\t",
  quote = FALSE,
  col.names = NA
)
message("Archivos de expresión exportados sin filtrado previo.")

# 7. Volcano Plot
res_df$Expresion <- "No significativo"

idx_up <- !is.na(res_df$padj) &
  res_df$padj < 0.05 &
  res_df$log2FoldChange > 0.5
res_df$Expresion[idx_up] <- "Sobreexpresado"

idx_down <- !is.na(res_df$padj) &
  res_df$padj < 0.05 &
  res_df$log2FoldChange < -0.5
res_df$Expresion[idx_down] <- "Subexpresado"

# Identificar el Top 15 de genes más significativos
res_df$Gene <- rownames(res_df)
top_genes <- res_df[!is.na(res_df$padj), ]
top_genes <- top_genes[order(top_genes$padj), ][1:15, ]

library(knitr)
library(kableExtra)

# Tomas tu top 30 de genes
res_df_top <- head(res_df[order(res_df$padj), ], 30)

# Generas el archivo .tex formateado
res_df_top |>
  kbl(format = "latex", booktabs = TRUE, digits = 4) |>
  kable_styling(latex_options = "scale_down") |>
  save_kable("Results/Tabla_S1_Top20.tex")

# Volcano Plot
volcano <- ggplot(
  res_df[!is.na(res_df$padj), ],
  aes(x = log2FoldChange, y = -log10(padj), color = Expresion)
) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_text(
    data = top_genes,
    aes(label = Gene),
    size = 3.5,
    vjust = -0.5,
    fontface = "bold",
    show.legend = FALSE
  ) +
  scale_color_manual(
    values = c(
      "Sobreexpresado" = "#FF6B00",
      "Subexpresado" = "#112CC2",
      "No significativo" = "#3A3A3A"
    )
  ) +
  coord_cartesian(ylim = c(0, 20)) +
  theme_minimal() +
  labs(
    title = "Volcano Plot - DESeq2",
    x = "Log2 Fold Change",
    y = "-Log10 padj"
  )

# Guardado de la imagen
save_file(out_v)
ggsave(out_v, plot = volcano, width = 7, height = 5, dpi = 300)
message(paste("Volcano plot guardado con éxito en:", normalizePath(out_v)))
if (interactive()) print(volcano)

# 8. Heatmap
vsd <- vst(dds, blind = FALSE)

# Seleccionar los 100 genes con mayor varianza en la matriz VST
top100_var <- head(order(rowVars(assay(vsd)), decreasing = TRUE), 100)
mat_top <- assay(vsd)[top100_var, ]
mat_top <- t(scale(t(mat_top)))

# Eliminar posibles NAs generados por genes con varianza cero
mat_top <- mat_top[complete.cases(mat_top), ]

save_file(out_h)

# Extraer los nombres de los 100 genes más variables
genes_top100 <- rownames(mat_top)

# Obtener la información de esos 100 genes desde los resultados de DESeq2
tabla_s2_df <- res_df |>
  filter(gene %in% genes_top100) |>
  select(gene, baseMean, log2FoldChange, pvalue, padj) |>
  arrange(padj) |>
  rename(
    Gene_Symbol = gene,
    baseMean = baseMean,
    log2FC = log2FoldChange,
    pvalue = pvalue,
    padj = padj
  )

# Guardar en Excel
save_file("Results/Tabla_S2_Top100_Genes.xlsx")
write_xlsx(tabla_s2_df, path = "Results/Tabla_S2_Top100_Genes.xlsx")

# 4. Guardar los primeros 30 genes formateados en código LaTeX
tabla_s2_df |>
  head(30) |>
  kbl(format = "latex", booktabs = TRUE, digits = 4, caption = "Top 30 genes") |>
  kable_styling(latex_options = "scale_down") |>
  save_kable("Results/Tabla_S2_Top30.tex")

message("Tabla Suplementaria S2 exportada exitosamente a Excel y LaTeX.")

# Configurar colores de anotación
grupos_presentes <- levels(meta$Health_Label)
colores_grupo <- setNames(
  c("#4D4D4D", "#FF7400", "#112CC2")[seq_along(grupos_presentes)],
  grupos_presentes
)

p_heat <- pheatmap(
  mat_top,
  annotation_col = data.frame(
    Grupo = meta$Health_Label,
    row.names = colnames(dds)
  ),
  annotation_colors = list(
    Grupo = colores_grupo
  ),
  show_colnames = FALSE,
  show_rownames = FALSE,
  cluster_cols = TRUE,
  cluster_rows = TRUE,
  main = "Top 100 Genes Más Variables (VST)",
  filename = out_h,
  width = 8,
  height = 10
)

message(paste("Heatmap guardado con éxito en:", normalizePath(out_h)))