suppressPackageStartupMessages({
  library(DESeq2)
  library(dplyr)
  library(ggplot2)
  library(pheatmap)
})

# Función auxiliar para crear directorios
save_file <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
}

# 1. Definición global de columnas clínicas a excluir de los conteos
cols_clinical <- c(
  "Health_Label", "Donor_ID", "RIN", "PMI", "Age", "Gender",
  "Region", "Disease"
)

# 2. Gestionar entrada/salida para Snakemake / Standalone
if (exists("snakemake")) {
  in_c <- snakemake@input[["counts"]]
  in_t <- snakemake@input[["test"]]
  out_tr <- snakemake@output[["train_top100"]]
  out_te <- snakemake@output[["test_top100"]]
  out_v <- snakemake@output[["volcano_deseq2"]]
  out_h <- snakemake@output[["heatmap_deseq2"]]
} else {
  in_c <- "Data/Processed/datos_train_completos.tsv"
  in_t <- "Data/Processed/datos_test_completos.tsv"
  out_tr <- "Data/Processed/train_top100_genes.tsv"
  out_te <- "Data/Processed/test_top100_genes.tsv"
  out_v <- "Results/Imagenes/volcano_deseq2.png"
  out_h <- "Results/Imagenes/heatmap_deseq2.png"
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
meta <- df_train[, colnames(df_train) %in% covs]
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
# Aislemos únicamente las columnas que corresponden a genes
df_expr <- df_train[, !colnames(df_train) %in% cols_clinical]

# Forzar la conversión explícita a matriz numérica
counts_num <- matrix(
  as.numeric(as.matrix(df_expr)),
  nrow = nrow(df_expr),
  ncol = ncol(df_expr),
  dimnames = list(rownames(df_expr), colnames(df_expr))
)

# Transponer a formato (Genes x Muestras) y limpiar valores
counts_raw <- t(counts_num)
counts_raw[is.na(counts_raw)] <- 0
counts_raw[counts_raw < 0] <- 0
counts_clean <- round(counts_raw)

terms <- intersect(c("RIN", "PMI", "Age", "Gender"), colnames(meta))
design_formula <- as.formula(
  paste("~", paste(c(terms, "Health_Label"), collapse = " + "))
)

dds <- DESeqDataSetFromMatrix(
  countData = counts_clean,
  colData = meta,
  design = design_formula
)

dds <- dds[rowSums(counts(dds)) >= 10, ]
dds$Health_Label <- relevel(dds$Health_Label, ref = "Control")
dds <- DESeq(dds)

# 6. Extracción de Resultados y Selección del Top 100
res <- results(dds, contrast = c("Health_Label", "Parkinson", "Control"))
res_df <- as.data.frame(res)

# Ranking por p-valor bruto (pvalue) asegurando selección completa
res_sorted <- res_df |>
  filter(!is.na(pvalue)) |>
  arrange(pvalue)

top100 <- rownames(res_sorted)[seq_len(min(100, nrow(res_sorted)))]

# Incluimos explícitamente "Donor_ID" junto con las variables requeridas
keep_cols <- c(top100, "Health_Label", "Donor_ID")

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

volcano <- ggplot(
  res_df[!is.na(res_df$padj), ],
  aes(x = log2FoldChange, y = -log10(padj), color = Expresion)
) +
  geom_point(alpha = 0.5, size = 1.5) +
  scale_color_manual(
    values = c(
      "Sobreexpresado" = "#FF6B00",
      "Subexpresado" = "#112CC2",
      "No significativo" = "#3A3A3A"
    )
  ) +
  theme_minimal() +
  labs(
    title = "Volcano Plot - DESeq2",
    x = "Log2 Fold Change",
    y = "-Log10 padj"
  )

save_file(out_v)
ggsave(out_v, plot = volcano, width = 7, height = 5, dpi = 300)
message(paste("[OK] Volcano plot guardado con éxito en:", normalizePath(out_v)))
if (interactive()) print(volcano)

# 8. Heatmap
vsd <- vst(dds, blind = FALSE)
mat_top <- t(scale(t(assay(vsd)[top100, ])))

save_file(out_h)
p_heat <- pheatmap(
  mat_top,
  annotation_col = data.frame(
    Grupo = meta$Health_Label,
    row.names = colnames(dds)
  ),
  annotation_colors = list(
    Grupo = c(Control = "#4D4D4D", Parkinson = "#FF7400")
  ),
  show_colnames = FALSE,
  show_rownames = FALSE,
  cluster_cols = TRUE,
  cluster_rows = TRUE,
  main = "Top 100 Genes (DESeq2)",
  filename = out_h,
  width = 8,
  height = 10
)

message(paste("[OK] Heatmap guardado con éxito en:", normalizePath(out_h)))
if (interactive()) print(p_heat)