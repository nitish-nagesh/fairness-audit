# === Load required libraries ===
library(faircause)

# === Load datasets ===
df <- read.csv("compas_cleaned_with_ids.csv")
preds <- read.csv("predictions.csv")

# === Ensure matching ID type ===
df$row_id <- as.integer(df$row_id)
preds$row_id <- as.integer(preds$row_id)

# === Merge on row_id to align rows and attach predictions ===
df_merged <- merge(df, preds, by = "row_id")

# === Convert sensitive and categorical variables ===
df_merged$race <- ifelse(df_merged$race_African.American == 1, "African-American",
                         ifelse(df_merged$race_Caucasian == 1, "Caucasian", NA))
df_merged$race <- factor(df_merged$race)
df_merged$race <- relevel(df_merged$race, ref = "Caucasian")
df_merged$sex <- factor(df_merged$sex)

# === Define static variables ===
X <- "race"
Z <- c("age", "sex")
model_columns <- c("Decision_Tree", "Logistic_Regression", "Random_Forest", "SVM", "XGBoost")

# === Function to extract fairness values from result$measures ===
extract_fairness_row <- function(result, label) {
  mdf <- result$measures
  row <- data.frame(Model = label)
  for (i in 1:nrow(mdf)) {
    metric <- mdf$measure[i]
    row[[paste0(metric, "_mean")]] <- round(mdf$value[i], 4)
    row[[paste0(metric, "_sd")]] <- round(mdf$sd[i], 4)
  }
  return(row)
}

# === Collect results ===
results <- list()

for (i in seq_along(model_columns)) {
  model <- model_columns[i]
  df_temp <- df_merged
  df_temp$two_year_recid <- as.integer(df_temp[[model]])  # Replace label with prediction
  
  W <- setdiff(colnames(df_temp), c(X, "two_year_recid", Z, "row_id",
                                    "race_African.American", "race_Caucasian", 
                                    "True", model_columns))
  
  result <- fairness_cookbook(df_temp, X = X, W = W, Z = Z, Y = "two_year_recid",
                              x0 = "Caucasian", x1 = "African-American")
  
  row <- extract_fairness_row(result, model)
  results[[i]] <- row
}

# === Combine and save to CSV ===
results_df <- do.call(rbind, results)
setwd(".")
write.csv(results_df, "fairness_results_of_original_ML_Models.csv", row.names = FALSE)

# === Print summary ===
cat("✅ Fairness results saved to fairness_results_from_preds.csv\n")
print(results_df)

