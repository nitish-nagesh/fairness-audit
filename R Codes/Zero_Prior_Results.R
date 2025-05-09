# === Load required libraries ===
library(faircause)

# === Load the dataset ===
df <- read.csv("Synt_Data_Zero_Prior_with_predictionsP1.csv")

# === Convert relevant variables ===
df$race <- ifelse(df$race_African.American == 1, "African-American",
                  ifelse(df$race_Caucasian == 1, "Caucasian", NA))
df$race <- factor(df$race)
df$race <- relevel(df$race, ref = "Caucasian")

df$two_year_recid <- as.integer(df$two_year_recid)
df$sex <- factor(df$sex)

# === Define variable groups ===
X <- "race"
Y <- "two_year_recid"
Z <- c("age", "sex")
model_columns <- c("Decision_Tree", "Logistic_Regression", "Random_Forest", "SVM", "XGBoost")
W <- setdiff(colnames(df), c(X, Y, Z, "race_African.American", "race_Caucasian", model_columns))

# === Extract from result$measures ===
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

# === Compute and collect all results ===
results <- list()

# Fairness for True Label
result_true <- fairness_cookbook(df, X = X, W = W, Z = Z, Y = Y,
                                 x0 = "Caucasian", x1 = "African-American")
results[[1]] <- extract_fairness_row(result_true, "True_Label")

# Fairness for model predictions
for (i in seq_along(model_columns)) {
  model <- model_columns[i]
  df_temp <- df
  df_temp$two_year_recid <- as.integer(df_temp[[model]])
  
  result_model <- fairness_cookbook(df_temp, X = X, W = W, Z = Z, Y = Y,
                                    x0 = "Caucasian", x1 = "African-American")
  results[[i + 1]] <- extract_fairness_row(result_model, model)
}

# Combine and write to CSV
results_df <- do.call(rbind, results)
setwd(".")
write.csv(results_df, "fairness_results_Zero_Prior.csv", row.names = FALSE)
cat("✅ Results saved to fairness_results._Zero_Priorcsv\n")
print(results_df)
