# === Load required libraries ===
library(faircause)

# === List of model names ===
model_names <- c("Decision_Tree", "Logistic_Regression", "Random_Forest", "SVM", "XGBoost")

# === Initialize results data frame ===
all_results <- data.frame()

# === Loop over model CSVs ===
for (model in model_names) {
  
  # Load model prediction CSV
  file_name <- paste0("Our_TS_COMPAS_predictions_", model, ".csv")
  df <- read.csv(file_name)
  
  # Recode race: African-American vs Non-African-American
  df$race_binary <- ifelse(df$race_African.American == 1, "African-American", "Non-African-American")
  df$race_binary <- factor(df$race_binary)
  df$race_binary <- relevel(df$race_binary, ref = "African-American")  # Privileged group
  
  # Ensure correct types
  df$two_year_recid <- as.integer(df$two_year_recid)
  if ("sex" %in% colnames(df)) {
    df$sex <- as.factor(df$sex)
  }
  
  # Define groups
  X <- "race_binary"                            # Sensitive attribute
  Y <- "two_year_recid"                         # Target variable
  Z <- c("sex")                                 # Moderator (if applicable)
  W <- setdiff(colnames(df), c("race_African.American", "race_binary", Y, Z))  # Features
  
  # Run fairness analysis
  result <- fairness_cookbook(df, X = X, W = W, Z = Z, Y = Y,
                              x0 = "African-American", x1 = "Non-African-American")
  
  # Extract and format metrics
  measures <- result$measures
  result_df <- data.frame(Model = model, Group_Comparison = "African-American vs Non-African-American")
  for (i in 1:nrow(measures)) {
    metric <- measures$measure[i]
    result_df[[paste0(metric, "_mean")]] <- round(measures$value[i], 4)
    result_df[[paste0(metric, "_sd")]]   <- round(measures$sd[i], 4)
  }
  
  # Append to combined result
  all_results <- rbind(all_results, result_df)
  
  cat("✅ Processed:", model, "\n")
}

# === Save all results to a single CSV ===
write.csv(all_results, "fairness_results_all_models_COMPAS_TS.csv", row.names = FALSE)
cat("📁 All fairness results saved to: fairness_results_all_models_COMPAS_TS.csv\n")
