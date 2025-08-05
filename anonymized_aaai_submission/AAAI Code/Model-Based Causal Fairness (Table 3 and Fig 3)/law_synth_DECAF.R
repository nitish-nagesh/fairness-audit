# === Load required libraries ===
library(faircause)

# === List of model names ===
model_names <- c("Decision_Tree", "XGBoost")

# === Initialize results data frame ===
all_results <- data.frame()

# === Loop over model CSVs ===
for (model in model_names) {
  
  # Load model prediction CSV
  file_name <- paste0("DECAF_predictions_", model, ".csv")
  df <- read.csv(file_name)
  
  # Recode race: White vs NonWhite
  df$race_binary <- ifelse(df$race == 7, "White", "NonWhite")
  df$race_binary <- factor(df$race_binary)
  df$race_binary <- relevel(df$race_binary, ref = "White")
  
  # Ensure correct types
  df$bar_passed <- as.integer(df$bar_passed)
  if ("sex" %in% colnames(df)) {
    df$sex <- as.factor(df$sex)
  }
  
  # Define groups
  X <- "race_binary"                            # Sensitive attribute
  Y <- "bar_passed"                             # Target
  Z <- c("sex")                                 # Moderator
  W <- setdiff(colnames(df), c("race", "race_binary", Y, Z))  # Features
  
  # Run fairness analysis
  result <- fairness_cookbook(df, X = X, W = W, Z = Z, Y = Y,
                              x0 = "White", x1 = "NonWhite")
  
  # Extract and format metrics
  measures <- result$measures
  result_df <- data.frame(Model = model, Group_Comparison = "White vs NonWhite")
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
write.csv(all_results, "fairness_results_all_models_DECAF_TS.csv", row.names = FALSE)
cat("📁 All fairness results saved to: fairness_results_all_models.csv\n")
