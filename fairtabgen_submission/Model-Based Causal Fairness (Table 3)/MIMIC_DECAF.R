# === Load required libraries ===
library(faircause)

# === List of model names ===
model_names <- c("Decision_Tree", "Random_Forest", "XGBoost")

# === Initialize results data frame ===
all_results <- data.frame()

# === Loop over model CSVs ===
for (model in model_names) {
  
  # Load model prediction CSV
  file_name <- paste0("DECAF_TS_MIMIC_predictions_", model, ".csv")
  df <- read.csv(file_name)
  
  # Recode race: WHITE vs NONWHITE
  df$race_binary <- ifelse(df$race == "WHITE", "White", "NonWhite")
  df$race_binary <- factor(df$race_binary)
  df$race_binary <- relevel(df$race_binary, ref = "White")
  
  # Ensure correct types
  df$predicted_label <- as.integer(df$predicted_label)
  if ("gender" %in% colnames(df)) {
    df$gender <- as.factor(df$gender)
  }
  
  # Define groups
  X <- "race_binary"                            # Sensitive attribute
  Y <- "predicted_label"                        # Target (predicted label)
  Z <- c("gender")                              # Moderator
  W <- setdiff(colnames(df), c("race", "race_binary", Y, Z, "label", "probability"))  # Features
  
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
write.csv(all_results, "fairness_results_all_models_mimic_DECAF.csv", row.names = FALSE)
cat("📁 All fairness results saved to: fairness_results_all_models_mimic.csv\n")
