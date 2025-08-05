# === Load required libraries ===
library(faircause)

# === Load the dataset ===
df <- read.csv("DECAF2_predictions_XGBoost_first_low_recall.csv")

# === Recode race: 1 vs all others ===
df$race_binary <- ifelse(df$race == 7, "White", "NonWhite")
df$race_binary <- factor(df$race_binary)
df$race_binary <- relevel(df$race_binary, ref = "White")  # White as privileged

# === Convert variable types ===
df$bar_passed <- as.integer(df$bar_passed)
df$sex <- as.factor(df$sex)

# === Define variable groups ===
X <- "race_binary"                            # Binary sensitive attribute
Y <- "bar_passed"                             # Target
Z <- c("sex")                                 # Moderator
W <- setdiff(colnames(df), c("race", "race_binary", Y, Z))  # Features


# === 🔍 Check class balance ===
cat("Label distribution:\n")
print(table(df[[Y]]))  # Shows how many 0s and 1s in bar_passed

# === Run fairness evaluation ===
result <- fairness_cookbook(df, X = X, W = W, Z = Z, Y = Y,
                            x0 = "White", x1 = "NonWhite")

# === Extract results ===
measures <- result$measures
result_df <- data.frame(Group_Comparison = "White vs NonWhite")
for (i in 1:nrow(measures)) {
  metric <- measures$measure[i]
  result_df[[paste0(metric, "_mean")]] <- round(measures$value[i], 4)
  result_df[[paste0(metric, "_sd")]]   <- round(measures$sd[i], 4)
}

# === Save to CSV ===
write.csv(result_df, "fairness_results_white_vs_nonwhite.csv", row.names = FALSE)

# === Print to console ===
cat("✅ Fairness result for White vs NonWhite:\n")
print(result_df)