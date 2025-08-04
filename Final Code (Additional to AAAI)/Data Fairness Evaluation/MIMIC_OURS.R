# Load required library
library(faircause)

# Read CSV
df_raw <- read.csv("generated_data_Our_prompts_MIMIC.csv")
names(df_raw) <- make.names(names(df_raw))

# Define expected columns
expected_columns <- make.names(c(
  "race", "mortality", "age", "gender", "charlson_index", "elective_status",
  "25000", "2724", "311", "4019", "41401", "42731", "4280", "53081",
  "5849", "E785", "F329", "F419", "I10", "I2510", "K219", "N179",
  "V1582", "Z20822", "Z7901", "Z87891", "los_seconds"
))

# Keep only those columns that exist
available_columns <- intersect(expected_columns, names(df_raw))
df <- df_raw[, available_columns]

# Ensure data types and preprocess
df$race <- factor(df$race)
df$race <- relevel(df$race, ref = "WHITE")

if ("gender" %in% names(df)) df$gender <- factor(df$gender)

# Binarize los_seconds: 1 if >= 345600 (4 days), else 0


# Define variables for fairness analysis
X <- "race"
Y <- "los_seconds"
Z <- intersect(c("age", "gender"), names(df))
W <- setdiff(names(df), c(X, Y, Z))

# Run fairness analysis
result <- fairness_cookbook(df, X = X, W = W, Z = Z, Y = Y,
                            x0 = "WHITE", x1 = "BLACK/AFRICAN AMERICAN")

# Extract results
measures <- result$measures
result_df <- data.frame(Model = "ICU_LOS_Binary")

for (i in 1:nrow(measures)) {
  metric <- measures$measure[i]
  result_df[[paste0(metric, "_mean")]] <- round(measures$value[i], 4)
  result_df[[paste0(metric, "_sd")]]   <- round(measures$sd[i], 4)
}

# Save and print
write.csv(result_df, "fairness_results_icu_los21.csv", row.names = FALSE)
cat("✅ Fairness results:\n")
print(result_df)
