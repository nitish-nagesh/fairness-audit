# Load required libraries
library(faircause)

# Load the dataset
df <- read.csv("compas_cleaned.csv")

# Convert relevant variables
df$race <- ifelse(df$race_African.American == 1, "African-American",
                  ifelse(df$race_Caucasian == 1, "Caucasian", NA))
df$race <- factor(df$race)
df$race <- relevel(df$race, ref = "Caucasian")

df$two_year_recid <- as.integer(df$two_year_recid)
df$sex <- factor(df$sex)

# Define variables
X <- "race"
Y <- "two_year_recid"
Z <- c("age", "sex")
W <- setdiff(colnames(df), c(X, Y, Z, "race_African.American", "race_Caucasian"))

# Run fairness cookbook
result <- fairness_cookbook(df, X = X, W = W, Z = Z, Y = Y,
                            x0 = "Caucasian", x1 = "African-American")

# Extract and format fairness metrics
measures <- result$measures
result_df <- data.frame(Model = "COMPAS")

for (i in 1:nrow(measures)) {
  metric <- measures$measure[i]
  result_df[[paste0(metric, "_mean")]] <- round(measures$value[i], 4)
  result_df[[paste0(metric, "_sd")]]   <- round(measures$sd[i], 4)
}

# Save to CSV
setwd(".")
write.csv(result_df, "fairness_results_compas.csv", row.names = FALSE)

# Print to console
cat("✅ Fairness results for COMPAS dataset:\n")
print(result_df)
