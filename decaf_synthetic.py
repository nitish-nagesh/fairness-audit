from synthcity.plugins import Plugins
import pandas as pd
from synthcity.plugins.core.dataloader import GenericDataLoader

compas_real = pd.read_csv("/Users/nitishnagesh/Downloads/compas_selected.csv")
# Initialize the plugin
plugin = Plugins().get("decaf", n_iter=200)

# Fit the plugin to the dataset
plugin.fit(compas_real)

# Generate synthetic data
compas_synthetic_data = plugin.generate(1000)

# Display synthetic data
compas_synthetic_data

# Convert GenericDataLoader to a DataFrame
compas_synthetic_data = compas_synthetic_data.dataframe()
# Convert 'num' column: if value is 0, keep it as 0; otherwise, set it to 1
# print(compas_synthetic_data.head())

compas_synthetic_data.to_csv("../compas_synthetic_data_1000_200_epochs.csv", index=False)