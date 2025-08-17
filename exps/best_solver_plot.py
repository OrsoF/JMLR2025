import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("best_solver.csv")
df['Final Precision'] = df['Final Precision'].apply(np.log10)

available_models = df["Model Name"].unique()
modelname = input(f"Choose a model from {available_models}: ")

df_temp = df[df["Model Name"] == modelname]

# Plotting
plt.figure(figsize=(10, 6))
sns.boxplot(x="Discount", y="Final Precision", hue="Best Solver", data=df_temp)

# Set plot labels and title
plt.title(
    f"{modelname} : Comparison of Final Precision for Different Solvers and Discount Values"
)
plt.xlabel("Discount")
plt.ylabel("Final Precision")

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
