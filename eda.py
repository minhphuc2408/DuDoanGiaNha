import matplotlib.pyplot as plt
import seaborn as sns

# Gộp X_scaled và y để phân tích tương quan
df_corr = X_scaled.copy()
df_corr["SalePrice"] = y

# Tính ma trận tương quan
corr_matrix = df_corr.corr(numeric_only=True)

# Lấy 15 biến có tương quan cao nhất với SalePrice
top_corr = corr_matrix["SalePrice"].abs().sort_values(ascending=False).head(15).index

# Vẽ heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_corr[top_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Heatmap - Tương quan giữa các đặc trưng và SalePrice", fontsize=14, pad=12)
plt.show()

# In top đặc trưng tương quan mạnh nhất với SalePrice
print("Top 15 đặc trưng có tương quan cao nhất với SalePrice:")
display(corr_matrix["SalePrice"].sort_values(ascending=False).head(15))