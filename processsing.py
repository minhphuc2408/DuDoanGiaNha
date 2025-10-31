from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

df = train.copy()

# 1. Tạo các đặc trưng mới
df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
df["TotalBath"] = (df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0) +df["BsmtFullBath"].fillna(0) + 0.5 * df["BsmtHalfBath"].fillna(0))
df["TotalPorchSF"] = (df["OpenPorchSF"].fillna(0) + df["EnclosedPorch"].fillna(0) +df["3SsnPorch"].fillna(0) + df["ScreenPorch"].fillna(0))

if "YrSold" in df.columns:
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
else:
    df["Age"] = 2025 - df["YearBuilt"]
    df["RemodAge"] = 2025 - df["YearRemodAdd"]

df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

# 2. Xử lý giá trị thiếu
cat_cols = df.select_dtypes(include="object").columns
for c in cat_cols:
    df[c] = df[c].fillna("None")

num_cols = df.select_dtypes(include=np.number).columns
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# 3. Mã hóa biến phân loại
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()

X = df_encoded.drop("SalePrice", axis=1)
y = df_encoded["SalePrice"]

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("Dữ liệu sau xử lý:", X_scaled.shape)
display(X_scaled.head())