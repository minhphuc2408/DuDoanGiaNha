import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle, os, datetime, traceback
import matplotlib.pyplot as plt

# ================================
# ⚙️ CẤU HÌNH CHUNG
# ================================
st.set_page_config(page_title="🏠 Ứng dụng Dự đoán Giá Nhà", page_icon="🏡", layout="wide")

st.title("🏡 DỰ ĐOÁN GIÁ NHÀ - BẢN MỞ RỘNG")


#  HÀM LOAD MODEL

def load_model_any(path_or_file):
    """Tự động nhận dạng và load model dù là object hay dict."""
    try:
        if isinstance(path_or_file, str):
            obj = joblib.load(path_or_file)
        else:
            obj = joblib.load(path_or_file)
    except Exception:
        path_or_file.seek(0)
        obj = pickle.load(path_or_file)

    if isinstance(obj, dict):
        for key in ["model", "best_model", "regressor", "estimator"]:
            if key in obj:
                st.sidebar.info(f"File chứa dict → lấy model ở key '{key}'")
                return obj[key]
        st.sidebar.warning("File chứa dictionary, không có key 'model' → dùng toàn bộ dict (có thể lỗi).")
        return obj
    return obj


# ================================
# 📋 MENU SIDEBAR
# ================================
menu = st.sidebar.radio(
    "📚 Menu",
    ["🏠 Dự đoán giá nhà", "📜 Lịch sử dự đoán", "📊 So sánh giá nhà", "🗺️ Bản đồ giá nhà", "⚙️ Thông tin mô hình"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Thiết lập mô hình")

# ================================
# 🔹 TẢI MÔ HÌNH
# ================================
model = None
model_info = ""

if os.path.exists("xgb_tuned_model.pkl"):
    try:
        model = load_model_any("xgb_tuned_model.pkl")
        model_info = "Đã tải mô hình từ `xgb_tuned_model.pkl`"
        st.sidebar.success("✅ Mô hình mặc định đã được tải!")
    except Exception as e:
        st.sidebar.error(f"❌ Không thể load model mặc định: {e}")
else:
    st.sidebar.warning("⚠️ Không tìm thấy file `xgb_tuned_model.pkl` trong thư mục hiện tại.")

uploaded_model = st.sidebar.file_uploader("📦 Tải mô hình mới (.pkl / .joblib)", type=["pkl", "joblib"])
if uploaded_model is not None:
    try:
        model = load_model_any(uploaded_model)
        model_info = f"Đã tải mô hình từ file: {uploaded_model.name}"
        st.sidebar.success("✅ Mô hình mới đã được tải!")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi tải mô hình: {e}")
        st.sidebar.code(traceback.format_exc())

if model is None:
    st.stop()



#  TRANG 1: DỰ ĐOÁN GIÁ NHÀ

if menu == "🏠 Dự đoán giá nhà":
    st.header("📋 Nhập thông tin ngôi nhà để dự đoán")

    # --- Nhóm 1: Thông tin cơ bản ---
    st.markdown("### 🏠 Thông tin chung")
    col1, col2, col3 = st.columns(3)
    with col1:
        overall_qual = st.slider("Chất lượng tổng thể (OverallQual)", 1, 10, 5)
        overall_cond = st.slider("Tình trạng tổng thể (OverallCond)", 1, 10, 5)
        year_built = st.number_input("Năm xây dựng (YearBuilt)", 1800, datetime.datetime.now().year, 2005)
    with col2:
        year_remod = st.number_input("Năm sửa chữa / cải tạo (YearRemodAdd)", 1800, datetime.datetime.now().year, 2010)
        lot_area = st.number_input("Diện tích lô đất (LotArea)", 500, 200000, 8000)
        neighborhood = st.selectbox("Khu vực (Neighborhood)", [
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Sawyer", "Gilbert",
            "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber", "IDOTRR", "NWAmes",
            "StoneBr", "SWISU", "ClearCr", "Blmngtn", "Veenker", "MeadowV", "BrDale", "NPkVill", "Blueste"
        ])
    with col3:
        exter_qual = st.selectbox("Chất lượng bên ngoài (ExterQual)", ["Ex", "Gd", "TA", "Fa", "Po"])
        kitchen_qual = st.selectbox("Chất lượng bếp (KitchenQual)", ["Ex", "Gd", "TA", "Fa", "Po"])
        bsmt_qual = st.selectbox("Chất lượng tầng hầm (BsmtQual)", ["Ex", "Gd", "TA", "Fa", "Po", "NA"])

    # --- Nhóm 2: Diện tích và tầng ---
    st.markdown("### 📐 Diện tích & Kết cấu")
    col4, col5, col6 = st.columns(3)
    with col4:
        total_bsmt_sf = st.number_input("Diện tích tầng hầm (TotalBsmtSF)", 0, 6000, 800)
        first_flr_sf = st.number_input("Diện tích tầng 1 (1stFlrSF)", 0, 6000, 1200)
    with col5:
        second_flr_sf = st.number_input("Diện tích tầng 2 (2ndFlrSF)", 0, 6000, 400)
        gr_liv_area = st.number_input("Diện tích sử dụng (GrLivArea)", 300, 10000, 1500)
    with col6:
        garage_area = st.number_input("Diện tích garage (GarageArea)", 0, 2000, 400)
        garage_cars = st.slider("Số xe chứa trong garage (GarageCars)", 0, 5, 2)

    # --- Nhóm 3: Phòng và tiện nghi ---
    st.markdown("### 🛏️ Phòng & Tiện nghi")
    col7, col8, col9 = st.columns(3)
    with col7:
        full_bath = st.slider("Số phòng tắm đầy đủ (FullBath)", 0, 5, 2)
        half_bath = st.slider("Số phòng tắm nửa (HalfBath)", 0, 5, 1)
    with col8:
        bedrooms = st.slider("Số phòng ngủ (BedroomAbvGr)", 0, 10, 3)
        total_rooms = st.slider("Tổng số phòng (TotRmsAbvGrd)", 1, 15, 7)
    with col9:
        fireplaces = st.slider("Số lò sưởi (Fireplaces)", 0, 3, 1)
        heating_qc = st.selectbox("Chất lượng hệ thống sưởi (HeatingQC)", ["Ex", "Gd", "TA", "Fa", "Po"])
        central_air = st.selectbox("Điều hòa trung tâm (CentralAir)", ["Y", "N"])
        paved_drive = st.selectbox("Đường lái xe lát gạch (PavedDrive)", ["Y", "P", "N"])

    # --- Biến phụ ---
    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    total_bath = full_bath + 0.5 * half_bath
    age = datetime.datetime.now().year - year_built

    input_df = pd.DataFrame([{
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "LotArea": lot_area,
        "Neighborhood": neighborhood,
        "ExterQual": exter_qual,
        "KitchenQual": kitchen_qual,
        "BsmtQual": bsmt_qual,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf,
        "GrLivArea": gr_liv_area,
        "GarageArea": garage_area,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "TotRmsAbvGrd": total_rooms,
        "Fireplaces": fireplaces,
        "HeatingQC": heating_qc,
        "CentralAir": central_air,
        "PavedDrive": paved_drive,
        "TotalSF": total_sf,
        "TotalBath": total_bath,
        "Age": age
    }])

    st.write("### 🧾 Dữ liệu nhập")
    st.dataframe(input_df.T)

    if st.button("🚀 Dự đoán giá nhà"):
        try:
            X = input_df.copy()
            feature_names = None
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
            elif hasattr(model, "get_booster"):
                try:
                    feature_names = model.get_booster().feature_names
                except Exception:
                    feature_names = None

            if feature_names is not None:
                for col in feature_names:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_names]

            y_pred = model.predict(X)
            y_value = float(y_pred[0])
            if y_value < 20:
                y_value = np.expm1(y_value)

            st.success(f"💰 Giá bán ước tính: **${y_value:,.2f}**")

            history_file = "prediction_history.csv"
            row = input_df.copy()
            row["PredictedPrice"] = y_value
            if not os.path.exists(history_file):
                row.to_csv(history_file, index=False)
            else:
                row.to_csv(history_file, mode="a", header=False, index=False)

        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
            st.code(traceback.format_exc())



#  TRANG 2: LỊCH SỬ DỰ ĐOÁN

elif menu == "📜 Lịch sử dự đoán":
    st.header("📜 Lịch sử các lần dự đoán")

    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        st.dataframe(history)
        st.download_button("⬇️ Tải lịch sử CSV", history.to_csv(index=False), "history.csv")
        if st.button("🧹 Xóa toàn bộ lịch sử"):
            os.remove(history_file)
            st.warning("Đã xóa toàn bộ lịch sử dự đoán.")
            st.experimental_rerun()
    else:
        st.info("Chưa có dữ liệu lịch sử nào.")



#  TRANG 3: SO SÁNH GIÁ NHÀ

elif menu == "📊 So sánh giá nhà":
    st.header("📊 So sánh giá nhà giữa các khu vực (dựa trên lịch sử dự đoán)")

    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        df = pd.read_csv(history_file)
        if "Neighborhood" in df.columns and "PredictedPrice" in df.columns:
            avg_prices = df.groupby("Neighborhood")["PredictedPrice"].mean().sort_values(ascending=False)
            top_n = st.slider("Hiển thị số khu vực:", 3, len(avg_prices), 10)

            st.subheader("📈 Biểu đồ giá trung bình theo khu vực")
            fig, ax = plt.subplots(figsize=(10, 5))
            avg_prices.head(top_n).plot(kind="bar", color="skyblue", ax=ax)
            ax.set_ylabel("Giá trung bình ($)")
            ax.set_xlabel("Khu vực")
            ax.set_title("So sánh giá trung bình giữa các khu vực")
            st.pyplot(fig)

            st.write("### 📊 Bảng giá trung bình")
            st.dataframe(avg_prices.head(top_n).reset_index())
        else:
            st.error("File lịch sử không có cột 'Neighborhood' hoặc 'PredictedPrice'")
    else:
        st.warning("⚠️ Chưa có lịch sử dự đoán nào để so sánh.")



#  TRANG 4: BẢN ĐỒ GIÁ NHÀ

elif menu == "🗺️ Bản đồ giá nhà":
    st.header("🗺️ Bản đồ phân bố giá nhà theo khu vực (dựa trên lịch sử dự đoán)")

    import folium
    from streamlit_folium import st_folium

    history_file = "prediction_history.csv"
    if not os.path.exists(history_file):
        st.warning("⚠️ Chưa có dữ liệu lịch sử nào để hiển thị bản đồ.")
    else:
        df = pd.read_csv(history_file)
        if "Neighborhood" not in df.columns or "PredictedPrice" not in df.columns:
            st.error("⚠️ File lịch sử thiếu cột cần thiết.")
        else:
            # ⚙️ Tọa độ mẫu cho từng khu vực
            neighborhood_coords = {
                "NAmes": [42.062, -93.625], "CollgCr": [42.025, -93.685], "OldTown": [42.034, -93.615],
                "Edwards": [42.045, -93.655], "Somerst": [42.05, -93.63], "NridgHt": [42.065, -93.64],
                "Sawyer": [42.055, -93.67], "Gilbert": [42.07, -93.61], "SawyerW": [42.045, -93.68],
                "BrkSide": [42.035, -93.61], "Crawfor": [42.045, -93.62], "Mitchel": [42.06, -93.63],
                "NoRidge": [42.07, -93.66], "Timber": [42.05, -93.66], "IDOTRR": [42.03, -93.61],
                "NWAmes": [42.07, -93.65], "StoneBr": [42.07, -93.63], "SWISU": [42.02, -93.63],
                "ClearCr": [42.07, -93.67], "Blmngtn": [42.08, -93.63], "Veenker": [42.08, -93.65],
                "MeadowV": [42.04, -93.67], "BrDale": [42.04, -93.62], "NPkVill": [42.03, -93.62],
                "Blueste": [42.02, -93.61],
            }

            avg_prices = df.groupby("Neighborhood")["PredictedPrice"].mean().reset_index()

            min_price, max_price = st.slider(
                "Lọc theo khoảng giá ($):",
                float(df["PredictedPrice"].min()),
                float(df["PredictedPrice"].max()),
                (float(df["PredictedPrice"].min()), float(df["PredictedPrice"].max()))
            )

            avg_prices = avg_prices[
                (avg_prices["PredictedPrice"] >= min_price) &
                (avg_prices["PredictedPrice"] <= max_price)
            ]

            m = folium.Map(location=[42.05, -93.64], zoom_start=12)

            for _, row in avg_prices.iterrows():
                name = row["Neighborhood"]
                price = row["PredictedPrice"]
                if name in neighborhood_coords:
                    folium.CircleMarker(
                        location=neighborhood_coords[name],
                        radius=9,
                        color="blue",
                        fill=True,
                        fill_color="lightblue",
                        popup=f"<b>{name}</b><br>💰 Giá TB: ${price:,.0f}"
                    ).add_to(m)

            st_folium(m, width=750, height=500)



# TRANG 5: THÔNG TIN MÔ HÌNH

elif menu == "⚙️ Thông tin mô hình":
    st.header("⚙️ Thông tin mô hình hiện tại")
    st.write(model_info)
    if hasattr(model, "feature_names_in_"):
        st.write(f"**Số lượng đặc trưng:** {len(model.feature_names_in_)}")
        st.dataframe(pd.DataFrame(model.feature_names_in_, columns=["Tên đặc trưng"]))
    else:
        st.info("Mô hình không có danh sách đặc trưng rõ ràng.")
