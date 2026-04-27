import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# 設定網頁標題與佈局
st.set_page_config(page_title="飯店預訂預測系統", layout="wide")

# 快取載入資料的函數，避免每次操作都重新讀取
@st.cache_data
def load_data():
    df = pd.read_csv('hotel_bookings.csv')
    # 簡單清理以供圖表展示
    df['children'] = df['children'].fillna(0)
    df = df[~((df['adults'] + df['children'] + df['babies']) == 0)]
    return df

# 載入模型與資料
try:
    model = joblib.load('nb_model.pkl')
    df = load_data()
except FileNotFoundError:
    st.error("找不到模型檔案 (nb_model.pkl) 或資料集 (hotel_bookings.csv)，請先執行訓練程式碼！")
    st.stop()

# 建立兩個分頁
tab1, tab2 = st.tabs(["📊 探索性資料分析 (EDA)", "🤖 訂單取消預測模型"])

# ================= 分頁 1：資料視覺化 =================
with tab1:
    st.title("探索性資料分析 (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        # 圖表 1：各飯店類型的取消比例
        st.subheader("不同飯店類型的取消狀況")
        fig1 = px.histogram(df, x="hotel", color="is_canceled", 
                            barmode="group", 
                            labels={'is_canceled': '是否取消 (1=是, 0=否)', 'hotel': '飯店類型'},
                            color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # 圖表 2：前置時間 (Lead Time) 與取消的關係
        st.subheader("前置時間分佈 (入住 vs 取消)")
        fig2 = px.box(df, x="is_canceled", y="lead_time", 
                      color="is_canceled",
                      labels={'is_canceled': '是否取消', 'lead_time': '前置時間 (天)'})
        st.plotly_chart(fig2, use_container_width=True)

# ================= 分頁 2：機器學習預測 =================
with tab2:
    st.title("訂單狀態預測器")
    st.write("請調整下方滑桿或輸入數值，預測該筆訂單最終是否會被取消。")
    
    st.sidebar.header("輸入客房訂單特徵")
    # 建立輸入介面
    lead_time = st.sidebar.slider("前置時間 (Lead Time - 天)", 0, 700, 30)
    special_requests = st.sidebar.slider("特殊需求總數", 0, 5, 0)
    parking_spaces = st.sidebar.selectbox("需要停車位數量", [0, 1, 2, 3])
    booking_changes = st.sidebar.slider("訂單變更次數", 0, 20, 0)
    previous_cancellations = st.sidebar.number_input("過往取消次數", min_value=0, max_value=26, value=0)
    is_repeated_guest = st.sidebar.radio("是否為回頭客？", ("否", "是"))

    # 轉換回頭客為模型需要的數值
    is_repeated_guest_val = 1 if is_repeated_guest == "是" else 0

    # 組織預測資料
    input_data = pd.DataFrame({
        'lead_time': [lead_time],
        'total_of_special_requests': [special_requests],
        'required_car_parking_spaces': [parking_spaces],
        'booking_changes': [booking_changes],
        'previous_cancellations': [previous_cancellations],
        'is_repeated_guest': [is_repeated_guest_val]
    })

    # 顯示目前輸入的數值
    st.subheader("目前的訂單特徵：")
    st.dataframe(input_data)

    # 預測按鈕
    if st.button("執行預測", type="primary"):
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ 模型預測結果：**此訂單高機率會被【取消】**")
            st.info("💡 商業建議：建議客服人員主動聯繫確認，或要求收取預付訂金。")
        else:
            st.success("✅ 模型預測結果：**此訂單將會順利【入住】**")
            st.info("💡 商業建議：這是一筆穩定訂單，可準備相關迎賓設施。")
            