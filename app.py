import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# 1. 設定網頁標題與佈局
st.set_page_config(page_title="飯店預訂預測與分析系統", layout="wide")

# 2. 快取載入資料的函數
@st.cache_data
def load_data():
    df = pd.read_csv('hotel_bookings.csv')
    # 基礎清洗
    df['children'] = df['children'].fillna(0)
    # 為了方便篩選，確保日期相關欄位正確
    return df

# 載入模型與資料
try:
    model = joblib.load('nb_model.pkl')
    df_raw = load_data()
except FileNotFoundError:
    st.error("找不到模型檔案 (nb_model.pkl) 或資料集 (hotel_bookings.csv)，請先執行訓練程式碼！")
    st.stop()

# --- 側邊欄：全域篩選器 ---
st.sidebar.header("📂 全域數據篩選")
hotel_choice = st.sidebar.multiselect("選擇飯店類型", options=df_raw['hotel'].unique(), default=df_raw['hotel'].unique())
cancel_choice = st.sidebar.selectbox("訂單狀態", ["全部", "僅限已入住 (0)", "僅限已取消 (1)"])

# 執行篩選邏輯
df_filtered = df_raw[df_raw['hotel'].isin(hotel_choice)]
if cancel_choice == "僅限已入住 (0)":
    df_filtered = df_filtered[df_filtered['is_canceled'] == 0]
elif cancel_choice == "僅限已取消 (1)":
    df_filtered = df_filtered[df_filtered['is_canceled'] == 1]

# --- 主畫面佈局 ---
tab1, tab2, tab3 = st.tabs(["🔍 數據探索 (Data Explorer)", "📊 統計圖表 (EDA)", "🤖 預測模型"])

# ================= 分頁 1：數據探索 =================
with tab1:
    st.title("飯店原始資料預覽與篩選")
    st.write(f"目前顯示：共有 **{len(df_filtered)}** 筆資料符合篩選條件")
    
    # 顯示指標 (Metric)
    m1, m2, m3 = st.columns(3)
    m1.metric("平均房價 (ADR)", f"${df_filtered['adr'].mean():.2f}")
    m2.metric("平均提前預訂天數", f"{int(df_filtered['lead_time'].mean())} 天")
    m3.metric("總特殊需求數", int(df_filtered['total_of_special_requests'].sum()))

    st.markdown("---")

    # ✨ 新增功能：資料特徵統計摘要（型態、缺失值、平均、標準差、極值）
    st.subheader("📊 資料集特徵與敘述性統計摘要")
    with st.expander("點擊 展開/收合 詳細資料特徵統計資訊（包含資料型態與缺失值）", expanded=True):
        
        # 1. 建立基礎摘要表 (資料型態與缺失值數量)
        summary_df = pd.DataFrame({
            '資料型態 (dtype)': df_filtered.dtypes.astype(str),
            '缺失值數量 (Missing Value)': df_filtered.isnull().sum()
        })
        
        # 2. 找出數值型的欄位，計算平均數、標準差、最小值、最大值
        numeric_cols = df_filtered.select_dtypes(include=['number']).columns
        
        summary_df.loc[numeric_cols, '平均數 (Mean)'] = df_filtered[numeric_cols].mean().round(2)
        summary_df.loc[numeric_cols, '標準差 (Std)'] = df_filtered[numeric_cols].std().round(2)
        summary_df.loc[numeric_cols, '最小值 (Min)'] = df_filtered[numeric_cols].min()
        summary_df.loc[numeric_cols, '最大值 (Max)'] = df_filtered[numeric_cols].max()
        
        # 3. 針對非數值型欄位（如文字、類別），將無法計算的統計量補上 '-' 符號以利閱讀
        summary_df = summary_df.fillna('-')
        
        # 4. 在網頁上呈現互動式表格
        st.dataframe(summary_df, use_container_width=True)

    st.markdown("---")

    # 顯示資料表格 (Dataframe)
    st.subheader("篩選後的資料明細")
    st.dataframe(df_filtered, height=400) # 使用者可以縮放、排序

    # 下載按鈕
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 下載篩選後的資料 (CSV)", data=csv, file_name='filtered_hotel_data.csv', mime='text/csv')

# ================= 分頁 2：統計圖表 =================
with tab2:
    st.title("資料視覺化分析")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("取消狀況比例")
        fig1 = px.pie(df_filtered, names='is_canceled', hole=0.4, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.subheader("前置天數 vs 取消狀態")
        fig2 = px.box(df_filtered, x="is_canceled", y="lead_time", color="is_canceled")
        st.plotly_chart(fig2, use_container_width=True)

# ================= 分頁 3：預測模型 =================
with tab3:
    st.title("🤖 預測該筆訂單是否會被取消")
    
    with st.expander("點擊輸入訂單資訊進行預測", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            lt = st.number_input("前置天數 (Lead Time)", value=30)
            sr = st.slider("特殊需求數量", 0, 5, 0)
            ps = st.selectbox("停車位需求", [0, 1, 2])
        with c2:
            bc = st.number_input("訂單變更次數", value=0)
            pc = st.number_input("過往取消次數", value=0)
            rg = st.radio("是否為回頭客", [0, 1])

    if st.button("執行模型預測", type="primary"):
        input_data = pd.DataFrame([[lt, sr, ps, bc, pc, rg]], 
                                  columns=['lead_time', 'total_of_special_requests', 'required_car_parking_spaces', 
                                           'booking_changes', 'previous_cancellations', 'is_repeated_guest'])
        res = model.predict(input_data)[0]
        if res == 1:
            st.error("🚨 預測結果：此訂單極大機率會被【取消】！")
        else:
            st.success("✨ 預測結果：此訂單將會順利【入住】。")