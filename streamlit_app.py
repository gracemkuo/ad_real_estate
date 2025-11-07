import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 讀取 xlsx 檔案
@st.cache_data
def load_data(file_path):
    """讀取並快取 xlsx 檔案"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"讀取檔案失敗: {e}")
        return None
    
try:
    df = load_data("data/data.xlsx")
    st.dataframe(df)
except FileNotFoundError:
        st.warning("檔案不存在，請上傳 Excel 檔案")
        # uploaded_file = st.file_uploader("上傳 Excel 檔案", type=["xlsx", "xls"])
    # if uploaded_file:
    #     df = pd.read_excel(uploaded_file)
    #     st.dataframe(df)