# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Raw Data - Abu Dhabi Real Estate",
    page_icon="📊",
    layout="wide",
)

"""
# 📊 Raw Data Explorer

View and interact with the cleaned real estate transaction data.
"""

# =========================
# 1) 資料讀取與基礎清理
# =========================
@st.cache_data(show_spinner=False)
def load_data(file_path: str):
    df = pd.read_excel(file_path)
    orig_len = len(df)
    print(f"📊 Raw rows: {orig_len:,}\n")

    # 基本文字正規化
    for c in ["Project", "Community", "District"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 日期
    df["Registration"] = pd.to_datetime(df["Registration"], format="%m/%d/%y", errors="coerce")

    # 數值欄位
    num_cols = ["Sold Area / GFA (sqm)", "Plot Area (sqm)", "Rate (AED/sqm)", "Price (AED)", "Share"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 便利欄位
    df["YearMonth"] = df["Registration"].dt.to_period("M").dt.to_timestamp()

    # --- 過濾無效列 ---
    # 移除沒日期
    before_len = len(df)
    df = df[df["Registration"].notna()].copy()
    deleted = before_len - len(df)
    if deleted > 0:
        print(f"❌ Step 1: Removed records without Registration date")
        print(f"   Deleted: {deleted:,} | Remaining: {len(df):,}\n")

    # 重要：先刪除 Price 或 Sold Area 任一為空的記錄（才能計算 Rate）
    if ("Price (AED)" in df.columns) and ("Sold Area / GFA (sqm)" in df.columns):
        before_len = len(df)
        df = df[df["Price (AED)"].notna() & df["Sold Area / GFA (sqm)"].notna()]
        deleted = before_len - len(df)
        if deleted > 0:
            print(f"❌ Step 2: Remove records with missing Price or Sold Area")
            print(f"   Deleted: {deleted:,} | Remaining: {len(df):,}\n")

    # Share 欄位過濾：只保留 Share = 1 (100%)
    if "Share" in df.columns:
        before_len = len(df)
        df = df[df["Share"] == 1]
        deleted = before_len - len(df)
        if deleted > 0:
            print(f"❌ Step 3: Keep only Share = 1 records")
            print(f"   Deleted: {deleted:,} | Remaining: {len(df):,}\n")

    # 移除非正數（<=0）的關鍵數值
    for c in ["Sold Area / GFA (sqm)", "Plot Area (sqm)", "Rate (AED/sqm)", "Price (AED)"]:
        if c in df.columns:
            before_len = len(df)
            df = df[(df[c].isna()) | (df[c] > 0)]
            deleted = before_len - len(df)
            if deleted > 0:
                print(f"❌ Step 4: Remove non-positive values in {c}")
                print(f"   Deleted: {deleted:,} | Remaining: {len(df):,}\n")

    # 移除 Project 屬於 Private 的列
    if "Project" in df.columns:
        before_len = len(df)
        df = df[~df["Project"].astype(str).str.contains("Private", case=False, na=False)]
        deleted = before_len - len(df)
        if deleted > 0:
            print(f"❌ Step 5: Remove rows where Project is Private")
            print(f"   Deleted: {deleted:,} | Remaining: {len(df):,}\n")

    # --- 新增：自動計算 Rate (AED/sqm) ---
    # 現在 Price 和 Sold Area 都保證有值，可以安心計算
    if "Price (AED)" in df.columns and "Sold Area / GFA (sqm)" in df.columns:
        df["Rate_Calculated"] = df["Price (AED)"] / df["Sold Area / GFA (sqm)"]

        # 新增驗證欄位：比較原始 Rate 和計算 Rate
        if "Rate (AED/sqm)" in df.columns:
            # 計算差異百分比
            df["Rate_Match"] = np.where(
                (df["Rate (AED/sqm)"].notna()) & (df["Rate_Calculated"].notna()),
                np.abs(df["Rate (AED/sqm)"] - df["Rate_Calculated"]) / df["Rate_Calculated"] < 0.01,  # 允許 1% 誤差
                False  # 只要其中一個是空或都是空，都視為不匹配
            )
            df["Rate_Difference"] = df["Rate (AED/sqm)"] - df["Rate_Calculated"]

            # 輸出不匹配的記錄統計
            mismatch_count = (~df["Rate_Match"]).sum()
            if mismatch_count > 0:
                print(f"⚠️  Warning: Rate validation")
                print(f"   Found {mismatch_count:,} records with mismatched Rate")
                print(f"   Mismatch rate: {mismatch_count/len(df)*100:.2f}%\n")
        else:
            # 如果原本沒有 Rate 欄位，用計算結果填入
            df["Rate (AED/sqm)"] = df["Rate_Calculated"]

    # --- 去極值 / 錯誤值（IQR 法）---
    def _remove_outliers_iqr(df_in: pd.DataFrame, group_col: str, value_col: str, k: float = 3.0) -> pd.DataFrame:
        # 僅對數值存在的列計算 IQR，NaN 保留
        sub = df_in[[group_col, value_col]].dropna()
        if sub.empty:
            return df_in
        q = sub.groupby(group_col)[value_col].quantile([0.25, 0.75]).unstack(level=-1)
        q.columns = ["q1", "q3"]
        q["iqr"] = q["q3"] - q["q1"]
        bounds = q.assign(
            lower=lambda x: x["q1"] - k * x["iqr"],
            upper=lambda x: x["q3"] + k * x["iqr"]
        )
        df_out = df_in.merge(bounds[["lower", "upper"]], left_on=group_col, right_index=True, how="left")
        mask_valid = (
            df_out[value_col].isna() |
            ((df_out[value_col] >= df_out["lower"]) & (df_out[value_col] <= df_out["upper"]))
        )
        df_out = df_out[mask_valid].drop(columns=["lower", "upper"])
        return df_out

    # 選擇可用的群組欄位：Project > Community > District
    _group_col = None
    for _c in ["Project", "Community", "District"]:
        if _c in df.columns:
            _group_col = _c
            break

    if _group_col:
        print(f"🔍 Step 6: Outlier removal (grouped by {_group_col}, k=3.0)")

        if "Rate (AED/sqm)" in df.columns:
            before_len = len(df)
            # Rate 的極值檢測
            # k=3.0 是比較寬鬆的設定，只會刪除非常明顯的異常值
            df = _remove_outliers_iqr(df, _group_col, "Rate (AED/sqm)", k=3.0)
            # 刪除每個 Project 中 Rate 異常高或異常低的記錄
            deleted = before_len - len(df)
            if deleted > 0:
                print(f"   ❌ Remove outliers in Rate (AED/sqm)")
                print(f"      Deleted: {deleted:,} | Remaining: {len(df):,}")

        if "Price (AED)" in df.columns:
            before_len = len(df)
            # Price 的極值檢測
            df = _remove_outliers_iqr(df, _group_col, "Price (AED)", k=3.0)
            # 刪除每個 Project 中 Price 異常高或異常低的記錄
            deleted = before_len - len(df)
            if deleted > 0:
                print(f"   ❌ Remove outliers in Price (AED)")
                print(f"      Deleted: {deleted:,} | Remaining: {len(df):,}")

        print()

    cleaned_len = len(df)
    total_deleted = orig_len - cleaned_len
    print("="*50)
    print(f"✅ Data cleaning completed")
    print(f"   Original: {orig_len:,} rows")
    print(f"   Deleted: {total_deleted:,} ({total_deleted/orig_len*100:.2f}%)")
    print(f"   Kept: {cleaned_len:,} ({cleaned_len/orig_len*100:.2f}%)")
    print("="*50)
    print()

    # 清理後輸出驗證結果
    if "Rate_Match" in df.columns:
        match_count = df["Rate_Match"].sum()
        match_rate = match_count / len(df) * 100 if len(df) > 0 else 0
        print(f"✓ Rate validation: {match_count:,} records matched (error < 1%, match rate {match_rate:.2f}%)\n")

    return df, {"orig_len": orig_len, "deleted": total_deleted, "cleaned_len": cleaned_len}

# 預設路徑可改
DATA_PATH = "data/data.xlsx"

with st.spinner("Loading and cleaning data..."):
    try:
        df, stats = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if df.empty:
        st.warning("Dataframe is empty. Please check the Excel file.")
        st.stop()

# =========================
# 2) 顯示數據統計
# =========================
st.markdown("### Data Cleaning Statistics")
if isinstance(stats, dict) and all(k in stats for k in ("orig_len", "deleted", "cleaned_len")) and stats["orig_len"]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", f"{stats['orig_len']:,}")
    with col2:
        st.metric("Deleted Rows", f"{stats['deleted']:,}",
                  delta_color="inverse")
    with col3:
        st.metric("Cleaned Rows", f"{stats['cleaned_len']:,}")

st.markdown("---")

# =========================
# 3) 過濾選項
# =========================
st.markdown("### Filter Options")

# 日期範圍過濾
min_date = df["Registration"].min().date()
max_date = df["Registration"].max().date()

date_range = st.date_input(
    "Registration Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

st.markdown("#### Filter by Dimensions")
col1, col2, col3 = st.columns(3)

with col1:
    # Project 過濾
    if "Project" in df.columns:
        selected_projects = st.multiselect(
            "Select Project(s)",
            options=sorted(df["Project"].dropna().unique().tolist()),
            default=None,
            help="Leave empty to show all projects"
        )
    else:
        selected_projects = []

with col2:
    # Community 過濾
    if "Community" in df.columns:
        selected_communities = st.multiselect(
            "Select Community(ies)",
            options=sorted(df["Community"].dropna().unique().tolist()),
            default=None,
            help="Leave empty to show all communities"
        )
    else:
        selected_communities = []

with col3:
    # District 過濾
    if "District" in df.columns:
        selected_districts = st.multiselect(
            "Select District(s)",
            options=sorted(df["District"].dropna().unique().tolist()),
            default=None,
            help="Leave empty to show all districts"
        )
    else:
        selected_districts = []

# =========================
# 4) 應用過濾
# =========================
filtered_df = df.copy()

# 日期過濾
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["Registration"].dt.date >= start_date) &
        (filtered_df["Registration"].dt.date <= end_date)
    ]

# Project 過濾
if selected_projects:
    filtered_df = filtered_df[filtered_df["Project"].isin(selected_projects)]

# Community 過濾
if selected_communities:
    filtered_df = filtered_df[filtered_df["Community"].isin(selected_communities)]

# District 過濾
if selected_districts:
    filtered_df = filtered_df[filtered_df["District"].isin(selected_districts)]

# =========================
# 5) 顯示過濾後的數據
# =========================
st.markdown(f"### Data Table ({len(filtered_df):,} rows)")

# 欄位選擇
all_columns = filtered_df.columns.tolist()
default_columns = [
    "Registration", "Project", "Community", "District",
    "Rate (AED/sqm)", "Price (AED)", "Sold Area / GFA (sqm)"
]
default_columns = [col for col in default_columns if col in all_columns]

selected_columns = st.multiselect(
    "Select columns to display",
    options=all_columns,
    default=default_columns
)

if selected_columns:
    display_df = filtered_df[selected_columns]
else:
    display_df = filtered_df

# 顯示數據表格 (可編輯和操作)
st.dataframe(
    display_df,
    width='stretch',
    height=600,
)

# =========================
# 6) 下載選項
# =========================
st.markdown("### Download Data")

col1, col2 = st.columns(2)

with col1:
    # 下載 CSV
    csv = display_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"abu_dhabi_real_estate_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with col2:
    # 下載 Excel
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        display_df.to_excel(writer, index=False, sheet_name='Data')
    excel_data = output.getvalue()

    st.download_button(
        label="Download as Excel",
        data=excel_data,
        file_name=f"abu_dhabi_real_estate_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# =========================
# 7) 數據摘要統計
# =========================
if st.checkbox("Show Summary Statistics", value=False):
    st.markdown("### Summary Statistics")

    numeric_cols = display_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(display_df[numeric_cols].describe(), width='stretch')
    else:
        st.info("No numeric columns to display statistics for.")
