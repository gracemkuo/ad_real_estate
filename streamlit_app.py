# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Abu Dhabi Real Estate — Peer Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
"""
# :material/query_stats: Abu Dhabi Real Estate Peer Analysis

Easily compare Project against others in their peer group.
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
    
    # 保存資料（確保有驗證欄位用於後續檢查）
    df.to_csv("data/processed_data.csv", index=False, encoding='utf-8-sig')
    return df, {"orig_len": orig_len, "deleted": total_deleted, "cleaned_len": cleaned_len}

# 預設路徑可改
DATA_PATH = "data/data.xlsx"

try:
    df, stats = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if df.empty:
    st.warning("Dataframe is empty. Please check the Excel file.")
    st.stop()

# =========================
# 2) 頁面佈局：左側控制，右側結果
# =========================
left, right = st.columns([1, 3])

# 左側控制欄位（資料概覽與參數設定）
with left:
    st.markdown("### Data overview")
    st.caption(f"Rows: {len(df):,}, Period: {df['Registration'].min().date()} → {df['Registration'].max().date()}, Update frequency: Biweekly")

    group_dim = st.selectbox(
        "Peer group dimension",
        options=["Community", "Project", "District"],
        index=1
    )

    metric_display = st.selectbox(
        "Metric",
        options=["Rate (AED/sqft)", "Rate (AED/sqm)"],
        index=0,
        help="Default: price per sqft; switch to sqm for comparison."
    )

    # 映射到實際數據欄位 (底層都使用 sqm)
    metric = "Rate (AED/sqm)"
    # 記錄是否需要單位轉換
    convert_to_sqft = (metric_display == "Rate (AED/sqft)")

    agg_fn_name = st.selectbox(
        "Aggregation",
        options=["median", "mean"],
        index=0,
        help="Aggregate monthly transactions per group (median is more robust to outliers)."
    )

    freq = st.selectbox(
        "Time frequency",
        options=["Monthly", "Quarterly"],
        index=0
    )

    horizon_label = st.pills(
        "Time window",
        options=["1M", "3M", "6M", "1Y", "3Y", "5Y", "Max"],
        default="1Y",
    )

    # 自定義日期選擇器 (始終顯示)
    #st.markdown("#### Custom date range")
    col1, col2 = st.columns(2)
    with col1:
        custom_start = st.date_input("Start date", value=None)
    with col2:
        custom_end = st.date_input("End date", value=None)

# =========================
# 3) 時間過濾與頻率轉換
# =========================
def to_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # 先依群組+月份聚合，再做頻率上卷（M / Q）
    base = (
        df.dropna(subset=["Registration", group_dim, metric])
          .groupby([group_dim, "YearMonth"])[metric]
    )

    if agg_fn_name == "median":
        g = base.median().reset_index()
    elif agg_fn_name == "mean":
        g = base.mean().reset_index()
    elif agg_fn_name == "count":
        g = base.count().reset_index().rename(columns={metric: f"{metric} Count"})
    else:
        g = base.median().reset_index()

    # 轉頻率
    if freq == "Monthly":
        g = g.rename(columns={"YearMonth": "Date"})
    else:  # Quarterly
        g["Date"] = g["YearMonth"].dt.to_period("Q").dt.to_timestamp(how="end")
        g = g.groupby([group_dim, "Date"]).agg({metric: "median" if agg_fn_name != "count" else "sum"}).reset_index()

    return g

agg_ts = to_freq(df, freq)

# 時間視窗計算
end_date = agg_ts["Date"].max()

# 優先使用自定義日期(如果兩個日期都有選擇)
if custom_start is not None and custom_end is not None:
    start_date = pd.Timestamp(custom_start)
    end_date = pd.Timestamp(custom_end)
# 否則使用 pills 選擇的時間範圍
elif horizon_label == "1M":
    start_date = end_date - pd.DateOffset(months=1)
elif horizon_label == "3M":
    start_date = end_date - pd.DateOffset(months=3)
elif horizon_label == "6M":
    start_date = end_date - pd.DateOffset(months=6)
elif horizon_label == "1Y":
    start_date = end_date - pd.DateOffset(years=1)
elif horizon_label == "3Y":
    start_date = end_date - pd.DateOffset(years=3)
elif horizon_label == "5Y":
    start_date = end_date - pd.DateOffset(years=5)
else:  # Max
    start_date = agg_ts["Date"].min()

agg_ts = agg_ts[(agg_ts["Date"] >= start_date) & (agg_ts["Date"] <= end_date)]

# 如果用戶選擇 sqft,轉換單位 (1 sqm = 10.764 sqft)
if convert_to_sqft:
    agg_ts[metric] = agg_ts[metric] / 10.764

# =========================
# 4) 右側頂部：選擇群組與指標顯示
# =========================
with right:
    st.markdown(f"### Select {group_dim} to compare")

    # 依期間內群組名稱字母排序，便於挑選
    sub_df = df[(df["Registration"] >= start_date) & (df["Registration"] <= end_date)]
    options = (
        sub_df[group_dim]
          .dropna()
          .astype(str)
          .unique()
          .tolist()
    )
    # Alphabetical sort (case-insensitive)
    options = sorted(options, key=lambda s: s.lower())
    #import re

    # 關鍵字模式（不分大小寫）
    # 關鍵字列表（不分大小寫匹配）
    default_pick_candidates = [
        "Park View Residence, Al Saadiyat Island",
        "Saadiyat Grove - The Source Residences",
        "Saadiyat Grove - The Source Terraces",
        "Saadiyat Grove - The Source",
        "Saadiyat Grove - The Arthouse",
        "Louvre Residences",
        "Canal Residence",
        "SAAS Heights",
        "Mayan",
    ]

    # 根據正則自動匹配 options 中的名稱
            # default_pick = [
            #     name for name in options
            #     if any(re.search(p, name, re.IGNORECASE) for p in patterns)
            # ]

    # 只保留在當前時間範圍內存在的預設選項
    default_pick = [item for item in default_pick_candidates if item in options]

    picked_groups = st.multiselect(
        f"Pick {group_dim} to compare",
        options=options,
        default=default_pick,
        placeholder=f"Type or select a {group_dim} name"
    )

    if not picked_groups:
        st.info("Please pick at least one group.")
        st.stop()

# =========================
# 5) 右側頂部：期間相對表現（起點=1）指標與說明文字
# =========================
with right:
    # 轉寬表、正規化（起點=1）、同儕平均
    pivot = agg_ts.pivot(index="Date", columns=group_dim, values=(metric if agg_fn_name != "count" else f"{metric} Count"))
    pivot = pivot.sort_index()

    # 僅保留使用者挑的群組
    missing = [g for g in picked_groups if g not in pivot.columns]
    picked_groups = [g for g in picked_groups if g in pivot.columns]

    if len(picked_groups) == 0:
        st.error("The selected groups have no data in the current window.")
        st.stop()
    if missing:
        st.warning(f"No data for the following groups in the current window; ignored: {', '.join(missing)}")

    sub = pivot[picked_groups].dropna(how="all")
    # 去掉全是 NaN 的列
    sub = sub.dropna(axis=0, how="all")

    # 正規化（各群組在期間第一個非空值為 1）
    def normalize_df(df_wide: pd.DataFrame) -> pd.DataFrame:
        norm = df_wide.copy()
        for c in norm.columns:
            series = norm[c].dropna()
            if series.empty:
                norm[c] = np.nan
            else:
                first = series.iloc[0]
                norm[c] = norm[c] / first
        return norm

    normalized = normalize_df(sub)

    latest_vals = normalized.iloc[-1].dropna()
    if not latest_vals.empty:
        best_name = latest_vals.idxmax()
        best_val = latest_vals.max()
        worst_name = latest_vals.idxmin()
        worst_val = latest_vals.min()

        st.markdown("### Relative performance over window (base=1)")
        c1, c2 = st.columns(2)
        c1.metric("Top group", best_name, delta=f"{round((best_val - 1) * 100)}%")
        c2.metric("Weakest group", worst_name, delta=f"{round((worst_val - 1) * 100)}%")

    st.caption("""
    - How to read: normalization=1 is window start; final value 1.25 ≈ +25% over the window.
    - Tip: prefer `Rate (AED/sqft)` or `Rate (AED/sqm)` with `median` to reduce luxury outlier skew.
    - For a stricter peer set: filter by the same `District` or `Property Type`.
    """)

# =========================
# 6) 右側中段：總覽圖（正規化折線）
# =========================
with right:
    st.markdown("## Normalized trend (base=1)")
    chart_df = normalized.reset_index().melt(id_vars="Date", var_name=group_dim, value_name="Normalized")
    fig = px.line(
        chart_df, x="Date", y="Normalized", color=group_dim,
        height=420,
        hover_data={group_dim: True, "Normalized": ":.3f", "Date": "|%Y-%m-%d"},
    )
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 7) 右側底部：個別 vs 同儕平均 + Delta（minus peer average）
# =========================
with right:
    if len(picked_groups) >= 2:
        st.markdown("## Each group vs peer average")
        grid_cols = st.columns(4)

        for i, gname in enumerate(picked_groups):
            peers = normalized.drop(columns=[gname])
            peer_avg = peers.mean(axis=1)

            # (a) 該群組 vs 同儕平均
            comp_df = pd.DataFrame({
                "Date": normalized.index,
                gname: normalized[gname],
                "Peer average": peer_avg
            })
            comp_df = comp_df.melt(id_vars="Date", var_name="Series", value_name="Value")

            fig1 = px.line(
                comp_df, x="Date", y="Value", color="Series",
                height=300,
                title=f"{gname} vs Peer average",
                hover_data={"Value": ":.3f", "Date": "|%Y-%m-%d"},
                color_discrete_map={gname: "red", "Peer average": "gray"} 
            )
            fig1.update_yaxes(title=None, rangemode="tozero")
            fig1.update_xaxes(title=None)
            grid_cols[(i * 2) % 4].plotly_chart(fig1, use_container_width=True)

            # (b) Delta：該群組 - 同儕平均
            delta_df = pd.DataFrame({
                "Date": normalized.index,
                "Delta": normalized[gname] - peer_avg
            }).dropna()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=delta_df["Date"], y=delta_df["Delta"],
                mode="lines", fill="tozeroy", name="Delta"
            ))
            fig2.update_layout(
                title=f"{gname} minus Peer average",
                height=300, showlegend=False, margin=dict(l=10, r=10, t=40, b=10)
            )
            fig2.update_yaxes(zeroline=True, zerolinewidth=1)
            grid_cols[(i * 2 + 1) % 4].plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Select at least 2 groups to view vs peer average and delta.")

# =========================
# 8) 原始/聚合資料
# =========================
# with st.expander("查看聚合後的時序資料", expanded=False):
#     st.dataframe(pivot, use_container_width=True)
with st.expander("View raw data (cleaned)", expanded=False):
    # 清理統計顯示
    if isinstance(stats, dict) and all(k in stats for k in ("orig_len", "deleted", "cleaned_len")) and stats["orig_len"]:
        st.code(
            f"""
               Original: {stats['orig_len']:,} rows
               Deleted: {stats['deleted']:,} ({stats['deleted']/stats['orig_len']*100:.2f}%)
               Kept: {stats['cleaned_len']:,} ({stats['cleaned_len']/stats['orig_len']*100:.2f}%)
            """,
            language="text",
        )
    raw = df[(df["Registration"] >= start_date) & (df["Registration"] <= end_date)].copy()
    st.dataframe(raw, use_container_width=True)