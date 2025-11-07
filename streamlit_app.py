# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime

# # 讀取 xlsx 檔案
# @st.cache_data
# def load_data(file_path):
#     """讀取並快取 xlsx 檔案"""
#     try:
#         df = pd.read_excel(file_path)
#         # 轉換 Registration 為日期格式
#         df['Registration'] = pd.to_datetime(df['Registration'], format='%m/%d/%y', errors='coerce')
#         return df
#     except Exception as e:
#         st.error(f"讀取檔案失敗: {e}")
#         return None
    
# try:
#     df = load_data("data/data.xlsx")
#     st.dataframe(df)
# except FileNotFoundError:
#         st.warning("檔案不存在，請更新 ad 房產資料檔案")

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Abu Dhabi Real Estate — Peer Analysis",
    page_icon=":house_buildings:",
    layout="wide",
)

# =========================
# 1) 資料讀取與基礎清理
# =========================
@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    # 日期
    df["Registration"] = pd.to_datetime(df["Registration"], format="%m/%d/%y", errors="coerce")
    # 數值欄位
    num_cols = ["Sold Area / GFA (sqm)", "Plot Area (sqm)", "Rate (AED/sqm)", "Price (AED)", "Share", "Sequence"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 便利欄位
    df["YearMonth"] = df["Registration"].dt.to_period("M").dt.to_timestamp()
    # 清掉沒日期或沒關鍵指標的
    return df

# 預設路徑可改
DATA_PATH = "data/data.xlsx"

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"讀取檔案失敗：{e}")
    st.stop()

if df.empty:
    st.warning("資料為空，請確認 Excel 檔內容。")
    st.stop()

# =========================
# 2) 左側控制面板
# =========================
left, right = st.columns([1, 3])
with left:
    st.markdown("### 資料來源概覽")
    st.caption(f"筆數：{len(df):,}，期間：{df['Registration'].min().date()} → {df['Registration'].max().date()}")

    group_dim = st.selectbox(
        "同儕群組維度",
        options=["Community", "Project", "District"],
        index=1
    )

    metric = st.selectbox(
        "分析指標",
        options=["Rate (AED/sqm)", "Price (AED)"],
        index=0,
        help="預設使用每平方公尺單價；也可切換為交易總價做對比。"
    )

    agg_fn_name = st.selectbox(
        "聚合方式",
        options=["median", "mean", "count"],
        index=0,
        help="每月對群組內多筆交易做聚合（常用 median 抗離群值）。"
    )

    freq = st.selectbox(
        "時間頻率",
        options=["Monthly", "Quarterly"],
        index=0
    )

    horizon_label = st.pills(
        "時間視窗",
        options=["3M", "6M", "1Y", "3Y", "5Y", "Max"],
        default="1Y",
    )

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
if horizon_label == "3M":
    start_date = end_date - pd.DateOffset(months=3)
elif horizon_label == "6M":
    start_date = end_date - pd.DateOffset(months=6)
elif horizon_label == "1Y":
    start_date = end_date - pd.DateOffset(years=1)
elif horizon_label == "3Y":
    start_date = end_date - pd.DateOffset(years=3)
elif horizon_label == "5Y":
    start_date = end_date - pd.DateOffset(years=5)
else:
    start_date = agg_ts["Date"].min()

agg_ts = agg_ts[(agg_ts["Date"] >= start_date) & (agg_ts["Date"] <= end_date)]

# =========================
# 4) 讓使用者挑群組（同儕集）
# =========================
with left:
    # 依期間內交易數量排序，便於挑熱門群組
    vol_rank = (df[(df["Registration"] >= start_date) & (df["Registration"] <= end_date)]
                  .groupby(group_dim)["Price (AED)"].count().sort_values(ascending=False))
    options = vol_rank.index.tolist()
    # 根據正則自動匹配 options 中的名稱
    import re

    # 關鍵字模式（不分大小寫）
    patterns = [
        r"saadiyat\s*park",
        r"saadiyat\s*grove",
        r"source",
        r"arthouse",
        r"louvre",
        r"canal",
        r"saas",
        r"mayan",
    ]

    # 根據正則自動匹配 options 中的名稱
    default_pick = [
        name for name in options
        if any(re.search(p, name, re.IGNORECASE) for p in patterns)
    ]

    picked_groups = st.multiselect(
        f"選擇要對比的 {group_dim}",
        options=options,
        default=default_pick,
        placeholder=f"輸入或選擇 {group_dim} 名稱"
    )

if not picked_groups:
    left.info("請至少選一個群組。")
    st.stop()

# =========================
# 5) 轉寬表、正規化（起點=1）、同儕平均
# =========================
pivot = agg_ts.pivot(index="Date", columns=group_dim, values=(metric if agg_fn_name != "count" else f"{metric} Count"))
pivot = pivot.sort_index()

# 僅保留使用者挑的群組
missing = [g for g in picked_groups if g not in pivot.columns]
picked_groups = [g for g in picked_groups if g in pivot.columns]

if len(picked_groups) == 0:
    st.error("選擇的群組在目前時間窗內沒有資料。")
    st.stop()
if missing:
    st.warning(f"以下群組在目前時間窗內無資料，已忽略：{', '.join(missing)}")

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

# =========================
# 6) 右側：總覽圖（正規化折線）
# =========================
with right:
    st.markdown("## 正規化走勢（起點=1）")
    chart_df = normalized.reset_index().melt(id_vars="Date", var_name=group_dim, value_name="Normalized")
    fig = px.line(
        chart_df, x="Date", y="Normalized", color=group_dim,
        height=420,
        hover_data={group_dim: True, "Normalized": ":.3f", "Date": "|%Y-%m-%d"},
    )
    fig.update_yaxes(title=None)
    fig.update_xaxes(title=None)
    st.plotly_chart(fig, use_container_width=True)

# 指標區塊（最佳/最差）
latest_vals = normalized.iloc[-1].dropna()
if not latest_vals.empty:
    best_name = latest_vals.idxmax()
    best_val = latest_vals.max()
    worst_name = latest_vals.idxmin()
    worst_val = latest_vals.min()

    with left:
        st.markdown("### 期間相對表現（起點=1）")
        c1, c2 = st.columns(2)
        c1.metric("最佳群組", best_name, delta=f"{round((best_val - 1) * 100)}%")
        c2.metric("最弱群組", worst_name, delta=f"{round((worst_val - 1) * 100)}%")

# =========================
# 7) 個別 vs 同儕平均 + Delta（minus peer average）
# =========================
if len(picked_groups) >= 2:
    st.markdown("## 個別群組 vs 同儕平均")
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
    st.info("要看 vs 同儕平均與 Delta，請至少選 2 個群組。")

# =========================
# 8) 原始/聚合資料
# =========================
# with st.expander("查看聚合後的時序資料", expanded=False):
#     st.dataframe(pivot, use_container_width=True)
with st.expander("查看原始資料（目前視窗篩選）", expanded=False):
    raw = df[(df["Registration"] >= start_date) & (df["Registration"] <= end_date)].copy()
    st.dataframe(raw, use_container_width=True)

# =========================
# 9) 小提示
# =========================
with left:
    st.caption("""
- 指標解讀：正規化=1 表示期間起點；最後值 1.25 ≈ 期間累計 +25%。
- 建議優先用 `Rate (AED/sqm)` + `median`，可減少豪宅極值對平均的干擾。
- 想做更嚴謹「同儕集」：可改為同區 `District` 或同產品型別 `Property Type` 的子集合。
""")