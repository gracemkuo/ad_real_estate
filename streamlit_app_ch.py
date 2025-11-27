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
    print(f"📊 原始資料：{orig_len:,} 筆\n")
    
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
        print(f"❌ 步驟 1：移除沒日期的記錄")
        print(f"   刪除：{deleted:,} 筆 | 剩餘：{len(df):,} 筆\n")

    # 重要：先刪除 Price 或 Sold Area 任一為空的記錄（才能計算 Rate）
    if ("Price (AED)" in df.columns) and ("Sold Area / GFA (sqm)" in df.columns):
        before_len = len(df)
        df = df[df["Price (AED)"].notna() & df["Sold Area / GFA (sqm)"].notna()]
        deleted = before_len - len(df)
        if deleted > 0:
            print(f"❌ 步驟 2：移除 Price 或 Sold Area 任一為空的記錄")
            print(f"   刪除：{deleted:,} 筆 | 剩餘：{len(df):,} 筆\n")
    
    # Share 欄位過濾：只保留 Share = 1 (100%)
    if "Share" in df.columns:
        before_len = len(df)
        df = df[df["Share"] == 1]
        deleted = before_len - len(df)
        if deleted > 0:
            print(f"❌ 步驟 3：只保留 Share = 1 的記錄")
            print(f"   刪除：{deleted:,} 筆 | 剩餘：{len(df):,} 筆\n")
        
    # 移除非正數（<=0）的關鍵數值
    for c in ["Sold Area / GFA (sqm)", "Plot Area (sqm)", "Rate (AED/sqm)", "Price (AED)"]:
        if c in df.columns:
            before_len = len(df)
            df = df[(df[c].isna()) | (df[c] > 0)]
            deleted = before_len - len(df)
            if deleted > 0:
                print(f"❌ 步驟 4：移除非正數的 {c}")
                print(f"   刪除：{deleted:,} 筆 | 剩餘：{len(df):,} 筆\n")
    
    # 移除 Project 屬於 Private 的列
    if "Project" in df.columns:
        before_len = len(df)
        df = df[~df["Project"].astype(str).str.contains("Private", case=False, na=False)]
        deleted = before_len - len(df)
        if deleted > 0:
            print(f"❌ 步驟 5：移除 Project 屬於 Private 的列")
            print(f"   刪除：{deleted:,} 筆 | 剩餘：{len(df):,} 筆\n")
    
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
                print(f"⚠️  警告：Rate 驗證")
                print(f"   發現 {mismatch_count:,} 筆 Rate 不匹配的記錄")
                print(f"   不匹配率：{mismatch_count/len(df)*100:.2f}%\n")
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
        print(f"🔍 步驟 6：去極值分析（使用 {_group_col} 分組，k=3.0）")
        
        if "Rate (AED/sqm)" in df.columns:
            before_len = len(df)
            # Rate 的極值檢測
            # k=3.0 是比較寬鬆的設定，只會刪除非常明顯的異常值
            df = _remove_outliers_iqr(df, _group_col, "Rate (AED/sqm)", k=3.0)
            # 刪除每個 Project 中 Rate 異常高或異常低的記錄
            deleted = before_len - len(df)
            if deleted > 0:
                print(f"   ❌ 移除 Rate (AED/sqm) 極值")
                print(f"      刪除：{deleted:,} 筆 | 剩餘：{len(df):,} 筆")
        
        if "Price (AED)" in df.columns:
            before_len = len(df)
            # Price 的極值檢測
            df = _remove_outliers_iqr(df, _group_col, "Price (AED)", k=3.0)
            # 刪除每個 Project 中 Price 異常高或異常低的記錄
            deleted = before_len - len(df)
            if deleted > 0:
                print(f"   ❌ 移除 Price (AED) 極值")
                print(f"      刪除：{deleted:,} 筆 | 剩餘：{len(df):,} 筆")
        
        print()
    
    cleaned_len = len(df)
    total_deleted = orig_len - cleaned_len
    print("="*50)
    print(f"✅ 資料清理完成")
    print(f"   原始：{orig_len:,} 筆")
    print(f"   刪除：{total_deleted:,} 筆（{total_deleted/orig_len*100:.2f}%）")
    print(f"   保留：{cleaned_len:,} 筆（{cleaned_len/orig_len*100:.2f}%）")
    print("="*50)
    print()
    
    # 清理後輸出驗證結果
    if "Rate_Match" in df.columns:
        match_count = df["Rate_Match"].sum()
        match_rate = match_count / len(df) * 100 if len(df) > 0 else 0
        print(f"✓ Rate 驗證結果：{match_count:,} 筆記錄匹配（誤差 < 1%，匹配率 {match_rate:.2f}%）\n")
    
    # 保存資料（確保有驗證欄位用於後續檢查）
    df.to_csv("data/processed_data.csv", index=False, encoding='utf-8-sig')
    return df, {"orig_len": orig_len, "deleted": total_deleted, "cleaned_len": cleaned_len}

# 預設路徑可改
DATA_PATH = "data/data.xlsx"

try:
    df, stats = load_data(DATA_PATH)
except Exception as e:
    st.error(f"讀取檔案失敗：{e}")
    st.stop()

if df.empty:
    st.warning("資料為空，請確認 Excel 檔內容。")
    st.stop()

# =========================
# 2) 頁面佈局：左側控制，右側結果
# =========================
left, right = st.columns([1, 3])

# 左側控制欄位（資料概覽與參數設定）
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
        options=["median", "mean"],
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
# 4) 右側頂部：選擇群組與指標顯示
# =========================
with right:
    st.markdown(f"### 選擇要對比的 {group_dim}")

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
    default_pick = [
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

    picked_groups = st.multiselect(
        f"選擇要對比的 {group_dim}",
        options=options,
        default=default_pick,
        placeholder=f"輸入或選擇 {group_dim} 名稱"
    )

    if not picked_groups:
        st.info("請至少選一個群組。")
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

    latest_vals = normalized.iloc[-1].dropna()
    if not latest_vals.empty:
        best_name = latest_vals.idxmax()
        best_val = latest_vals.max()
        worst_name = latest_vals.idxmin()
        worst_val = latest_vals.min()

        st.markdown("### 期間相對表現（起點=1）")
        c1, c2 = st.columns(2)
        c1.metric("最佳群組", best_name, delta=f"{round((best_val - 1) * 100)}%")
        c2.metric("最弱群組", worst_name, delta=f"{round((worst_val - 1) * 100)}%")

    st.caption("""
    - 指標解讀：正規化=1 表示期間起點；最後值 1.25 ≈ 期間累計 +25%。
    - 建議優先用 `Rate (AED/sqm)` + `median`，可減少豪宅極值對平均的干擾。
    - 想做更嚴謹「同儕集」：可改為同區 `District` 或同產品型別 `Property Type` 的子集合。
    """)

# =========================
# 6) 右側中段：總覽圖（正規化折線）
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
    st.plotly_chart(fig, width='stretch')

# =========================
# 7) 右側底部：個別 vs 同儕平均 + Delta（minus peer average）
# =========================
with right:
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
            grid_cols[(i * 2) % 4].plotly_chart(fig1, width='stretch')

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
            grid_cols[(i * 2 + 1) % 4].plotly_chart(fig2, width='stretch')
    else:
        st.info("要看 vs 同儕平均與 Delta，請至少選 2 個群組。")

# =========================
# 8) 原始/聚合資料
# =========================
# with st.expander("查看聚合後的時序資料", expanded=False):
#     st.dataframe(pivot, width='stretch')
with st.expander("查看原始資料（經清洗）", expanded=False):
    # 清理統計顯示
    if isinstance(stats, dict) and all(k in stats for k in ("orig_len", "deleted", "cleaned_len")) and stats["orig_len"]:
        st.code(
            f"""
               原始：{stats['orig_len']:,} 筆
               刪除：{stats['deleted']:,} 筆（{stats['deleted']/stats['orig_len']*100:.2f}%）
               保留：{stats['cleaned_len']:,} 筆（{stats['cleaned_len']/stats['orig_len']*100:.2f}%）
            """,
            language="text",
        )
    raw = df[(df["Registration"] >= start_date) & (df["Registration"] <= end_date)].copy()
    st.dataframe(raw, width='stretch')