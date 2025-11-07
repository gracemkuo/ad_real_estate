import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
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
    return df

#load_data("data/data.xlsx")