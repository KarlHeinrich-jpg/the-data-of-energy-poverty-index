# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd

# ===================== 1) 基本配置 =====================
DATA_PATH = "EP数据(2).xlsx"   # 改成你的文件路径（也支持csv）
AREA_COL = "AREA"
YEAR_COL = "YEAR"

INDICATORS = [
    "人均用电量（千瓦时/人）",
    "人均天然气消费量（立方米/人）",
    "城市天然气渗透率（%）",
    "城市人均天然气供应量（立方米）",
    "非火力发电占比",
    "国有经济电力、蒸汽、热水生产和供应业固定资产投资（亿元）",
    "城镇居民平均每百户电冰箱拥有量（台）",
    "城镇居民平均每百户空调拥有量（台）",
    "农村居民平均每百户抽油烟机拥有量（台）",
    "农村太阳能热水器人均覆盖面积（平方米）",
    "居民生活二氧化硫人均排放量（吨）",
]

# 方向：'-' 越大越不贫困（反向）；'+' 越大越贫困（正向）
DIRECTIONS = {
    "人均用电量（千瓦时/人）": "-",
    "人均天然气消费量（立方米/人）": "-",
    "城市天然气渗透率（%）": "-",
    "城市人均天然气供应量（立方米）": "-",
    "非火力发电占比": "-",
    "国有经济电力、蒸汽、热水生产和供应业固定资产投资（亿元）": "-",
    "城镇居民平均每百户电冰箱拥有量（台）": "-",
    "城镇居民平均每百户空调拥有量（台）": "-",
    "农村居民平均每百户抽油烟机拥有量（台）": "-",
    "农村太阳能热水器人均覆盖面积（平方米）": "-",
    "居民生活二氧化硫人均排放量（吨）": "+",
}

# 是否对“归一化后的Z”计算均值/标准差（默认False：对原始值做描述统计）
STATS_ON_NORMALIZED = False

# 标准差是否用样本标准差（ddof=1, 默认）还是总体标准差（ddof=0）
STD_DDOF = 1


# ===================== 2) 工具函数 =====================
def extract_unit(colname: str) -> str:
    """
    从列名里提取单位：支持中文括号（）和英文括号()
    例如：'人均用电量（千瓦时/人）' -> '千瓦时/人'
    """
    m = re.search(r"（([^）]+)）", colname)
    if m:
        return m.group(1).strip()
    m = re.search(r"\(([^)]+)\)", colname)
    if m:
        return m.group(1).strip()
    return ""


def minmax_normalize_poverty_oriented(df: pd.DataFrame, cols: list, directions: dict) -> pd.DataFrame:
    """
    最大最小归一化到[0,1]，并统一为“值越大=越贫困”方向：
      '+'：Z=(x-min)/(max-min)
      '-'：Z=(max-x)/(max-min)
    """
    Z = pd.DataFrame(index=df.index)
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        xmin, xmax = np.nanmin(x.values), np.nanmax(x.values)

        if np.isnan(xmin) or np.isnan(xmax) or np.isclose(xmax, xmin):
            Z[c] = 0.0
            continue

        if directions[c] == "+":
            Z[c] = (x - xmin) / (xmax - xmin)
        elif directions[c] == "-":
            Z[c] = (xmax - x) / (xmax - xmin)
        else:
            raise ValueError(f"Unknown direction for {c}: {directions[c]}")
    return Z


# ===================== 3) 元数据：中文表 + 英文表（翻译映射） =====================
# 你现在这11个指标的“Category/Variable/Indicator/Measurement”建议这样组织（可按你论文再微调）
META = [
    # ESA
    {
        "indicator_col": "人均用电量（千瓦时/人）",
        "Category_CN": "能源消费",
        "Variable_CN": "能源服务可及性（ESA）",
        "Indicator_CN": "人均用电量",
        "Measurement_CN": "千瓦时/人",
        "Category_EN": "Energy consumption",
        "Variable_EN": "Energy Service Accessibility (ESA)",
        "Indicator_EN": "Per capita electricity consumption",
        "Measurement_EN": "kWh/person",
    },
    {
        "indicator_col": "人均天然气消费量（立方米/人）",
        "Category_CN": "能源消费",
        "Variable_CN": "能源服务可及性（ESA）",
        "Indicator_CN": "人均天然气消费量",
        "Measurement_CN": "立方米/人",
        "Category_EN": "Energy consumption",
        "Variable_EN": "Energy Service Accessibility (ESA)",
        "Indicator_EN": "Per capita natural gas consumption",
        "Measurement_EN": "m³/person",
    },
    {
        "indicator_col": "城市天然气渗透率（%）",
        "Category_CN": "能源供给",
        "Variable_CN": "能源服务可及性（ESA）",
        "Indicator_CN": "城市天然气渗透率",
        "Measurement_CN": "%",
        "Category_EN": "Energy supply",
        "Variable_EN": "Energy Service Accessibility (ESA)",
        "Indicator_EN": "Urban natural gas penetration rate",
        "Measurement_EN": "%",
    },
    {
        "indicator_col": "城市人均天然气供应量（立方米）",
        "Category_CN": "能源供给",
        "Variable_CN": "能源服务可及性（ESA）",
        "Indicator_CN": "城市人均天然气供应量",
        # 你列名没写“/人”，但语义是人均；这里按你列名保留单位
        "Measurement_CN": "立方米",
        "Category_EN": "Energy supply",
        "Variable_EN": "Energy Service Accessibility (ESA)",
        "Indicator_EN": "Urban per capita natural gas supply",
        "Measurement_EN": "m³",
    },
    {
        "indicator_col": "国有经济电力、蒸汽、热水生产和供应业固定资产投资（亿元）",
        "Category_CN": "能源投资",
        "Variable_CN": "能源服务可及性（ESA）",
        "Indicator_CN": "国有能源行业固定资产投资",
        "Measurement_CN": "亿元",
        "Category_EN": "Energy investment",
        "Variable_EN": "Energy Service Accessibility (ESA)",
        "Indicator_EN": "Fixed asset investment in state-owned electricity/steam/hot water production and supply",
        "Measurement_EN": "100 million RMB",
    },

    # ECC
    {
        "indicator_col": "非火力发电占比",
        "Category_CN": "低碳能源结构",
        "Variable_CN": "能源消费清洁性（ECC）",
        "Indicator_CN": "非火力发电占比",
        "Measurement_CN": "%",  # 如果你数据是0-1比例，请改为“比例”
        "Category_EN": "Low-carbon energy structure",
        "Variable_EN": "Energy Consumption Cleanliness (ECC)",
        "Indicator_EN": "Share of non-thermal power generation",
        "Measurement_EN": "%",
    },
    {
        "indicator_col": "居民生活二氧化硫人均排放量（吨）",
        "Category_CN": "能源使用污染",
        "Variable_CN": "能源消费清洁性（ECC）",
        "Indicator_CN": "居民生活SO₂人均排放量",
        "Measurement_CN": "吨",
        "Category_EN": "Pollution from energy use",
        "Variable_EN": "Energy Consumption Cleanliness (ECC)",
        "Indicator_EN": "Per capita SO₂ emissions from residential activities",
        "Measurement_EN": "tons",
    },

    # HEA
    {
        "indicator_col": "城镇居民平均每百户电冰箱拥有量（台）",
        "Category_CN": "城镇能源设施",
        "Variable_CN": "家庭能源设施可得性（HEA）",
        "Indicator_CN": "城镇每百户电冰箱拥有量",
        "Measurement_CN": "台/百户",
        "Category_EN": "Urban energy facilities",
        "Variable_EN": "Household Energy Appliances (HEA)",
        "Indicator_EN": "Refrigerators per 100 urban households",
        "Measurement_EN": "units/100 households",
    },
    {
        "indicator_col": "城镇居民平均每百户空调拥有量（台）",
        "Category_CN": "城镇能源设施",
        "Variable_CN": "家庭能源设施可得性（HEA）",
        "Indicator_CN": "城镇每百户空调拥有量",
        "Measurement_CN": "台/百户",
        "Category_EN": "Urban energy facilities",
        "Variable_EN": "Household Energy Appliances (HEA)",
        "Indicator_EN": "Air conditioners per 100 urban households",
        "Measurement_EN": "units/100 households",
    },
    {
        "indicator_col": "农村居民平均每百户抽油烟机拥有量（台）",
        "Category_CN": "农村能源设施",
        "Variable_CN": "家庭能源设施可得性（HEA）",
        "Indicator_CN": "农村每百户抽油烟机拥有量",
        "Measurement_CN": "台/百户",
        "Category_EN": "Rural energy facilities",
        "Variable_EN": "Household Energy Appliances (HEA)",
        "Indicator_EN": "Range hoods per 100 rural households",
        "Measurement_EN": "units/100 households",
    },
    {
        "indicator_col": "农村太阳能热水器人均覆盖面积（平方米）",
        "Category_CN": "农村能源设施",
        "Variable_CN": "家庭能源设施可得性（HEA）",
        "Indicator_CN": "农村太阳能热水器人均覆盖面积",
        "Measurement_CN": "平方米",
        "Category_EN": "Rural energy facilities",
        "Variable_EN": "Household Energy Appliances (HEA)",
        "Indicator_EN": "Per capita coverage area of rural solar water heaters",
        "Measurement_EN": "m²",
    },
]


# ===================== 4) 生成“示例格式”的汇总表（CN+EN） =====================
def build_indicator_summary_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 检查列
    required = [AREA_COL, YEAR_COL] + INDICATORS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少以下列：{missing}")

    # 原始值转为数值
    X_raw = df[INDICATORS].apply(pd.to_numeric, errors="coerce")

    # 归一化（可选：如果你希望均值/标准差统计在Z上）
    Z = minmax_normalize_poverty_oriented(df, INDICATORS, DIRECTIONS)

    X_for_stats = Z if STATS_ON_NORMALIZED else X_raw

    # 计算均值/标准差（忽略缺失值）
    means = X_for_stats.mean(axis=0, skipna=True)
    stds = X_for_stats.std(axis=0, ddof=STD_DDOF, skipna=True)

    # 组装元数据表
    meta_df = pd.DataFrame(META)

    # 如果 Measurement_CN 没填，就从列名里自动抓（这里我们已经人工填了，仍保留自动兜底）
    meta_df["Measurement_CN"] = meta_df.apply(
        lambda r: r["Measurement_CN"] if str(r["Measurement_CN"]).strip() else extract_unit(r["indicator_col"]),
        axis=1
    )

    # 合并统计量
    stat_df = pd.DataFrame({
        "indicator_col": INDICATORS,
        "Average_Value": [means[c] for c in INDICATORS],
        "Standard_Deviation": [stds[c] for c in INDICATORS],
    })

    merged = meta_df.merge(stat_df, on="indicator_col", how="left")

    # 按你的指标顺序输出（不会排序打乱）
    merged["__order"] = merged["indicator_col"].apply(lambda x: INDICATORS.index(x))
    merged = merged.sort_values("__order", kind="stable").drop(columns="__order")

    # 中文表
    table_cn = merged[[
        "Category_CN", "Variable_CN", "Indicator_CN", "Measurement_CN",
        "Average_Value", "Standard_Deviation"
    ]].rename(columns={
        "Category_CN": "类别(Category)",
        "Variable_CN": "变量(Variable)",
        "Indicator_CN": "指标(Indicator)",
        "Measurement_CN": "度量(Measurement)",
        "Average_Value": "均值(Average Value)",
        "Standard_Deviation": "标准差(Standard Deviation)",
    })

    # 英文表
    table_en = merged[[
        "Category_EN", "Variable_EN", "Indicator_EN", "Measurement_EN",
        "Average_Value", "Standard_Deviation"
    ]].rename(columns={
        "Category_EN": "Category",
        "Variable_EN": "Variable",
        "Indicator_EN": "Indicator",
        "Measurement_EN": "Measurement",
        "Average_Value": "Average Value",
        "Standard_Deviation": "Standard Deviation",
    })

    return table_cn, table_en


def main():
    # 读取数据
    if DATA_PATH.lower().endswith(".csv"):
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    else:
        df = pd.read_excel(DATA_PATH)

    table_cn, table_en = build_indicator_summary_table(df)

    out_path = "Indicator_Summary_Table.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        table_cn.to_excel(writer, sheet_name="CN", index=False)
        table_en.to_excel(writer, sheet_name="EN", index=False)

    print(f"Done. Saved: {out_path}")
    print("\nCN preview:")
    print(table_cn.head(5))
    print("\nEN preview:")
    print(table_en.head(5))


if __name__ == "__main__":
    main()
