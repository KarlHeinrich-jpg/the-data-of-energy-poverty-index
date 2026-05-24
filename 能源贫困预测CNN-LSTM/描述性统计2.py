# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# ===================== 1) 文件路径 =====================
DATA_PATH = "EPI_entropy_output.xlsx"   # 改成你的文件路径（也支持csv）

# ===================== 2) 自动识别地区/年份列（中文导入也OK） =====================
AREA_CANDIDATES = ["AREA", "地区", "省份", "区域"]
YEAR_CANDIDATES = ["YEAR", "年份", "year"]

# 标准差：默认样本标准差(ddof=1)；若你要总体标准差改成0
STD_DDOF = 1

# 是否额外导出“审计统计表”（含方差、CV）用于核对
EXPORT_AUDIT_STATS = True

def pick_col(df, candidates, name_for_error):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"找不到{name_for_error}列，候选：{candidates}\n请把真实列名加入候选列表。")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

# ===================== 3) 你要的“变量表”（按你给的格式与顺序） =====================
VAR_TABLE_ROWS = [
    # 经济因子
    {"因子类别": "经济因子", "Variable": "GDP per capita", "单位": "CNY/person"},
    {"因子类别": "",       "Variable": "Household consumption", "单位": "100 million CNY"},
    {"因子类别": "",       "Variable": "Share of secondary industry", "单位": "%"},
    # 人口因子
    {"因子类别": "人口因子", "Variable": "Population density", "单位": "persons/km²"},
    {"因子类别": "",       "Variable": "Urbanization rate", "单位": "%"},
    # 能源因子
    {"因子类别": "能源因子", "Variable": "Energy intensity", "单位": "ton /10⁴ CNY"},
    {"因子类别": "",       "Variable": "Energy consumption share", "单位": "%"},
    {"因子类别": "",       "Variable": "Power generation per capita", "单位": "kWh/person"},
    # 环境因子
    {"因子类别": "环境因子", "Variable": "Forest coverage rate", "单位": "%"},
    {"因子类别": "",       "Variable": "Local fiscal expenditure on environmental protection", "单位": "100 million CNY"},
    {"因子类别": "",       "Variable": "Water resources per capita", "单位": "m³/person"},
]

# ===================== 4) 每个外生变量的“中文列名候选”与“计算方法” =====================
CANDIDATES = {
    # 经济
    "GDP per capita": ["人均地区生产总值(元/人)", "人均地区生产总值（元/人）", "人均GDP(元/人)"],
    "Household consumption": ["居民消费(亿元)", "居民消费（亿元）"],
    "GDP_total": ["地区生产总值(亿元)", "地区生产总值（亿元）", "GDP(亿元)"],
    "Secondary_value": ["第二产业增加值(亿元)", "第二产业增加值（亿元）"],
    "Share of secondary industry": ["第二产业占比", "第二产业占比(%)"],

    # 人口
    "Population density": ["人口密度(人/平方公里)", "人口密度", "城市人口密度(人/平方公里)"],
    "Urban_pop": ["城镇人口(万人)", "城镇人口（万人）"],
    "Pop_total": ["年末常住人口(万人)", "年末常住人口（万人）"],
    "Urbanization rate": ["城镇人口占比", "城镇人口占比(%)", "城镇化率", "城镇化率(%)"],

    # 能源
    "Energy intensity": ["能源消费强度", "单位地区生产总值能耗(等价值)(吨标准煤/万元)"],
    "Energy consumption share": ["能源消费比", "能源消费占比", "能源消费占比(%)", "能源消费比(%)"],

    "Power_gen": ["发电量(亿千瓦小时)", "发电量(亿千瓦时)", "发电量（亿千瓦时）", "发电量（亿千瓦小时）"],
    "Power generation per capita": ["人均发电量", "人均发电量(kWh/person)"],

    # 环境
    "Forest coverage rate": ["森林覆盖率(%)", "森林覆盖率（%）"],
    "Local fiscal expenditure on environmental protection": ["地方财政环境保护支出(亿元)", "地方财政环境保护支出（亿元）"],
    "Water resources per capita": ["人均水资源量(立方米/人)", "人均水资源量（立方米/人）"],
}

def first_existing(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

# ===================== 5) 构建外生变量数据（保持原始顺序不变） =====================
def build_exog_dataset(df: pd.DataFrame, area_col: str, year_col: str):
    out = df[[area_col, year_col]].copy()  # 不排序，保持原顺序

    # GDP per capita
    col = first_existing(df, CANDIDATES["GDP per capita"])
    if col is None:
        raise ValueError("缺少列：人均地区生产总值(元/人)（请在CANDIDATES里补充真实列名）")
    out["GDP per capita"] = to_num(df[col])

    # Household consumption
    col = first_existing(df, CANDIDATES["Household consumption"])
    if col is None:
        raise ValueError("缺少列：居民消费(亿元)（请在CANDIDATES里补充真实列名）")
    out["Household consumption"] = to_num(df[col])

    # Share of secondary industry (direct or computed)
    direct = first_existing(df, CANDIDATES["Share of secondary industry"])
    if direct is not None:
        out["Share of secondary industry"] = to_num(df[direct])
    else:
        gdp_col = first_existing(df, CANDIDATES["GDP_total"])
        sec_col = first_existing(df, CANDIDATES["Secondary_value"])
        if gdp_col is None or sec_col is None:
            raise ValueError("缺少计算第二产业占比所需列：地区生产总值(亿元) 或 第二产业增加值(亿元)")
        gdp = to_num(df[gdp_col])
        sec = to_num(df[sec_col])
        out["Share of secondary industry"] = (sec / gdp) * 100.0

    # Population density
    dens_col = first_existing(df, CANDIDATES["Population density"])
    if dens_col is None:
        raise ValueError("缺少列：人口密度（请提供人口密度列，或提供面积以便计算）")
    out["Population density"] = to_num(df[dens_col])

    # Urbanization rate (direct or computed)
    direct = first_existing(df, CANDIDATES["Urbanization rate"])
    if direct is not None:
        out["Urbanization rate"] = to_num(df[direct])
    else:
        up = first_existing(df, CANDIDATES["Urban_pop"])
        tp = first_existing(df, CANDIDATES["Pop_total"])
        if up is None or tp is None:
            raise ValueError("缺少计算城镇化率所需列：城镇人口(万人) 或 年末常住人口(万人)")
        out["Urbanization rate"] = (to_num(df[up]) / to_num(df[tp])) * 100.0

    # Energy intensity
    ei_col = first_existing(df, CANDIDATES["Energy intensity"])
    if ei_col is None:
        raise ValueError("缺少列：能源消费强度/单位GDP能耗（请在CANDIDATES里补充真实列名）")
    out["Energy intensity"] = to_num(df[ei_col])

    # Energy consumption share (must exist)
    ecs_col = first_existing(df, CANDIDATES["Energy consumption share"])
    if ecs_col is None:
        raise ValueError(
            "缺少列：能源消费比/能源消费占比。\n"
            "请在原始数据中提供该列，或把真实列名加入CANDIDATES['Energy consumption share']。"
        )
    out["Energy consumption share"] = to_num(df[ecs_col])

    # Power generation per capita (direct or computed)
    direct = first_existing(df, CANDIDATES["Power generation per capita"])
    if direct is not None:
        out["Power generation per capita"] = to_num(df[direct])
    else:
        pg = first_existing(df, CANDIDATES["Power_gen"])
        tp = first_existing(df, CANDIDATES["Pop_total"])
        if pg is None or tp is None:
            raise ValueError("缺少计算人均发电量所需列：发电量(亿千瓦时) 或 年末常住人口(万人)")
        # 人均发电量 = 发电量(亿kWh)*1e8 / 人口(万人)*1e4 = 发电量*10000/人口
        out["Power generation per capita"] = to_num(df[pg]) * 10000.0 / to_num(df[tp])

    # Forest coverage rate
    col = first_existing(df, CANDIDATES["Forest coverage rate"])
    if col is None:
        raise ValueError("缺少列：森林覆盖率(%)")
    out["Forest coverage rate"] = to_num(df[col])

    # Local fiscal expenditure on environmental protection
    col = first_existing(df, CANDIDATES["Local fiscal expenditure on environmental protection"])
    if col is None:
        raise ValueError("缺少列：地方财政环境保护支出(亿元)")
    out["Local fiscal expenditure on environmental protection"] = to_num(df[col])

    # Water resources per capita
    col = first_existing(df, CANDIDATES["Water resources per capita"])
    if col is None:
        raise ValueError("缺少列：人均水资源量(立方米/人)")
    out["Water resources per capita"] = to_num(df[col])

    return out

# ===================== 6) 主程序：读取 -> 生成变量表(含均值/标准差) -> 导出 =====================
def main():
    if DATA_PATH.lower().endswith(".csv"):
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    else:
        df = pd.read_excel(DATA_PATH)

    area_col = pick_col(df, AREA_CANDIDATES, "地区(AREA)")
    year_col = pick_col(df, YEAR_CANDIDATES, "年份(YEAR)")

    # 外生变量数据（英文列名，保持原始顺序）
    exog_en = build_exog_dataset(df, area_col, year_col)

    # ===== 计算均值与标准差（基于 exog_en 的数值列）=====
    var_list = [r["Variable"] for r in VAR_TABLE_ROWS]  # 按表格顺序
    stats = []
    audit_stats = []

    for v in var_list:
        x = pd.to_numeric(exog_en[v], errors="coerce")
        mean_val = x.mean(skipna=True)
        std_val = x.std(skipna=True, ddof=STD_DDOF)   # ✅ 修正：这里用 std，不是 var
        var_val = x.var(skipna=True, ddof=STD_DDOF)   # 仅用于审计核对（可不输出到主表）

        # 变异系数（便于快速发现异常；均值接近0时设为NaN）
        if pd.isna(mean_val) or np.isclose(mean_val, 0.0):
            cv_val = np.nan
        else:
            cv_val = std_val / mean_val

        stats.append({
            "Variable": v,
            "均值(Mean)": mean_val,
            "标准差(Standard Deviation)": std_val,   # ✅ 主表输出标准差
        })

        audit_stats.append({
            "Variable": v,
            "均值(Mean)": mean_val,
            "标准差(Standard Deviation)": std_val,
            "方差(Variance)": var_val,
            "变异系数(CV=SD/Mean)": cv_val,
            "有效样本数(N)": x.notna().sum(),
            "最小值(Min)": x.min(skipna=True),
            "最大值(Max)": x.max(skipna=True),
        })

    stats_df = pd.DataFrame(stats)
    audit_stats_df = pd.DataFrame(audit_stats)

    # ===== Variable_Table：合并均值/标准差 =====
    var_table = pd.DataFrame(VAR_TABLE_ROWS, columns=["因子类别", "Variable", "单位"])
    var_table = var_table.merge(stats_df, on="Variable", how="left")

    # 中文列名版（便于核对）
    cn_map = {
        "GDP per capita": "人均地区生产总值(元/人)",
        "Household consumption": "居民消费(亿元)",
        "Share of secondary industry": "第二产业占比(%)",
        "Population density": "人口密度(persons/km²)",
        "Urbanization rate": "城镇人口占比(%)",
        "Energy intensity": "能源消费强度(ton/10⁴ CNY)",
        "Energy consumption share": "能源消费比(%)",
        "Power generation per capita": "人均发电量(kWh/person)",
        "Forest coverage rate": "森林覆盖率(%)",
        "Local fiscal expenditure on environmental protection": "地方财政环境保护支出(亿元)",
        "Water resources per capita": "人均水资源量(m³/person)",
    }
    exog_cn = exog_en.rename(columns=cn_map)

    # 导出（不排序，不重排）
    out_path = "Exogenous_Variables_and_Table.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        var_table.to_excel(writer, sheet_name="Variable_Table", index=False)   # 主表：均值+标准差
        exog_en.to_excel(writer, sheet_name="Exog_Data_EN", index=False)
        exog_cn.to_excel(writer, sheet_name="Exog_Data_CN", index=False)

        # 可选：导出审计表，便于你核查和写回复
        if EXPORT_AUDIT_STATS:
            audit_stats_df.to_excel(writer, sheet_name="Stats_Audit", index=False)

    print(f"Done. Saved: {out_path}")
    print("\nVariable_Table preview (Mean + Standard Deviation):")
    print(var_table)

    if EXPORT_AUDIT_STATS:
        print("\nStats_Audit preview:")
        print(audit_stats_df)

if __name__ == "__main__":
    main()