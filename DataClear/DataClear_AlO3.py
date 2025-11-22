import pandas as pd
import os

# ----------------------------------------------------------------------
# 文件路径配置
# ----------------------------------------------------------------------
INPUT_FILE = os.path.join("Pre_Data", "Al2O3.csv")  # 输入文件
OUTPUT_N_FILE = os.path.join("Pre_Data", "Al2O3_n_split.csv")  # 拆分后的n文件
OUTPUT_K_FILE = os.path.join("Pre_Data", "Al2O3_k_split.csv")  # 拆分后的k文件


# ----------------------------------------------------------------------
# 核心拆分函数（修复字符串数据+格式化问题）
# ----------------------------------------------------------------------
def split_al2o3_nk_2col_fixed(input_path, output_n_path, output_k_path):
    print("🚀 开始拆分 2列格式的 Al2O3.csv 为 n 和 k 数据...")

    # 1. 读取文件+强制转换数值类型（跳过字符串）
    try:
        # 读取文件（保留原始数据类型）
        df_raw = pd.read_csv(input_path)
        print(f"ℹ️  识别到表头：{df_raw.columns.tolist()}")
        print(f"ℹ️  文件总行数（含表头）: {len(df_raw) + 1}")
        print(f"ℹ️  数据行数（不含表头）: {len(df_raw)}")

        # 统一列名，确保是 [wl, value]
        df_raw.columns = ['wl', 'value']
        print(f"ℹ️  统一列名后：{df_raw.columns.tolist()}")

        # 强制转换为数值类型（无效值转为NaN，后续删除）
        df_raw['wl'] = pd.to_numeric(df_raw['wl'], errors='coerce')
        df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')

        # 删除含NaN的行（去除字符串/无效数据）
        df_clean = df_raw.dropna(subset=['wl', 'value']).reset_index(drop=True)
        print(f"ℹ️  去除无效数据后，有效行数: {len(df_clean)}")

        if len(df_clean) == 0:
            print("❌ 错误：无有效数值数据（全部为字符串或NaN）")
            return False

        # 预览数据（修复格式化问题）
        print(f"ℹ️  有效数据前10行预览:\n{df_clean.head(10).round(6)}")
        print(f"ℹ️  有效数据后10行预览:\n{df_clean.tail(10).round(6)}")

        # 计算数值范围（用round避免格式化报错）
        min_val = df_clean['value'].min()
        max_val = df_clean['value'].max()
        print(f"ℹ️  数值范围：{min_val:.6f} ~ {max_val:.6f}")

    except Exception as e:
        print(f"❌ 读取文件失败 - 错误类型: {type(e).__name__}, 详情: {e}")
        return False

    # 2. 自动识别n和k的分界（Al2O3专属阈值，已验证适用）
    n_threshold = 0.5  # n>0.5，k<0.5（完全匹配你的数据：n≈1.7-1.9，k≈0.002）
    df_n = df_clean[df_clean['value'] >= n_threshold].copy()  # n数据
    df_k = df_clean[df_clean['value'] < n_threshold].copy()  # k数据

    print(f"\nℹ️  拆分逻辑：value >= {n_threshold} 视为n，value < {n_threshold} 视为k")
    print(f"ℹ️  拆分结果：")
    print(f"  - Al2O3_n 行数: {len(df_n)}")
    print(f"  - Al2O3_k 行数: {len(df_k)}")

    # 验证拆分合理性
    if len(df_n) == 0 or len(df_k) == 0:
        print(f"❌ 拆分异常：n/k数据行数为0，请检查阈值或数据分布")
        return False

    # 3. 整理输出格式（只保留需要的列）
    df_n = df_n[['wl', 'value']].rename(columns={'value': 'Al2O3_n'}).reset_index(drop=True)
    df_k = df_k[['wl', 'value']].rename(columns={'value': 'Al2O3_k'}).reset_index(drop=True)

    print(f"\n✅ 最终拆分结果：")
    print(f"  - Al2O3_n 波长范围: {df_n['wl'].min():.2f} ~ {df_n['wl'].max():.2f} μm")
    print(f"  - Al2O3_n 数值范围: {df_n['Al2O3_n'].min():.6f} ~ {df_n['Al2O3_n'].max():.6f}")
    print(f"  - Al2O3_k 波长范围: {df_k['wl'].min():.2f} ~ {df_k['wl'].max():.2f} μm")
    print(f"  - Al2O3_k 数值范围: {df_k['Al2O3_k'].min():.8f} ~ {df_k['Al2O3_k'].max():.8f}")

    # 4. 保存拆分文件
    try:
        df_n.to_csv(output_n_path, index=False, encoding='utf-8')
        df_k.to_csv(output_k_path, index=False, encoding='utf-8')
        print(f"\n🎉 拆分完成！文件已保存：")
        print(f"  - n数据：{output_n_path}")
        print(f"  - k数据：{output_k_path}")
        return True
    except Exception as e:
        print(f"❌ 保存文件失败 - 错误类型: {type(e).__name__}, 详情: {e}")
        return False


# ----------------------------------------------------------------------
# 运行拆分
# ----------------------------------------------------------------------
if __name__ == "__main__":
    success = split_al2o3_nk_2col_fixed(INPUT_FILE, OUTPUT_N_FILE, OUTPUT_K_FILE)
    if not success:
        print("\n❌ 拆分失败，请根据日志检查问题。")
    else:
        print("\n✅ 拆分步骤完成！已生成 Al2O3_n_split.csv 和 Al2O3_k_split.csv")
        print("👉 下一步将进行数据清洗和插值。")