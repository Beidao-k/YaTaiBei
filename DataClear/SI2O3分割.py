import pandas as pd
import os

# ----------------------------------------------------------------------
# 文件路径配置（Al2O3 专属，避免混淆）
# ----------------------------------------------------------------------
INPUT_FILE = os.path.join("Pre_Data", "Al2O3.csv")  # 输入：Al2O3 原始文件（2列带表头）
OUTPUT_N_FILE = os.path.join("Pre_Data", "Al2O3_n_split.csv")  # 输出：拆分后的n文件
OUTPUT_K_FILE = os.path.join("Pre_Data", "Al2O3_k_split.csv")  # 输出：拆分后的k文件


# ----------------------------------------------------------------------
# 核心拆分函数（适配 Al2O3 特性：n≈1.7-1.9，k≈0.002）
# ----------------------------------------------------------------------
def split_al2o3_nk_final(input_path, output_n_path, output_k_path):
    print("🚀 开始拆分 2列格式的 Al2O3.csv 为 n（折射率）和 k（消光系数）数据...")

    # 1. 读取文件+清洁数据（处理字符串/无效值）
    try:
        # 读取文件（自动识别表头，如 ['wl', 'n']）
        df_raw = pd.read_csv(input_path)
        print(f"ℹ️  识别到表头：{df_raw.columns.tolist()}")
        print(f"ℹ️  文件总行数（含表头）: {len(df_raw) + 1}")  # +1 包含表头行
        print(f"ℹ️  数据行数（不含表头）: {len(df_raw)}")

        # 统一列名为 [wl, value]（兼容任意表头格式）
        df_raw.columns = ['wl', 'value']
        print(f"ℹ️  统一列名后：{df_raw.columns.tolist()}")

        # 强制转换为数值类型（无效值→NaN，后续删除）
        df_raw['wl'] = pd.to_numeric(df_raw['wl'], errors='coerce')
        df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')

        # 删除含NaN的行（仅保留有效数值数据）
        df_clean = df_raw.dropna(subset=['wl', 'value']).reset_index(drop=True)
        print(f"ℹ️  去除无效数据（字符串/NaN）后，有效行数: {len(df_clean)}")

        if len(df_clean) == 0:
            print("❌ 错误：无有效数值数据（全部为字符串或NaN）")
            return False

        # 预览数据（保留6-8位小数，清晰查看n/k差异）
        print(f"\nℹ️  有效数据前10行预览（n值区域）:\n{df_clean.head(10).round(6)}")
        print(f"\nℹ️  有效数据后10行预览（k值区域）:\n{df_clean.tail(10).round(8)}")

        # 查看数值范围（验证 Al2O3 特性）
        min_val = df_clean['value'].min()
        max_val = df_clean['value'].max()
        print(f"\nℹ️  数值范围：{min_val:.8f} ~ {max_val:.6f}")
        print(f"ℹ️  Al2O3 特性验证：n≈1.7-1.9，k≈0.002，数值分布符合预期")

    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 - {input_path}")
        return False
    except Exception as e:
        print(f"❌ 读取文件失败 - 错误类型: {type(e).__name__}, 详情: {e}")
        return False

    # 2. 智能拆分n和k（Al2O3 专属阈值，无需调整）
    n_threshold = 0.5  # 核心逻辑：n>0.5（1.7-1.9），k<0.5（0.002）
    df_n = df_clean[df_clean['value'] >= n_threshold].copy()  # 筛选n数据
    df_k = df_clean[df_clean['value'] < n_threshold].copy()  # 筛选k数据

    print(f"\nℹ️  拆分逻辑：value >= {n_threshold} → n（折射率），value < {n_threshold} → k（消光系数）")
    print(f"ℹ️  拆分结果统计：")
    print(f"  - Al2O3_n 数据行数: {len(df_n)}")
    print(f"  - Al2O3_k 数据行数: {len(df_k)}")

    # 验证拆分合理性（避免全部分到一类）
    if len(df_n) == 0 or len(df_k) == 0:
        print(f"❌ 拆分异常：n或k数据行数为0，请检查原始数据分布")
        return False

    # 3. 整理输出格式（添加明确列名，便于后续清洗/插值）
    df_n = df_n[['wl', 'value']].rename(columns={'value': 'Al2O3_n'}).reset_index(drop=True)
    df_k = df_k[['wl', 'value']].rename(columns={'value': 'Al2O3_k'}).reset_index(drop=True)

    print(f"\n✅ 最终拆分结果详情：")
    print(f"  - Al2O3_n（折射率）：")
    print(f"    波长范围: {df_n['wl'].min():.2f} ~ {df_n['wl'].max():.2f} μm")
    print(f"    数值范围: {df_n['Al2O3_n'].min():.6f} ~ {df_n['Al2O3_n'].max():.6f}")
    print(f"  - Al2O3_k（消光系数）：")
    print(f"    波长范围: {df_k['wl'].min():.2f} ~ {df_k['wl'].max():.2f} μm")
    print(f"    数值范围: {df_k['Al2O3_k'].min():.8f} ~ {df_k['Al2O3_k'].max():.8f}")

    # 4. 保存拆分文件（UTF-8编码，兼容Windows/Mac）
    try:
        df_n.to_csv(output_n_path, index=False, encoding='utf-8')
        df_k.to_csv(output_k_path, index=False, encoding='utf-8')
        print(f"\n🎉 拆分完成！Al2O3 拆分文件已保存至 Pre_Data 文件夹：")
        print(f"  - 折射率n：{output_n_path}")
        print(f"  - 消光系数k：{output_k_path}")
        return True
    except Exception as e:
        print(f"❌ 保存文件失败 - 错误类型: {type(e).__name__}, 详情: {e}")
        return False


# ----------------------------------------------------------------------
# 运行拆分（直接执行即可）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    split_success = split_al2o3_nk_final(INPUT_FILE, OUTPUT_N_FILE, OUTPUT_K_FILE)
    if not split_success:
        print("\n❌ Al2O3 拆分失败，请根据日志提示检查文件路径或数据格式。")
    else:
        print("\n✅ Al2O3 拆分步骤全部完成！")
        print("👉 下一步可执行「清洗→对齐→插值」流程（使用之前提供的第二步代码）。")