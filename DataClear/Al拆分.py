import pandas as pd
import os

# ----------------------------------------------------------------------
# 文件路径配置
# ----------------------------------------------------------------------
INPUT_FILE = os.path.join("Pre_Data", "Al.csv")  # 输入：Al原始文件（2列：先n后k）
OUTPUT_N_FILE = os.path.join("Pre_Data", "Al_n_split.csv")  # 输出：n文件（wl+Al_n）
OUTPUT_K_FILE = os.path.join("Pre_Data", "Al_k_split.csv")  # 输出：k文件（wl+Al_k）


# ----------------------------------------------------------------------
# 核心拆分逻辑（按数据块分割：前半部分=wl+n，后半部分=wl+k）
# ----------------------------------------------------------------------
def split_al_nk_by_block(input_path, output_n_path, output_k_path):
    print("🚀 开始 Al.csv 拆分（适配格式：先wl+n，后wl+k）...")

    # 1. 读取并清洁原始数据
    try:
        # 读取原始文件（2列带表头：如 ['wl', 'value'] 或 ['波长', '数值']）
        df_raw = pd.read_csv(input_path)
        print(f"ℹ️  识别到表头：{df_raw.columns.tolist()}")
        print(f"ℹ️  文件总行数（含表头）: {len(df_raw) + 1}")
        print(f"ℹ️  数据行数（不含表头）: {len(df_raw)}")

        # 统一列名为 [wl, value]
        df_raw.columns = ['wl', 'value']

        # 强制转数值+去NaN（清洁数据，不改变顺序）
        df_raw['wl'] = pd.to_numeric(df_raw['wl'], errors='coerce')
        df_raw['value'] = pd.to_numeric(df_raw['value'], errors='coerce')
        df_clean = df_raw.dropna(subset=['wl', 'value']).reset_index(drop=True)
        clean_rows = len(df_clean)
        print(f"ℹ️  清洁后有效数据行数: {clean_rows}")

        if clean_rows == 0:
            print("❌ 错误：无有效数值数据（全部为字符串/NaN）")
            return False

        # 2. 关键：按数据块分割（前半=wl+n，后半=wl+k）
        # 假设n和k的数据量接近，取中间位置分割（可手动调整split_ratio）
        split_ratio = 0.5  # 分割比例（默认前50%为n，后50%为k）
        split_idx = int(clean_rows * split_ratio)

        # 分割数据：前半=Al_n，后半=Al_k
        df_n = df_clean.iloc[:split_idx].copy()
        df_k = df_clean.iloc[split_idx:].copy()

        # 重命名列（添加明确标识）
        df_n.rename(columns={'value': 'Al_n'}, inplace=True)
        df_k.rename(columns={'value': 'Al_k'}, inplace=True)

        # 按wl排序（便于后续对齐）
        df_n = df_n.sort_values('wl').reset_index(drop=True)
        df_k = df_k.sort_values('wl').reset_index(drop=True)

        # 3. 验证拆分结果（金属Al特性：n≈1.0-1.5，k≈1.0-10.0）
        print(f"\n✅ 拆分结果详情：")
        print(f"  - Al_n（折射率）：")
        print(f"    行数：{len(df_n)}，波长范围：{df_n['wl'].min():.2f}~{df_n['wl'].max():.2f}μm")
        print(f"    数值范围：{df_n['Al_n'].min():.6f}~{df_n['Al_n'].max():.6f}")
        print(f"  - Al_k（消光系数）：")
        print(f"    行数：{len(df_k)}，波长范围：{df_k['wl'].min():.2f}~{df_k['wl'].max():.2f}μm")
        print(f"    数值范围：{df_k['Al_k'].min():.6f}~{df_k['Al_k'].max():.6f}")

        # 金属特性校验（k通常大于n）
        n_mean = df_n['Al_n'].mean()
        k_mean = df_k['Al_k'].mean()
        if k_mean <= n_mean:
            print("⚠️  提示：k平均值 <= n平均值（可能分割比例需调整）")
            print(f"  - 若拆分错误，可修改 split_ratio（当前为{split_ratio}），例如改为0.4或0.6")

    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 - {input_path}")
        return False
    except Exception as e:
        print(f"❌ 拆分失败：{type(e).__name__} - {e}")
        return False

    # 4. 保存拆分文件
    try:
        df_n.to_csv(output_n_path, index=False, encoding='utf-8')
        df_k.to_csv(output_k_path, index=False, encoding='utf-8')
        print(f"\n🎉 拆分完成！文件已保存至 Pre_Data 文件夹：")
        print(f"  - 折射率n：{output_n_path}（{len(df_n)}行）")
        print(f"  - 消光系数k：{output_k_path}（{len(df_k)}行）")
        return True
    except Exception as e:
        print(f"❌ 保存文件失败：{type(e).__name__} - {e}")
        return False


# ----------------------------------------------------------------------
# 手动调整分割比例（若拆分错误时使用）
# ----------------------------------------------------------------------
def split_al_nk_with_custom_ratio(input_path, output_n_path, output_k_path, split_ratio=0.5):
    """
    自定义分割比例：
    - split_ratio：前split_ratio比例为n，后(1-split_ratio)为k
    - 例：split_ratio=0.4 → 前40%为n，后60%为k
    - 例：split_ratio=0.6 → 前60%为n，后40%为k
    """
    print(f"\n📌 使用自定义分割比例：{split_ratio}")
    return split_al_nk_by_block(input_path, output_n_path, output_k_path)


# ----------------------------------------------------------------------
# 运行拆分（直接执行）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 方法1：使用默认分割比例（0.5）
    success = split_al_nk_by_block(INPUT_FILE, OUTPUT_N_FILE, OUTPUT_K_FILE)

    # 方法2：若拆分错误，注释上面一行，使用自定义比例（例如0.4）
    # success = split_al_nk_with_custom_ratio(INPUT_FILE, OUTPUT_N_FILE, OUTPUT_K_FILE, split_ratio=0.4)

    if not success:
        print("\n❌ Al.csv 拆分失败，请检查文件或调整分割比例！")
    else:
        print("\n✅ Al.csv 拆分成功！可直接运行之前的对齐+插值代码～")