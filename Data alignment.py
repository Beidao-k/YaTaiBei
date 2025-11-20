import pandas as pd
import numpy as np
import os  # 导入 os 模块来检查和创建目录

# ----------------------------------------------------------------------
# 文件路径配置
# ----------------------------------------------------------------------
# 您的输入文件路径
file_path_1 = r'D:\Python_Code\YaTaiBei\Pre_Data\Ag_k.csv'
file_path_2 = r'D:\Python_Code\YaTaiBei\Pre_Data\PDMS_k.csv'

# 您的输出目录
output_dir = r'D:\Python_Code\YaTaiBei\Data'
# 您的输出文件路径
output_interpolated_path = os.path.join(output_dir, 'interpolated_data_0_01.csv')


# ----------------------------------------------------------------------
# 辅助函数：读取、清洗和去重
# ----------------------------------------------------------------------
def clean_and_group_data(path, n_col_name):
    """
    读取 CSV 文件，强制将 'wl' 和 'n' 值列转换为数字，
    并对重复的 wl 值取平均。
    返回一个以 wl 为索引 (index) 的 DataFrame。
    """
    try:
        df = pd.read_csv(path, header=None, names=['wl', n_col_name])
    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 - {path}")
        return None
    except Exception as e:
        print(f"❌ 错误: 读取文件 {path} 时出错 - {e}")
        return None

    # --- 关键修正 ---
    # 1. 强制将 'wl' 列转换为数字，无效值变为 NaN
    df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
    # 2. 强制将 'n' 值列转换为数字，无效值变为 NaN
    df[n_col_name] = pd.to_numeric(df[n_col_name], errors='coerce')

    # 3. 在分组前丢弃任何 'wl' 或 'n' 值为 NaN 的行
    df = df.dropna(subset=['wl', n_col_name])
    # --- 修正结束 ---

    # 对重复的 'wl' 值，取 'n' 的平均值
    df_cleaned = df.groupby('wl', as_index=False)[n_col_name].mean()

    # 将 wl 设置为索引，为插值做准备
    return df_cleaned.set_index('wl')


# ----------------------------------------------------------------------
# 主函数：数据插值与对齐
# ----------------------------------------------------------------------
def interpolate_and_align_data(path1, path2, output_path):
    print("🚀 开始数据处理、清洗和插值...")

    # --- 1. 数据清洗和去重 (使用您要求的列名) ---
    df1_cleaned = clean_and_group_data(path1, 'Ag_k')
    df2_cleaned = clean_and_group_data(path2, 'PDMS_k')

    if df1_cleaned is None or df2_cleaned is None:
        print("❌ 因文件读取失败，处理中止。")
        return

    # 检查清洗后是否有数据
    if df1_cleaned.empty or df2_cleaned.empty:
        print("❌ 错误：清洗后数据为空，请检查原始 CSV 文件内容是否有效。")
        return

    # 这里的 print 语句现在是安全的，因为索引是数字
    print(f"✅ 数据1 (Ag_k) 清洗完成。范围: {df1_cleaned.index.min():.2f} - {df1_cleaned.index.max():.2f}")
    print(f"✅ 数据2 (PDMS_k) 清洗完成。范围: {df2_cleaned.index.min():.2f} - {df2_cleaned.index.max():.2f}")

    # --- 2. 定义新的均匀波长网格 ---
    start_wl = 0.40
    stop_wl = 20.00
    step_wl = 0.01

    # 使用 np.round 确保浮点数精度
    new_wl_grid = np.round(np.arange(start_wl, stop_wl + step_wl, step_wl), 2)

    print(f"📐 定义新的波长网格: {start_wl} 到 {stop_wl}，间隔 {step_wl}。总点数: {len(new_wl_grid)}")

    # --- 3. 对两组数据分别进行插值 ---

    # 步骤 3.1: 将原始数据点和新网格点合并
    df1_union_index = df1_cleaned.index.union(new_wl_grid)
    df1_resampled = df1_cleaned.reindex(df1_union_index)

    df2_union_index = df2_cleaned.index.union(new_wl_grid)
    df2_resampled = df2_cleaned.reindex(df2_union_index)

    # 步骤 3.2: 使用线性插值（'linear'）填充 NaN 值
    # limit_direction='both' 确保插值可以向前和向后填充（外插）
    df1_interpolated = df1_resampled.interpolate(method='linear', limit_direction='both')
    df2_interpolated = df2_resampled.interpolate(method='linear', limit_direction='both')

    # 步骤 3.3: 仅选择我们关心的新网格点
    df1_final = df1_interpolated.loc[new_wl_grid]
    df2_final = df2_interpolated.loc[new_wl_grid]

    # --- 4. 最终合并 ---
    # 使用 pd.concat 按列合并（axis=1）两个插值后的 DataFrame
    final_merged_df = pd.concat([df1_final, df2_final], axis=1)

    # 重置索引，使 'wl' 成为一列
    final_merged_df = final_merged_df.reset_index().rename(columns={'index': 'wl'})

    print(f"⭐ 数据插值和对齐完成。总行数: {len(final_merged_df)}")
    print("   - 使用线性插值 (linear interpolation) 完成。")

    print("\n--- 预览插值后的前5行数据 ---")
    print(final_merged_df.head())
    print("----------------------------------\n")

    # --- 5. 格式化和保存结果 ---

    # === 关键修改：强制转换为字符串以避免科学计数法 ===
    # 我们使用 .apply() 和 f-string 来精确控制输出格式

    # 格式化 'wl' 为 2 位小数
    final_merged_df['wl'] = final_merged_df['wl'].apply(lambda x: f"{x:.2f}")

    # 格式化 'Ag_k' 为 6 位小数
    final_merged_df['Ag_k'] = final_merged_df['Ag_k'].apply(lambda x: f"{x:.6f}")

    # 格式化 'PDMS_k' 为 8 位小数 (您要求的)
    final_merged_df['PDMS_k'] = final_merged_df['PDMS_k'].apply(lambda x: f"{x:.8f}")

    print("✅ 已将所有列格式化为字符串，以强制避免科学计数法。")
    # === 修改结束 ===

    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 保存为 CSV 文件，不包含 DataFrame 索引
        # 因为所有内容都已是字符串，所以不需要 'float_format'
        final_merged_df.to_csv(output_path, index=False)
        print(f"🎉 结果已保存至: {output_path}")
    except Exception as e:
        print(f"❌ 错误: 保存文件失败 - {e}")


# ----------------------------------------------------------------------
# 运行主函数
# ----------------------------------------------------------------------
if __name__ == "__main__":
    interpolate_and_align_data(file_path_1, file_path_2, output_interpolated_path)