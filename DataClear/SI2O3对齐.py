import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 全局配置（英文图表，适配学术场景）
# ----------------------------------------------------------------------
plt.rcParams['axes.unicode_minus'] = False  # 支持负号（实际k值非负，仅为兼容）
plt.rcParams['figure.figsize'] = (12, 8)  # 图表尺寸（宽×高）
plt.rcParams['grid.alpha'] = 0.3  # 网格透明度（不遮挡数据）
plt.rcParams['font.size'] = 10  # 基础字体大小，保证可读性

# ----------------------------------------------------------------------
# 文件路径配置（对接拆分后的n/k文件）
# ----------------------------------------------------------------------
# 输入：拆分后的清洁数据
INPUT_N_FILE = os.path.join("Pre_Data", "Al2O3_n_split.csv")
INPUT_K_FILE = os.path.join("Pre_Data", "Al2O3_k_split.csv")
# 输出：插值结果+对比图
output_dir = "Pre_Data"
output_interpolated_csv = os.path.join(output_dir, "Al2O3_nk_interpolated_0.01μm.csv")
output_plot_png = os.path.join(output_dir, "Al2O3_nk_raw_vs_interpolated.png")

# ----------------------------------------------------------------------
# 步骤1：数据清洗（筛选波长+去重，为对齐插值做准备）
# ----------------------------------------------------------------------
def clean_al2o3_data(file_path, col_name):
    """
    清洗单个n/k数据：
    - 读取文件 → 强制转数值 → 筛选0.40-20.00μm → 去重（重复波长取平均）
    返回：以wl为索引的清洁数据DataFrame
    """
    print(f"\n📋 正在清洗 {col_name} 数据...")
    try:
        # 读取拆分后的文件（已带表头：wl + Al2O3_n/Al2O3_k）
        df = pd.read_csv(file_path)
        print(f"ℹ️  原始数据行数：{len(df)}")
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {file_path}（请先完成拆分步骤）")
        return None
    except Exception as e:
        print(f"❌ 读取文件失败：{type(e).__name__} - {e}")
        return None

    # 强制转换为数值类型（避免字符串干扰）
    df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    # 去除含NaN的行（无效数据）
    df = df.dropna(subset=['wl', col_name])
    print(f"ℹ️  去除无效数据后行数：{len(df)}")

    # 筛选核心波长范围：0.40-20.00μm（与目标对齐范围一致）
    df = df[(df['wl'] >= 0.40) & (df['wl'] <= 20.00)]
    if len(df) == 0:
        print(f"❌ 错误：{col_name} 在 0.40-20.00μm 范围内无有效数据")
        return None
    print(f"ℹ️  筛选0.40-20.00μm后行数：{len(df)}")

    # 去重：重复波长取平均值（避免数据冲突）
    df_clean = df.groupby('wl', as_index=False)[col_name].mean()
    print(f"ℹ️  去重后唯一波长数：{len(df_clean)}")

    # 输出关键信息（验证数据合理性）
    print(f"ℹ️  {col_name} 波长范围：{df_clean['wl'].min():.2f} ~ {df_clean['wl'].max():.2f} μm")
    print(f"ℹ️  {col_name} 数值范围：{df_clean[col_name].min():.6f} ~ {df_clean[col_name].max():.6f}")

    # 设置wl为索引，便于后续插值
    return df_clean.set_index('wl')

# ----------------------------------------------------------------------
# 步骤2：对齐+线性插值（核心功能）
# ----------------------------------------------------------------------
def align_and_interpolate(df_clean, col_name):
    """
    核心功能：
    1. 生成目标波长网格（0.40, 0.41, ..., 20.00μm，步长0.01μm）
    2. 基于清洁数据进行线性插值（支持前后外插，确保网格完整）
    返回：插值后的DataFrame（仅含目标网格数据）
    """
    print(f"\n🎯 正在对 {col_name} 进行对齐和插值...")
    # 定义目标波长网格（精确到2位小数，避免浮点数误差）
    start_wl = 0.40
    end_wl = 20.00
    step_wl = 0.01
    target_wavelengths = np.round(np.arange(start_wl, end_wl + step_wl, step_wl), 2)
    total_points = len(target_wavelengths)
    print(f"ℹ️  目标网格：{start_wl}~{end_wl}μm，步长{step_wl}μm，总点数：{total_points}")

    # 合并原始波长和目标网格 → 插值 → 提取目标网格数据
    # 1. 合并索引（确保目标网格的每个波长都被覆盖）
    merged_index = df_clean.index.union(target_wavelengths)
    df_merged = df_clean.reindex(merged_index)
    # 2. 线性插值（limit_direction='both' 支持首尾外插）
    df_interpolated = df_merged.interpolate(method='linear', limit_direction='both')
    # 3. 只保留目标网格数据（删除原始波长，仅保留0.40-20.00μm步长0.01μm的数据）
    df_final = df_interpolated.loc[target_wavelengths]

    print(f"✅ {col_name} 插值完成，最终数据点数：{len(df_final)}")
    return df_final

# ----------------------------------------------------------------------
# 步骤3：绘制英文对比图（原始数据 vs 插值数据）
# ----------------------------------------------------------------------
def plot_comparison_chart(df_n_raw, df_k_raw, df_n_interp, df_k_interp):
    """
    绘制上下两个子图：
    - 上：Al2O3_n 原始散点 + 插值线
    - 下：Al2O3_k 原始散点 + 插值线
    英文标注，适配学术报告
    """
    print("\n📊 正在生成对比图...")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.15})  # 减小子图间距

    # -------------------------- 绘制 Al2O3_n 对比 --------------------------
    ax1.scatter(
        df_n_raw.index, df_n_raw['Al2O3_n'],
        s=15, c='#1f77b4', alpha=0.6, label='Raw Data', zorder=2
    )
    ax1.plot(
        df_n_interp.index, df_n_interp['Al2O3_n'],
        c='#ff7f0e', linewidth=1.2, label='Interpolated (0.01μm step)', zorder=3
    )
    ax1.set_xlim(0.40, 20.00)
    ax1.set_title('Al2O3 Refractive Index (n): Raw vs Interpolated', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('Refractive Index (n)', fontsize=11)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    # y轴微调（避免数据贴边，提升美观度）
    n_min = df_n_raw['Al2O3_n'].min() * 0.995
    n_max = df_n_raw['Al2O3_n'].max() * 1.005
    ax1.set_ylim(n_min, n_max)

    # -------------------------- 绘制 Al2O3_k 对比 --------------------------
    ax2.scatter(
        df_k_raw.index, df_k_raw['Al2O3_k'],
        s=15, c='#2ca02c', alpha=0.6, label='Raw Data', zorder=2
    )
    ax2.plot(
        df_k_interp.index, df_k_interp['Al2O3_k'],
        c='#d62728', linewidth=1.2, label='Interpolated (0.01μm step)', zorder=3
    )
    ax2.set_xlim(0.40, 20.00)
    ax2.set_title('Al2O3 Extinction Coefficient (k): Raw vs Interpolated', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('Wavelength (μm)', fontsize=11)
    ax2.set_ylabel('Extinction Coefficient (k)', fontsize=11)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.3)
    # k轴从0开始（符合物理意义：消光系数非负）
    k_max = df_k_raw['Al2O3_k'].max() * 1.2
    ax2.set_ylim(bottom=0, top=k_max)

    # 保存高清图表（150 DPI，支持缩放）
    plt.tight_layout()
    plt.savefig(output_plot_png, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ 对比图已保存至：{output_plot_png}")

# ----------------------------------------------------------------------
# 步骤4：合并n/k插值结果，格式化输出（无科学计数法）
# ----------------------------------------------------------------------
def merge_and_save_result(df_n_interp, df_k_interp):
    """
    合并n和k的插值数据，格式化小数位数，保存为CSV文件：
    - wl：2位小数（0.40, 0.41, ...）
    - Al2O3_n：6位小数（适配折射率精度）
    - Al2O3_k：8位小数（适配消光系数极小值）
    """
    print("\n💾 正在合并并保存插值结果...")
    # 合并数据（按索引wl对齐）
    final_df = pd.concat([df_n_interp, df_k_interp], axis=1).reset_index()
    final_df.rename(columns={'index': 'wl'}, inplace=True)

    # 格式化小数位数（避免科学计数法，确保可读性）
    final_df['wl'] = final_df['wl'].apply(lambda x: f"{x:.2f}")
    final_df['Al2O3_n'] = final_df['Al2O3_n'].apply(lambda x: f"{x:.6f}")
    final_df['Al2O3_k'] = final_df['Al2O3_k'].apply(lambda x: f"{x:.8f}")

    # 保存文件（UTF-8编码，兼容Windows/Mac/Linux）
    try:
        final_df.to_csv(output_interpolated_csv, index=False, encoding='utf-8')
        print(f"✅ 插值结果已保存至：{output_interpolated_csv}")
        # 预览前5行数据（验证格式）
        print("\n📄 插值结果预览（前5行）：")
        print(final_df.head())
        print(f"\n🎉 全部流程完成！共生成 {len(final_df)} 行数据（0.40-20.00μm，步长0.01μm）")
    except Exception as e:
        print(f"❌ 保存文件失败：{type(e).__name__} - {e}")

# ----------------------------------------------------------------------
# 主函数：串联所有步骤（清洗→对齐插值→绘图→保存）
# ----------------------------------------------------------------------
def main_al2o3_interpolation():
    print("🚀 开始 Al2O3 数据对齐+插值流程...")
    print("="*50)

    # Step 1: 清洗n和k数据
    df_n_clean = clean_al2o3_data(INPUT_N_FILE, 'Al2O3_n')
    df_k_clean = clean_al2o3_data(INPUT_K_FILE, 'Al2O3_k')
    if df_n_clean is None or df_k_clean is None:
        print("\n❌ 流程中止：数据清洗失败")
        return

    # Step 2: 对齐+插值
    df_n_interp = align_and_interpolate(df_n_clean, 'Al2O3_n')
    df_k_interp = align_and_interpolate(df_k_clean, 'Al2O3_k')

    # Step 3: 绘制对比图
    plot_comparison_chart(df_n_clean, df_k_clean, df_n_interp, df_k_interp)

    # Step 4: 合并并保存结果
    merge_and_save_result(df_n_interp, df_k_interp)

    print("\n" + "="*50)
    print("✅ Al2O3 对齐+插值流程全部完成！")

# ----------------------------------------------------------------------
# 运行主函数（直接执行即可）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main_al2o3_interpolation()