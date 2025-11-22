import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 全局配置（英文图表）
# ----------------------------------------------------------------------
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 10

# ----------------------------------------------------------------------
# 文件路径配置
# ----------------------------------------------------------------------
INPUT_N_FILE = os.path.join("Pre_Data", "Al_n_split.csv")
INPUT_K_FILE = os.path.join("Pre_Data", "Al_k_split.csv")
output_dir = "Pre_Data"
output_interpolated_csv = os.path.join(output_dir, "Al_nk_interpolated_0.01μm.csv")
output_plot_png = os.path.join(output_dir, "Al_nk_raw_vs_interpolated.png")

# ----------------------------------------------------------------------
# 步骤1：数据清洗
# ----------------------------------------------------------------------
def clean_al_data(file_path, col_name):
    print(f"\n📋 正在清洗 {col_name} 数据...")
    try:
        df = pd.read_csv(file_path)
        print(f"ℹ️  原始数据行数：{len(df)}")
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {file_path}")
        return None
    except Exception as e:
        print(f"❌ 读取文件失败：{type(e).__name__} - {e}")
        return None

    df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df = df.dropna(subset=['wl', col_name])
    df = df[(df['wl'] >= 0.40) & (df['wl'] <= 20.00)]
    if len(df) == 0:
        print(f"❌ 错误：{col_name} 在 0.40-20.00μm 范围内无有效数据")
        return None
    df_clean = df.groupby('wl', as_index=False)[col_name].mean()
    print(f"ℹ️  {col_name} 波长范围：{df_clean['wl'].min():.2f} ~ {df_clean['wl'].max():.2f} μm")
    print(f"ℹ️  {col_name} 数值范围：{df_clean[col_name].min():.6f} ~ {df_clean[col_name].max():.6f}")
    return df_clean.set_index('wl')

# ----------------------------------------------------------------------
# 步骤2：对齐+插值
# ----------------------------------------------------------------------
def align_and_interpolate(df_clean, col_name):
    print(f"\n🎯 正在对 {col_name} 进行对齐和插值...")
    start_wl = 0.40
    end_wl = 20.00
    step_wl = 0.01
    target_wavelengths = np.round(np.arange(start_wl, end_wl + step_wl, step_wl), 2)
    merged_index = df_clean.index.union(target_wavelengths)
    df_merged = df_clean.reindex(merged_index)
    df_interpolated = df_merged.interpolate(method='linear', limit_direction='both')
    df_final = df_interpolated.loc[target_wavelengths]
    print(f"✅ {col_name} 插值完成，最终数据点数：{len(df_final)}")
    return df_final

# ----------------------------------------------------------------------
# 步骤3：绘制对比图（修复n的y轴+子图布局）
# ----------------------------------------------------------------------
def plot_comparison_chart(df_n_raw, df_k_raw, df_n_interp, df_k_interp):
    print("\n📊 正在生成对比图...")
    # 不使用 sharex，避免布局异常；手动同步x轴范围
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.3})  # 增大子图间距

    # -------------------------- 绘制 Al_n 对比（修复y轴范围） --------------------------
    ax1.scatter(
        df_n_raw.index, df_n_raw['Al_n'],
        s=15, c='#1f77b4', alpha=0.6, label='Raw Data', zorder=2
    )
    ax1.plot(
        df_n_interp.index, df_n_interp['Al_n'],
        c='#ff7f0e', linewidth=1.2, label='Interpolated (0.01μm step)', zorder=3
    )
    ax1.set_xlim(0.40, 20.00)
    ax1.set_title('Aluminum Refractive Index (n): Raw vs Interpolated', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylabel('Refractive Index (n)', fontsize=11)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    # 适配实际n值范围（取数据的min*0.98 ~ max*1.02）
    n_min = df_n_raw['Al_n'].min() * 0.98
    n_max = df_n_raw['Al_n'].max() * 1.02
    ax1.set_ylim(n_min, n_max)  # 不再固定0.9，而是根据数据动态调整

    # -------------------------- 绘制 Al_k 对比 --------------------------
    ax2.scatter(
        df_k_raw.index, df_k_raw['Al_k'],
        s=15, c='#2ca02c', alpha=0.6, label='Raw Data', zorder=2
    )
    ax2.plot(
        df_k_interp.index, df_k_interp['Al_k'],
        c='#d62728', linewidth=1.2, label='Interpolated (0.01μm step)', zorder=3
    )
    ax2.set_xlim(0.40, 20.00)  # 手动同步x轴范围
    ax2.set_title('Aluminum Extinction Coefficient (k): Raw vs Interpolated', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('Wavelength (μm)', fontsize=11)
    ax2.set_ylabel('Extinction Coefficient (k)', fontsize=11)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.3)
    k_max = df_k_raw['Al_k'].max() * 1.2
    ax2.set_ylim(bottom=0, top=k_max)

    # 保存图表
    plt.tight_layout()
    plt.savefig(output_plot_png, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ 对比图已保存至：{output_plot_png}")

# ----------------------------------------------------------------------
# 步骤4：合并并保存结果
# ----------------------------------------------------------------------
def merge_and_save_result(df_n_interp, df_k_interp):
    print("\n💾 正在合并并保存插值结果...")
    final_df = pd.concat([df_n_interp, df_k_interp], axis=1).reset_index()
    final_df.rename(columns={'index': 'wl'}, inplace=True)
    final_df['wl'] = final_df['wl'].apply(lambda x: f"{x:.2f}")
    final_df['Al_n'] = final_df['Al_n'].apply(lambda x: f"{x:.6f}")
    final_df['Al_k'] = final_df['Al_k'].apply(lambda x: f"{x:.6f}")
    try:
        final_df.to_csv(output_interpolated_csv, index=False, encoding='utf-8')
        print(f"✅ 插值结果已保存至：{output_interpolated_csv}")
        print("\n📄 插值结果预览（前5行）：")
        print(final_df.head())
        print(f"\n🎉 全部流程完成！共生成 {len(final_df)} 行数据")
    except Exception as e:
        print(f"❌ 保存文件失败：{type(e).__name__} - {e}")

# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main_al_interpolation():
    print("🚀 开始 Al（铝）数据对齐+插值流程...")
    print("="*50)

    df_n_clean = clean_al_data(INPUT_N_FILE, 'Al_n')
    df_k_clean = clean_al_data(INPUT_K_FILE, 'Al_k')
    if df_n_clean is None or df_k_clean is None:
        print("\n❌ 流程中止：数据清洗失败")
        return

    df_n_interp = align_and_interpolate(df_n_clean, 'Al_n')
    df_k_interp = align_and_interpolate(df_k_clean, 'Al_k')

    plot_comparison_chart(df_n_clean, df_k_clean, df_n_interp, df_k_interp)
    merge_and_save_result(df_n_interp, df_k_interp)

    print("\n" + "="*50)
    print("✅ Al（铝）对齐+插值流程全部完成！")

if __name__ == "__main__":
    main_al_interpolation()