import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 全局配置（移除中文支持，使用默认英文字体）
# ----------------------------------------------------------------------
plt.rcParams['axes.unicode_minus'] = False  # Support negative numbers
plt.rcParams['figure.figsize'] = (12, 8)  # Figure size (width × height)
plt.rcParams['grid.alpha'] = 0.3  # Grid transparency

# ----------------------------------------------------------------------
# 文件路径配置（保持不变）
# ----------------------------------------------------------------------
INPUT_N_FILE = os.path.join("Pre_Data", "SIO2_n.csv")
INPUT_K_FILE = os.path.join("Pre_Data", "SIO2_k.csv")
output_dir = "Pre_Data"
output_interpolated_path = os.path.join(output_dir, "SIO2_nk_interpolated_0_01.csv")
output_plot_path = os.path.join(output_dir, "SIO2_nk_comparison_plot.png")

# ----------------------------------------------------------------------
# 辅助函数：读取、清洗和去重（控制台输出改为英文）
# ----------------------------------------------------------------------
def clean_and_group_data(path, value_col_name):
    try:
        df = pd.read_csv(path, header=None, names=['wl', value_col_name])
    except FileNotFoundError:
        print(f"❌ Error: File not found - {path}")
        return None
    except Exception as e:
        print(f"❌ Error: Failed to read file {path} - {e}")
        return None

    df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
    df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')
    df = df.dropna(subset=['wl', value_col_name])
    df_cleaned = df.groupby('wl', as_index=False)[value_col_name].mean()
    df_cleaned = df_cleaned[(df_cleaned['wl'] >= 0.40) & (df_cleaned['wl'] <= 20.00)]

    return df_cleaned.set_index('wl') if not df_cleaned.empty else None

# ----------------------------------------------------------------------
# 核心函数：绘制处理前后对比图（全英文，限定 0.40-20.00μm）
# ----------------------------------------------------------------------
def plot_comparison(df_n_raw, df_k_raw, df_n_interpolated, df_k_interpolated):
    """
    Plot comparison of raw and interpolated data (SiO2 n/k)
    Wavelength range: 0.40-20.00μm
    """
    print("\n📊 Starting to plot raw vs interpolated comparison (Wavelength range: 0.40-20.00μm)...")

    # Create 2 subplots (vertical arrangement, shared x-axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # -------------------------- Plot SiO2_n Comparison --------------------------
    # Raw data: Blue scatter
    ax1.scatter(
        df_n_raw.index, df_n_raw['SIO2_n'],
        s=20, c='#1f77b4', alpha=0.6, label='Raw Data', zorder=2
    )
    # Interpolated data: Orange line
    ax1.plot(
        df_n_interpolated.index, df_n_interpolated['SIO2_n'],
        c='#ff7f0e', linewidth=1.2, label='Interpolated Data (0.01μm step)', zorder=3
    )
    # Force x-axis range
    ax1.set_xlim(0.40, 20.00)
    # Plot styling (English)
    ax1.set_title('SiO2 Refractive Index n: Raw vs Interpolated', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Refractive Index n', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--')
    # Y-axis adaptation
    y_min = df_n_raw['SIO2_n'].min() * 0.995
    y_max = df_n_raw['SIO2_n'].max() * 1.005
    ax1.set_ylim(y_min, y_max)

    # -------------------------- Plot SiO2_k Comparison --------------------------
    # Raw data: Green scatter
    ax2.scatter(
        df_k_raw.index, df_k_raw['SIO2_k'],
        s=20, c='#2ca02c', alpha=0.6, label='Raw Data', zorder=2
    )
    # Interpolated data: Red line
    ax2.plot(
        df_k_interpolated.index, df_k_interpolated['SIO2_k'],
        c='#d62728', linewidth=1.2, label='Interpolated Data (0.01μm step)', zorder=3
    )
    # Force x-axis range
    ax2.set_xlim(0.40, 20.00)
    # Plot styling (English)
    ax2.set_title('SiO2 Extinction Coefficient k: Raw vs Interpolated', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Wavelength λ (μm)', fontsize=12)
    ax2.set_ylabel('Extinction Coefficient k', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--')
    # Y-axis starts from 0 (physical meaning)
    ax2.set_ylim(bottom=0, top=df_k_raw['SIO2_k'].max() * 1.2)

    # -------------------------- Save Plot --------------------------
    plt.tight_layout()
    plt.savefig(output_plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✅ Comparison plot saved to: {output_plot_path}")

# ----------------------------------------------------------------------
# 主函数：数据插值与对齐（控制台输出改为英文）
# ----------------------------------------------------------------------
def interpolate_and_align_sio2_data(path_n, path_k, output_path):
    print("🚀 Starting data processing, cleaning and interpolation (SiO2 n/k data)...")

    # --- 1. Data cleaning and deduplication ---
    df_n_cleaned = clean_and_group_data(path_n, 'SIO2_n')
    df_k_cleaned = clean_and_group_data(path_k, 'SIO2_k')

    if df_n_cleaned is None or df_k_cleaned is None:
        print("❌ Processing aborted due to file reading failure.")
        return
    if df_n_cleaned.empty or df_k_cleaned.empty:
        print("❌ Error: No valid data within 0.40-20.00μm range. Please check raw CSV files.")
        return

    print(f"✅ SiO2_n data cleaned. Valid wavelength range: {df_n_cleaned.index.min():.2f} - {df_n_cleaned.index.max():.2f} μm")
    print(f"✅ SiO2_k data cleaned. Valid wavelength range: {df_k_cleaned.index.min():.2f} - {df_k_cleaned.index.max():.2f} μm")

    # --- 2. Define target wavelength grid ---
    start_wl = 0.40
    stop_wl = 20.00
    step_wl = 0.01
    new_wl_grid = np.round(np.arange(start_wl, stop_wl + step_wl, step_wl), 2)
    print(f"📐 Defined new wavelength grid: {start_wl} ~ {stop_wl} μm, step {step_wl} μm. Total points: {len(new_wl_grid)}")

    # --- 3. Linear interpolation ---
    def interpolate_data(df_cleaned, new_wl_grid):
        union_index = df_cleaned.index.union(new_wl_grid)
        df_resampled = df_cleaned.reindex(union_index)
        df_interpolated = df_resampled.interpolate(method='linear', limit_direction='both')
        return df_interpolated.loc[new_wl_grid]

    df_n_interpolated = interpolate_data(df_n_cleaned, new_wl_grid)
    df_k_interpolated = interpolate_data(df_k_cleaned, new_wl_grid)

    # --- 4. Plot comparison ---
    plot_comparison(df_n_cleaned, df_k_cleaned, df_n_interpolated, df_k_interpolated)

    # --- 5. Merge and format output ---
    final_merged_df = pd.concat(
        [df_n_interpolated, df_k_interpolated],
        axis=1
    ).reset_index().rename(columns={'index': 'wl'})

    final_merged_df['wl'] = final_merged_df['wl'].apply(lambda x: f"{x:.2f}")
    final_merged_df['SIO2_n'] = final_merged_df['SIO2_n'].apply(lambda x: f"{x:.6f}")
    final_merged_df['SIO2_k'] = final_merged_df['SIO2_k'].apply(lambda x: f"{x:.8f}")

    # --- 6. Save result ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_merged_df.to_csv(output_path, index=False)
        print(f"\n🎉 Processing result saved to: {output_path}")
        print("\n--- Preview of interpolated data (first 5 rows) ---")
        print(final_merged_df.head())
    except Exception as e:
        print(f"❌ Error: Failed to save file - {e}")

# ----------------------------------------------------------------------
# Run main function
# ----------------------------------------------------------------------
if __name__ == "__main__":
    interpolate_and_align_sio2_data(
        path_n=INPUT_N_FILE,
        path_k=INPUT_K_FILE,
        output_path=output_interpolated_path
    )