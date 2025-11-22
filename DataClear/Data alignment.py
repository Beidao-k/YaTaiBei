import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 全局配置（英文显示，适配k值图表）
# ----------------------------------------------------------------------
plt.rcParams['axes.unicode_minus'] = False  # Support negative numbers
plt.rcParams['figure.figsize'] = (12, 8)  # Figure size (width × height)
plt.rcParams['grid.alpha'] = 0.3  # Grid transparency

# ----------------------------------------------------------------------
# 文件路径配置（k值专用文件）
# ----------------------------------------------------------------------
# Input file paths (k data files)
file_path_1 = r'D:\Python_Code\YaTaiBei\Pre_Data\Ag_k.csv'
file_path_2 = r'D:\Python_Code\YaTaiBei\Pre_Data\PDMS_k.csv'

# Output directory and file paths (distinguish from n data)
output_dir = r'D:\Python_Code\YaTaiBei\Data'
output_interpolated_path = os.path.join(output_dir, 'Ag_PDMS_k_interpolated_0_01.csv')
output_plot_path = os.path.join(output_dir, 'Ag_PDMS_k_comparison_plot.png')

# ----------------------------------------------------------------------
# Auxiliary function: Read, clean and deduplicate data
# ----------------------------------------------------------------------
def clean_and_group_data(path, k_col_name):
    """
    Read CSV file → Convert to numeric → Deduplicate → Filter 0.40-20.00μm range
    Return DataFrame with wl as index
    """
    try:
        df = pd.read_csv(path, header=None, names=['wl', k_col_name])
    except FileNotFoundError:
        print(f"❌ Error: File not found - {path}")
        return None
    except Exception as e:
        print(f"❌ Error: Failed to read file {path} - {e}")
        return None

    # Force convert to numeric, invalid values become NaN
    df['wl'] = pd.to_numeric(df['wl'], errors='coerce')
    df[k_col_name] = pd.to_numeric(df[k_col_name], errors='coerce')

    # Drop rows with NaN
    df = df.dropna(subset=['wl', k_col_name])

    # Filter: Only keep data within 0.40-20.00μm
    df = df[(df['wl'] >= 0.40) & (df['wl'] <= 20.00)]

    # Deduplicate: Average values for duplicate wl
    df_cleaned = df.groupby('wl', as_index=False)[k_col_name].mean()

    # Set wl as index for interpolation
    return df_cleaned.set_index('wl') if not df_cleaned.empty else None

# ----------------------------------------------------------------------
# Plot function: Raw vs Interpolated (k value, English display)
# ----------------------------------------------------------------------
def plot_comparison(df1_raw, df2_raw, df1_interpolated, df2_interpolated):
    """
    Plot comparison of raw and interpolated data (extinction coefficient k)
    Wavelength range: 0.40-20.00μm
    """
    print("\n📊 Starting to plot raw vs interpolated comparison (Wavelength range: 0.40-20.00μm)...")

    # Create 2 subplots (vertical arrangement, shared x-axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # -------------------------- Plot Ag_k Comparison --------------------------
    # Raw data: Blue scatter
    ax1.scatter(
        df1_raw.index, df1_raw['Ag_k'],
        s=20, c='#1f77b4', alpha=0.6, label='Raw Data', zorder=2
    )
    # Interpolated data: Orange line
    ax1.plot(
        df1_interpolated.index, df1_interpolated['Ag_k'],
        c='#ff7f0e', linewidth=1.2, label='Interpolated Data (0.01μm step)', zorder=3
    )
    # Force x-axis range
    ax1.set_xlim(0.40, 20.00)
    # Plot styling
    ax1.set_title('Ag Extinction Coefficient k: Raw vs Interpolated', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Extinction Coefficient k (Ag)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, linestyle='--')
    # Y-axis adaptation (k values are close to 0)
    y_min = 0  # k values can't be negative, start from 0
    y_max = df1_raw['Ag_k'].max() * 1.1  # Add 10% margin to avoid clipping
    ax1.set_ylim(y_min, y_max)

    # -------------------------- Plot PDMS_k Comparison --------------------------
    # Raw data: Green scatter
    ax2.scatter(
        df2_raw.index, df2_raw['PDMS_k'],
        s=20, c='#2ca02c', alpha=0.6, label='Raw Data', zorder=2
    )
    # Interpolated data: Red line
    ax2.plot(
        df2_interpolated.index, df2_interpolated['PDMS_k'],
        c='#d62728', linewidth=1.2, label='Interpolated Data (0.01μm step)', zorder=3
    )
    # Force x-axis range
    ax2.set_xlim(0.40, 20.00)
    # Plot styling
    ax2.set_title('PDMS Extinction Coefficient k: Raw vs Interpolated', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Wavelength λ (μm)', fontsize=12)
    ax2.set_ylabel('Extinction Coefficient k (PDMS)', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, linestyle='--')
    # Y-axis adaptation (PDMS k is very small)
    y_min = 0
    y_max = df2_raw['PDMS_k'].max() * 1.2  # Add 20% margin for better visibility
    ax2.set_ylim(y_min, y_max)

    # -------------------------- Save Plot --------------------------
    plt.tight_layout()
    plt.savefig(output_plot_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✅ Comparison plot saved to: {output_plot_path}")

# ----------------------------------------------------------------------
# Main function: Data interpolation and alignment (k value dedicated)
# ----------------------------------------------------------------------
def interpolate_and_align_data(path1, path2, output_path):
    print("🚀 Starting data processing, cleaning and interpolation (Extinction Coefficient k data)...")

    # --- 1. Data cleaning and deduplication ---
    df1_cleaned = clean_and_group_data(path1, 'Ag_k')
    df2_cleaned = clean_and_group_data(path2, 'PDMS_k')

    if df1_cleaned is None or df2_cleaned is None:
        print("❌ Processing aborted due to file reading failure.")
        return
    if df1_cleaned.empty or df2_cleaned.empty:
        print("❌ Error: No valid data within 0.40-20.00μm range. Please check raw CSV files.")
        return

    print(f"✅ Data 1 (Ag_k) cleaned. Range: {df1_cleaned.index.min():.2f} - {df1_cleaned.index.max():.2f} μm")
    print(f"✅ Data 2 (PDMS_k) cleaned. Range: {df2_cleaned.index.min():.2f} - {df2_cleaned.index.max():.2f} μm")

    # --- 2. Define target wavelength grid ---
    start_wl = 0.40
    stop_wl = 20.00
    step_wl = 0.01
    new_wl_grid = np.round(np.arange(start_wl, stop_wl + step_wl, step_wl), 2)
    print(f"📐 Defined new wavelength grid: {start_wl} to {stop_wl}, step {step_wl}. Total points: {len(new_wl_grid)}")

    # --- 3. Linear interpolation ---
    def interpolate_data(df_cleaned, new_wl_grid):
        union_index = df_cleaned.index.union(new_wl_grid)
        df_resampled = df_cleaned.reindex(union_index)
        df_interpolated = df_resampled.interpolate(method='linear', limit_direction='both')
        return df_interpolated.loc[new_wl_grid]

    df1_interpolated = interpolate_data(df1_cleaned, new_wl_grid)
    df2_interpolated = interpolate_data(df2_cleaned, new_wl_grid)

    # --- Plot comparison ---
    plot_comparison(df1_cleaned, df2_cleaned, df1_interpolated, df2_interpolated)

    # --- 4. Merge and format output (k value high precision) ---
    final_merged_df = pd.concat([df1_interpolated, df2_interpolated], axis=1).reset_index().rename(
        columns={'index': 'wl'})
    final_merged_df['wl'] = final_merged_df['wl'].apply(lambda x: f"{x:.2f}")
    final_merged_df['Ag_k'] = final_merged_df['Ag_k'].apply(lambda x: f"{x:.6f}")  # k value: 6 decimal places
    final_merged_df['PDMS_k'] = final_merged_df['PDMS_k'].apply(lambda x: f"{x:.8f}")  # PDMS k: higher precision

    print("✅ All columns formatted as strings to avoid scientific notation.")

    # --- 5. Save result ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_merged_df.to_csv(output_path, index=False)
        print(f"\n🎉 Result saved to: {output_path}")
        print("\n--- Preview of interpolated data (first 5 rows) ---")
        print(final_merged_df.head())
    except Exception as e:
        print(f"❌ Error: Failed to save file - {e}")

# ----------------------------------------------------------------------
# Run main function
# ----------------------------------------------------------------------
if __name__ == "__main__":
    interpolate_and_align_data(file_path_1, file_path_2, output_interpolated_path)