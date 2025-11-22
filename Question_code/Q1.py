import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# -------------------------- 1. Read and Preprocess Data --------------------------
# --- 请确保这些路径是正确的 ---
n_file_path = r"D:\Python_Code\YaTaiBei\Data\Ag-PDMS_n.csv"
k_file_path = r"D:\Python_Code\YaTaiBei\Data\Ag-PDMS_k.csv"

try:
    df_n = pd.read_csv(n_file_path)
    df_k = pd.read_csv(k_file_path)
except FileNotFoundError:
    print(f"Error: Could not find data files at {n_file_path} or {k_file_path}")
    print("Please update the file paths.")
    # 生成一些虚拟数据以便代码可以运行
    wl = np.linspace(0.4, 25, 500)
    PDMS_n = np.full_like(wl, 1.41)
    PDMS_k = np.zeros_like(wl)
    PDMS_k[(wl >= 0.4) & (wl <= 2.5)] = 0.001
    PDMS_k[(wl >= 8) & (wl <= 13)] = 0.05
    Ag_n = np.full_like(wl, 0.15)
    Ag_k = np.full_like(wl, 4.0)
    print("--- Using dummy data to proceed ---")
else:
    # Unify wavelength baseline
    wl_n = df_n["wl"].values
    wl_k = df_k["wl"].values
    common_wl = np.intersect1d(wl_n, wl_k)

    # Extract and sort data based on common wavelengths
    Ag_n = df_n[df_n["wl"].isin(common_wl)]["Ag_n"].values
    PDMS_n = df_n[df_n["wl"].isin(common_wl)]["PDMS_n"].values
    Ag_k = df_k[df_k["wl"].isin(common_wl)]["Ag_k"].values
    PDMS_k = df_k[df_k["wl"].isin(common_wl)]["PDMS_k"].values

    sorted_idx = np.argsort(common_wl)
    wl = common_wl[sorted_idx]
    Ag_n = Ag_n[sorted_idx]
    Ag_k = Ag_k[sorted_idx]
    PDMS_n = PDMS_n[sorted_idx]
    PDMS_k = PDMS_k[sorted_idx]

print(f"Data loaded. Wavelength range: {wl.min():.1f}μm ~ {wl.max():.1f}μm, {len(wl)} data points")


# -------------------------- 2. Emissivity Calculation Function --------------------------
def calculate_emissivity(wavelengths, pdms_n, pdMS_k, ag_n, ag_k, film_thickness):
    """
    Calculate emissivity (E = 1 - R) using thin-film interference theory.
    """
    emissivity = []
    # Ensure all inputs are NumPy arrays
    wavelengths = np.asarray(wavelengths)
    pdms_n = np.asarray(pdms_n)
    pdms_k = np.asarray(pdMS_k)
    ag_n = np.asarray(ag_n)
    ag_k = np.asarray(ag_k)

    lam = wavelengths * 1e-6  # Wavelength from μm to m
    N1 = pdms_n + 1j * pdms_k  # PDMS complex refractive index
    N2 = ag_n + 1j * ag_k  # Ag (Substrate) complex refractive index
    N0 = 1.0 + 0j  # Air (N0 ≈ 1)

    r10 = (N0 - N1) / (N0 + N1)
    r21 = (N1 - N2) / (N1 + N2)
    delta = (2 * np.pi * N1 * film_thickness) / lam

    # Fresnel reflection coefficient (total)
    total_r = (r10 + r21 * np.exp(1j * 2 * delta)) / (1 + r10 * r21 * np.exp(1j * 2 * delta))

    # Reflectivity R = |r|^2
    total_R = np.abs(total_r) ** 2

    # Emissivity E = 1 - R
    emissivity = 1 - total_R
    return emissivity


# -------------------------- 3. Core Thicknesses (for Plot A) --------------------------
# --- MODIFIED LINES ---
# Thickness list (m) -> 1, 5, 10, 20, 50, 100, 200μm
thickness_list = [1e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6]
thickness_labels = [1, 5, 10, 20, 50, 100, 200]  # for labels (μm)

# Calculate emissivity
eps_dict = {}
for idx, thick in enumerate(thickness_list):
    eps = calculate_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, thick)
    eps_dict[thickness_labels[idx]] = eps
    print(f"✅ Calculation complete for {thickness_labels[idx]}μm PDMS (for Plot A)")

# -------------------------- 4. [Plot A] Spectral Curves for Core Thicknesses (Modified - Styled Lines) --------------------------
plt.figure(figsize=(12, 7))

# --- MODIFIED LINES (Expanded color and style lists for 7 curves) ---
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#888888']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
linewidths = [2.5, 2.5, 2.5, 2.5, 2.0, 2.0, 2.0] # Make thicker lines slightly thinner
marker_style = 'o'
marker_frequency = 100 # Add a marker every 100 points

for idx, (thick, eps) in enumerate(eps_dict.items()):
    plt.plot(wl, eps,
             color=colors[idx],
             linestyle=linestyles[idx],
             linewidth=linewidths[idx],
             marker=marker_style,
             markevery=marker_frequency * (idx + 1), # Stagger markers for less clutter
             markerfacecolor='none',          # Hollow markers
             markeredgecolor=colors[idx],
             markersize=5,
             label=f'PDMS Thickness = {thick}μm')

# Highlight atmospheric window
plt.axvspan(8, 13, alpha=0.1, color='red', label='Atmospheric Window (8-13μm)')

# Chart settings
plt.xlabel(r'Wavelength (μm)', fontsize=12)
plt.ylabel(r'Emissivity $\epsilon(\lambda)$', fontsize=12)
# --- MODIFIED LINE (Updated title) ---
plt.title(r'[Plot A] Emissivity vs. Wavelength of PDMS Thin Film (1-200μm) on Silver Substrate',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(wl.min(), wl.max())
plt.ylim(0, 1.0)

# Save figure (Updated filename for this style)
plt.savefig(r"D:\Python_Code\YaTaiBei\Data\PDMS_Emissivity_Plot_A_Styled_Expanded.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 5. Quantitative Analysis (based on core thicknesses) --------------------------
# 8-13μm Atmospheric Window (Core)
atm_idx = (wl >= 8) & (wl <= 13)
print(f"\n=== [Plot A] Average Emissivity in Atmospheric Window (8-13μm) ===")
for thick, eps in eps_dict.items():
    avg_eps = np.mean(eps[atm_idx])
    print(f"  {thick:<5}μm: {avg_eps:.4f}")

# 0.4-0.7μm Visible Light (starting from 0.4 based on your data)
vis_idx = (wl >= 0.4) & (wl <= 0.7)
if np.any(vis_idx):
    print(f"\n=== [Plot A] Average Emissivity in Visible Light (0.4-0.7μm) ===")
    for thick, eps in eps_dict.items():
        avg_eps_vis = np.mean(eps[vis_idx])
        print(f"  {thick:<5}μm: {avg_eps_vis:.4f}")

# -------------------------- 6. Prepare Continuous Data for [Plot B] and [Plot C] --------------------------
print(f"\n--- Now scanning thicknesses for Plot B and Plot C... ---")
# --- MODIFIED LINES (Expanded scan range and points) ---
# Define continuous thickness scan range (e.g., 0.1 to 210 μm)
thickness_scan = np.linspace(0.1e-6, 210e-6, 150)  # 150 thickness points
thickness_scan_labels = thickness_scan * 1e6  # convert to μm
n_thickness = len(thickness_scan)
n_wavelength = len(wl)

# Prepare arrays to store results
eps_heatmap = np.zeros((n_thickness, n_wavelength))
avg_eps_atm_list = []
avg_eps_solar_list = []

# Define indices for key bands
# Solar spectrum (0.4 ~ 2.5 μm) - Note: your data starts at 0.4μm
solar_idx = (wl >= 0.4) & (wl <= 2.5)
# Atmospheric window (8 ~ 13 μm)
atm_idx = (wl >= 8) & (wl <= 13)

start_time = time.time()
# Loop and calculate
for i, thick in enumerate(thickness_scan):
    eps_spectrum = calculate_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, thick)

    # Store full spectrum for Heatmap (Plot B)
    eps_heatmap[i, :] = eps_spectrum

    # Calculate and store average values for Plot C
    avg_eps_atm_list.append(np.mean(eps_spectrum[atm_idx]))
    if np.any(solar_idx):
        avg_eps_solar_list.append(np.mean(eps_spectrum[solar_idx]))
    else:
        avg_eps_solar_list.append(np.nan)  # Mark as NaN if no data

    if (i + 1) % 25 == 0: # Adjusted print frequency
        print(f"  ...completed {i + 1}/{n_thickness} thickness points")

end_time = time.time()
print(f"✅ Continuous scan complete. Time taken: {end_time - start_time:.2f} seconds")

# -------------------------- 7. [Plot B] 2D Emissivity Heatmap --------------------------
plt.figure(figsize=(12, 7))

# Use pcolormesh
# X: Wavelength, Y: Thickness, Color: Emissivity
plt.pcolormesh(wl, thickness_scan_labels, eps_heatmap,
               shading='gouraud', cmap='inferno', vmin=0, vmax=1)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label(r'Emissivity $\epsilon(\lambda, d)$', fontsize=12)

# Highlight atmospheric window
plt.axvspan(8, 13, alpha=0.15, facecolor='white', linestyle='--',
            edgecolor='white', label=r'Atmospheric Window (8-13μm)')

# Chart settings
plt.xlabel(r'Wavelength (μm)', fontsize=12)
plt.ylabel(r'PDMS Thickness (μm)', fontsize=12)
plt.title(r'[Plot B] Emissivity vs. Wavelength and Thickness (Heatmap)',
          fontsize=14, fontweight='bold')
plt.xlim(wl.min(), wl.max())
plt.ylim(thickness_scan_labels.min(), thickness_scan_labels.max()) # Y-limit now auto-adjusts to 210μm
plt.legend(loc='lower right')
plt.grid(True, alpha=0.2, linestyle=':')

# Save figure
plt.savefig(r"D:\Python_Code\YaTaiBei\Data\PDMS_Emissivity_Plot_B_Heatmap_Expanded.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 8. [Plot C] Average Emissivity/Absorptivity vs. Thickness --------------------------
# This is the key plot linking Problem 1 and Problem 2
fig, ax1 = plt.subplots(figsize=(12, 7))

# Curve 1: Atmospheric Window Average Emissivity (Left Y-axis)
color1 = '#d62728'  # Red
ax1.plot(thickness_scan_labels, avg_eps_atm_list,
         color=color1, linewidth=2.5, label=r'$\epsilon_{atm}$ (8-13μm)')
ax1.set_xlabel(r'PDMS Thickness (μm)', fontsize=12)
ax1.set_ylabel(r'Atmospheric Window Emissivity', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1.0)
ax1.grid(True, alpha=0.3, linestyle='--')

# Curve 2: Solar Spectrum Average Absorptivity (Right Y-axis)
ax2 = ax1.twinx()
color2 = '#1f77b4'  # Blue
# 使用实线
ax2.plot(thickness_scan_labels, avg_eps_solar_list,
         color=color2, linewidth=2.5, linestyle='-', label=r'$\alpha_{solar}$ (0.4-2.5μm)')
ax2.set_ylabel(r'Solar Absorptivity', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
# --- MODIFIED LINE (Expanded Y-limit for solar) ---
# Solar absorptivity should be low, set a slightly larger Y-limit to see changes
ax2.set_ylim(0, 0.2)

# Chart settings
plt.title(r'[Plot C] Key Performance Metrics vs. PDMS Thickness',
          fontsize=14, fontweight='bold')
plt.xlim(thickness_scan_labels.min(), thickness_scan_labels.max()) # X-limit now auto-adjusts to 210μm

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
           bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=11)

# Adjust layout to prevent legend overlap
fig.tight_layout()
# Adjust again to make space for the bottom legend
plt.subplots_adjust(bottom=0.2)

# Save figure
plt.savefig(r"D:\Python_Code\YaTaiBei\Data\PDMS_Emissivity_Plot_C_Average_Expanded.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n=== All analysis and plotting complete ===")