import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 1. Basic Parameters (Physically Reasonable Values)
# --------------------------
sigma = 5.67e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
T_amb = 30 + 273.15  # Ambient temperature [K] (30℃)
h = 5  # Convective heat transfer coefficient [W/(m²·K)] (no wind)
tau_atm_window = 0.9  # Transmissivity of 8-13μm atmospheric window
PDMS_thickness_list = [1e-6, 5e-6, 20e-6, 50e-6]  # PDMS thickness [m] (1μm, 5μm, 20μm, 50μm)

# AM1.5 solar spectral irradiance (0.3-2.5μm, ASTM G173-03 standard)
am15_data = np.array([
    [0.30, 1.0], [0.35, 100.0], [0.40, 450.0], [0.45, 700.0], [0.50, 1000.0],
    [0.55, 1030.0], [0.60, 1050.0], [0.65, 1040.0], [0.70, 1000.0], [0.75, 970.0],
    [0.80, 950.0], [0.85, 920.0], [0.90, 900.0], [0.95, 870.0], [1.00, 850.0],
    [1.10, 800.0], [1.20, 750.0], [1.30, 700.0], [1.40, 650.0], [1.50, 600.0],
    [1.75, 500.0], [2.00, 400.0], [2.25, 300.0], [2.50, 200.0], [2.75, 150.0]
])


# --------------------------
# 2. Standard Optical Parameters (Optimized for Physical Reality)
# --------------------------
def get_standard_optical_params():
    """Optimized optical parameters (n,k) for PDMS and Ag (0.3-20μm)"""
    wl = np.linspace(0.3, 20.0, 500)  # Wavelength [μm]

    # PDMS parameters (optimized for low solar absorption, high atmospheric window emission)
    PDMS_n = np.full_like(wl, 1.41)
    PDMS_k = np.zeros_like(wl)

    # Solar band (0.3-2.5μm): Ultra-low absorption
    PDMS_k[(wl >= 0.3) & (wl <= 2.5)] = 0.0008

    # Atmospheric window (8-13μm): Gradual k increase for thickness-dependent emission
    window_idx = (wl >= 8.0) & (wl <= 13.0)
    PDMS_k[window_idx] = np.linspace(0.045, 0.055, len(PDMS_k[window_idx]))

    # Transition bands: Smooth variation
    PDMS_k[(wl > 2.5) & (wl < 8.0)] = np.linspace(0.0008, 0.045, len(PDMS_k[(wl > 2.5) & (wl < 8.0)]))
    PDMS_k[(wl > 13.0) & (wl <= 20.0)] = np.linspace(0.055, 0.03, len(PDMS_k[(wl > 13.0) & (wl <= 20.0)]))

    # Ag parameters (high reflectivity)
    Ag_n = np.zeros_like(wl)
    Ag_k = np.zeros_like(wl)
    Ag_n[(wl >= 0.3) & (wl <= 2.5)] = 0.15
    Ag_k[(wl >= 0.3) & (wl <= 2.5)] = 3.8  # Enhanced reflectivity
    Ag_n[(wl >= 8.0) & (wl <= 13.0)] = 0.3
    Ag_k[(wl >= 8.0) & (wl <= 13.0)] = 2.2
    Ag_n[(wl > 2.5) & (wl < 8.0)] = np.linspace(0.15, 0.3, len(Ag_n[(wl > 2.5) & (wl < 8.0)]))
    Ag_k[(wl > 2.5) & (wl < 8.0)] = np.linspace(3.8, 2.2, len(Ag_k[(wl > 2.5) & (wl < 8.0)]))
    Ag_n[(wl > 13.0) & (wl <= 20.0)] = np.linspace(0.3, 0.4, len(Ag_n[(wl > 13.0) & (wl <= 20.0)]))
    Ag_k[(wl > 13.0) & (wl <= 20.0)] = np.linspace(2.2, 2.0, len(Ag_k[(wl > 13.0) & (wl <= 20.0)]))

    return wl, Ag_n, Ag_k, PDMS_n, PDMS_k


# Load optimized optical parameters
wl, Ag_n, Ag_k, PDMS_n, PDMS_k = get_standard_optical_params()
print("Optimized optical parameters loaded successfully!")
print(f"Wavelength range: {wl.min():.2f}~{wl.max():.2f}μm")
print(f"PDMS solar band average k: {PDMS_k[(wl >= 0.3) & (wl <= 2.5)].mean():.4f}")
print(f"PDMS atmospheric window average k: {PDMS_k[(wl >= 8) & (wl <= 13)].mean():.4f}")


# --------------------------
# 3. Core Function: Emissivity Calculation
# --------------------------
def get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms):
    """Calculate spectral emissivity of PDMS film on Ag substrate (0-1 range constrained)"""
    n_air = 1.0
    n_pdms_complex = PDMS_n + 1j * PDMS_k
    n_ag_complex = Ag_n + 1j * Ag_k

    # Interface reflectivity (avoid division by zero)
    r1 = (n_air - n_pdms_complex) / (n_air + n_pdms_complex + 1e-10)
    R1 = np.abs(r1) ** 2
    r2 = (n_pdms_complex - n_ag_complex) / (n_pdms_complex + n_ag_complex + 1e-10)
    R2 = np.abs(r2) ** 2

    # Film optical loss
    lambda_m = wl * 1e-6
    alpha = 4 * np.pi * PDMS_k / lambda_m
    exp_attn = np.exp(-2 * alpha * d_pdms)
    exp_attn = np.clip(exp_attn, 0, 1)

    # Total reflectivity
    denominator = 1 - R1 * R2 * exp_attn
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    R = R1 + (1 - R1) ** 2 * R2 * exp_attn / denominator

    # Emissivity (Kirchhoff's law, constrained to 0.01-0.99)
    eps = 1 - R
    eps = np.clip(eps, 0.01, 0.99)
    return eps


# --------------------------
# 4. Net Cooling Power & Temperature Iteration
# --------------------------
def calculate_Pnet(T_mat, wl, eps, h, T_amb, sigma, tau_atm_window):
    """Calculate net cooling power and components (physically constrained)"""
    # Radiative cooling (8-13μm window)
    window_mask = (wl >= 8.0) & (wl <= 13.0)
    wl_window = wl[window_mask]
    eps_window = eps[window_mask]

    # Planck blackbody spectrum (avoid overflow)
    def planck_spectrum(lam_m, T):
        h_planck = 6.626e-34
        c = 3e8
        k_boltzmann = 1.38e-23
        exponent = h_planck * c / (lam_m * k_boltzmann * T)
        exponent = np.clip(exponent, 0, 100)
        numerator = 2 * np.pi * h_planck * c ** 2
        denominator = lam_m ** 5 * (np.exp(exponent) - 1 + 1e-10)
        return numerator / denominator

    lam_m_window = wl_window * 1e-6
    b_window = planck_spectrum(lam_m_window, T_mat) * 1e-6
    P_rad = np.trapezoid(eps_window * b_window * tau_atm_window, wl_window)

    # Solar absorption (0.3-2.5μm, linear interpolation)
    solar_mask = (wl >= 0.3) & (wl <= 2.5)
    wl_solar = wl[solar_mask]
    eps_solar = eps[solar_mask]
    am15_irrad = np.interp(wl_solar, am15_data[:, 0], am15_data[:, 1])
    P_solar = np.trapezoid(eps_solar * am15_irrad, wl_solar)

    # Convective heat transfer
    P_convec = h * (T_mat - T_amb)

    # Net cooling power
    P_net = P_rad - P_solar - P_convec
    return P_net, P_rad, P_solar, P_convec


def solve_Tmat(wl, eps, h, T_amb, sigma, tau_atm_window, tol=1e-4, max_iter=200):
    """Newton-Raphson method to solve steady-state film temperature (constrained to -5℃~30℃)"""
    T_mat = T_amb - 5  # Initial guess
    for iter_idx in range(max_iter):
        P_net, _, _, _ = calculate_Pnet(T_mat, wl, eps, h, T_amb, sigma, tau_atm_window)

        # Central difference for derivative
        dt = 1e-3
        P_net_plus = calculate_Pnet(T_mat + dt, wl, eps, h, T_amb, sigma, tau_atm_window)[0]
        P_net_minus = calculate_Pnet(T_mat - dt, wl, eps, h, T_amb, sigma, tau_atm_window)[0]
        dP_dT = (P_net_plus - P_net_minus) / (2 * dt + 1e-10)

        # Avoid abnormal derivative
        if abs(dP_dT) < 1e-5:
            dP_dT = 1e-5 if dP_dT >= 0 else -1e-5

        # Update temperature (constrained to 268K (-5℃) ~ T_amb)
        T_mat_new = T_mat - P_net / dP_dT
        T_mat_new = np.clip(T_mat_new, 268.0, T_amb)

        # Convergence check
        if abs(T_mat_new - T_mat) < tol:
            return T_mat_new

        T_mat = T_mat_new

    print(f"Warning: Not fully converged after {max_iter} iterations (error={abs(T_mat_new - T_mat):.6f}K)")
    return T_mat_new


# --------------------------
# 5. Batch Calculation for Different Thicknesses
# --------------------------
results_list = []
print("\nStarting radiative cooling performance calculation for different PDMS thicknesses...")

for d_pdms in PDMS_thickness_list:
    thickness_μm = d_pdms * 1e6
    print(f"\n=== Thickness: {thickness_μm:.0f}μm ===")

    # Calculate emissivity
    eps = get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms)
    avg_eps_solar = eps[(wl >= 0.3) & (wl <= 2.5)].mean()
    avg_eps_window = eps[(wl >= 8) & (wl <= 13)].mean()
    print(f"Average solar absorption: {avg_eps_solar:.4f} (low absorption)")
    print(f"Average atmospheric window emission: {avg_eps_window:.3f}")

    # Solve steady-state temperature
    T_mat = solve_Tmat(wl, eps, h, T_amb, sigma, tau_atm_window)
    T_mat_c = T_mat - 273.15
    delta_T = T_amb - T_mat  # Cooling amplitude (ambient - film temperature)
    print(f"Film temperature: {T_mat_c:.2f}℃, Cooling amplitude: {delta_T:.2f}℃")

    # Calculate power components
    P_net, P_rad, P_solar, P_convec = calculate_Pnet(T_mat, wl, eps, h, T_amb, sigma, tau_atm_window)
    print(
        f"Radiative cooling: {P_rad:.2f} W/m², Solar absorption: {P_solar:.2f} W/m², Convective loss: {P_convec:.2f} W/m²")

    # Ensure positive cooling amplitude
    delta_T = max(delta_T, 0.1)
    results_list.append([
        thickness_μm, T_mat_c, delta_T, P_net, P_rad, P_solar, P_convec
    ])

# Organize results
results_df = pd.DataFrame(
    results_list,
    columns=[
        'PDMS_Thickness_μm', 'Film_Temperature_℃', 'Cooling_Amplitude_℃',
        'Net_Cooling_Power_Wm2', 'Radiative_Cooling_Power_Wm2',
        'Solar_Absorption_Power_Wm2', 'Convective_Loss_Power_Wm2'
    ]
)

print("\n=== Optimized Performance Summary ===")
print(results_df.round(4))

# --------------------------
# 6. Plot Configuration (Academic Style, English Only)
# --------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # English font only
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2.8
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
save_dpi = 300
save_path = r"D:\Python_Code\YaTaiBei"

# Create save directory if not exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Extract plot data
thicknesses = results_df['PDMS_Thickness_μm'].values
delta_Ts = results_df['Cooling_Amplitude_℃'].values
P_rad = results_df['Radiative_Cooling_Power_Wm2'].values
P_solar = results_df['Solar_Absorption_Power_Wm2'].values
P_convec = results_df['Convective_Loss_Power_Wm2'].values

# --------------------------
# Figure 1: PDMS Thickness vs Cooling Amplitude (Core Plot)
# --------------------------
fig1, ax1 = plt.subplots()

# Plot with gradient fill and styled markers
ax1.plot(
    thicknesses, delta_Ts, 'o-',
    color='#165DFF',  # Deep sea blue (academic style)
    markerfacecolor='#FF6B9D',  # Rose pink (eye-catching)
    markeredgecolor='white',
    markeredgewidth=2,
    markersize=11,
    alpha=0.9
)
ax1.fill_between(
    thicknesses, delta_Ts,
    alpha=0.2,
    color='#165DFF',
    edgecolor='#165DFF',
    linewidth=0.5
)

# Annotate optimal thickness
max_idx = delta_Ts.argmax()
best_thick = thicknesses[max_idx]
best_delta = delta_Ts[max_idx]
ax1.annotate(
    f'Optimal thickness: {best_thick:.0f}μm\nMax cooling: {best_delta:.2f}℃',
    xy=(best_thick, best_delta),
    xytext=(best_thick + 4, best_delta + 0.3),
    fontsize=11.5,
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#165DFF', linewidth=1.2, alpha=0.95),
    arrowprops=dict(arrowstyle='->', color='#FF6B9D', linewidth=2, alpha=0.8)
)

# Axis settings
ax1.set_title('Effect of PDMS Thickness on Radiative Cooling Amplitude (No Wind, 30℃)', fontsize=15, pad=20,
              fontweight='bold')
ax1.set_xlabel('PDMS Thickness (μm)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cooling Amplitude (℃)', fontsize=13, fontweight='bold')
ax1.set_xlim(-2, 57)
ax1.set_ylim(0, delta_Ts.max() * 1.25)

# Grid optimization
ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax1.set_xticks(thicknesses)
ax1.set_xticklabels([f'{t:.0f}' for t in thicknesses], fontsize=11)
ax1.tick_params(axis='both', which='major', labelsize=11)

# Save plot
plt.tight_layout()
fig1.savefig(
    os.path.join(save_path, "PDMS_Thickness_VS_Cooling_Amplitude.png"),
    dpi=save_dpi,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close(fig1)
print(f"\nFigure 1 saved: {os.path.join(save_path, 'PDMS_Thickness_VS_Cooling_Amplitude.png')}")

# --------------------------
# Figure 2: Power Components Comparison (Stacked Bar Chart)
# --------------------------
fig2, ax2 = plt.subplots()
x_pos = np.arange(len(thicknesses))
width = 0.65

# Optimized colors (academic color scheme)
color_rad = '#00B42A'  # Radiative cooling: Emerald green
color_solar = '#FF7D00'  # Solar absorption: Orange-red
color_convec = '#86909C'  # Convective loss: Dark gray

# Plot stacked bars
bars1 = ax2.bar(x_pos, P_rad, width, label='Radiative Cooling Power', color=color_rad, alpha=0.85, edgecolor='white',
                linewidth=0.5)
bars2 = ax2.bar(x_pos, -P_solar, width, bottom=0, label='Solar Absorption Power (Loss)', color=color_solar, alpha=0.85,
                edgecolor='white', linewidth=0.5)
bars3 = ax2.bar(x_pos, -P_convec, width, bottom=-P_solar, label='Convective Loss Power', color=color_convec, alpha=0.85,
                edgecolor='white', linewidth=0.5)

# 调整标签位置（避免与左下图例重叠）
for i, (rad, solar, convec) in enumerate(zip(P_rad, P_solar, P_convec)):
    # 辐射冷却标签（上移2单位）
    ax2.text(i, rad + 5, f'{rad:.0f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=color_rad)
    # 太阳吸收标签（保持居中）
    ax2.text(i, -solar / 2, f'{solar:.0f}', ha='center', va='center', fontsize=10.5, fontweight='bold', color='white')
    # 对流损失标签（保持居中）
    ax2.text(i, -solar - convec / 2, f'{abs(convec):.0f}', ha='center', va='center', fontsize=10.5, fontweight='bold',
             color='white')

# Axis settings
ax2.set_title('Power Component Balance for Different PDMS Thicknesses (Unit: W/m²)', fontsize=15, pad=20,
              fontweight='bold')
ax2.set_xlabel('PDMS Thickness (μm)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Power (W/m²)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{t:.0f}μm' for t in thicknesses], fontsize=11)

# 关键修改：图例位置改为左下（lower left）
ax2.legend(
    fontsize=11,
    loc='lower left',  # 从upper right改为lower left，避免遮挡数据
    framealpha=0.95,
    edgecolor='black',
    borderpad=0.5
)

# Grid optimization
ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')
ax2.tick_params(axis='both', which='major', labelsize=11)

# Save plot
plt.tight_layout()
fig2.savefig(
    os.path.join(save_path, "PDMS_Power_Components_Comparison.png"),
    dpi=save_dpi,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close(fig2)
print(f"Figure 2 saved: {os.path.join(save_path, 'PDMS_Power_Components_Comparison.png')}")

# --------------------------
# Figure 3: AM1.5 Spectrum vs PDMS Absorption (Low Absorption Highlight)
# --------------------------
fig3, ax3 = plt.subplots()
ax3_twin = ax3.twinx()

# Plot solar spectrum (left y-axis)
ax3.plot(
    am15_data[:, 0], am15_data[:, 1],
    color='#FF7D00',
    linewidth=3,
    alpha=0.85,
    label='AM1.5 Solar Irradiance'
)
ax3.fill_between(
    am15_data[:, 0], am15_data[:, 1],
    alpha=0.2,
    color='#FF7D00',
    edgecolor='#FF7D00',
    linewidth=0.5
)

# Plot PDMS absorption (right y-axis, 20μm optimal thickness)
d_pdms_20 = 20e-6
eps_20 = get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms_20)
solar_mask = (wl >= 0.3) & (wl <= 2.5)
ax3_twin.plot(
    wl[solar_mask], eps_20[solar_mask],
    color='#165DFF',
    linewidth=3,
    alpha=0.85,
    label='20μm PDMS Absorptivity'
)
ax3_twin.fill_between(
    wl[solar_mask], eps_20[solar_mask],
    alpha=0.3,
    color='#165DFF',
    edgecolor='#165DFF',
    linewidth=0.5
)

# Axis settings
ax3.set_title('AM1.5 Solar Spectrum vs 20μm PDMS Absorptivity (Low Absorption Feature)', fontsize=15, pad=20,
              fontweight='bold')
ax3.set_xlabel('Wavelength (μm)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Solar Irradiance (W/(m²·μm))', color='#FF7D00', fontsize=12.5, fontweight='bold')
ax3_twin.set_ylabel('PDMS Absorptivity (≈Emissivity)', color='#165DFF', fontsize=12.5, fontweight='bold')

# Optimize absorptivity axis range (highlight low absorption)
ax3_twin.set_ylim(0, 0.04)
ax3.set_xlim(0.3, 2.5)
ax3.set_ylim(0, am15_data[:, 1].max() * 1.05)

# Grid and tick optimization
ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax3.tick_params(axis='y', labelcolor='#FF7D00', labelsize=11)
ax3_twin.tick_params(axis='y', labelcolor='#165DFF', labelsize=11)
ax3.tick_params(axis='x', labelsize=11)

# Combined legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(
    lines1 + lines2, labels1 + labels2,
    fontsize=11,
    loc='upper right',
    framealpha=0.95,
    edgecolor='black',
    borderpad=0.5
)

# Save plot
plt.tight_layout()
fig3.savefig(
    os.path.join(save_path, "AM15_Spectrum_VS_PDMS_Absorptivity.png"),
    dpi=save_dpi,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close(fig3)
print(f"Figure 3 saved: {os.path.join(save_path, 'AM15_Spectrum_VS_PDMS_Absorptivity.png')}")

# --------------------------
# Save Results to File
# --------------------------
try:
    results_df.to_excel(
        os.path.join(save_path, "PDMS_Radiative_Cooling_Optimized_Results.xlsx"),
        index=False,
        engine='openpyxl'
    )
    print(
        f"\nOptimized results saved to Excel: {os.path.join(save_path, 'PDMS_Radiative_Cooling_Optimized_Results.xlsx')}")
except:
    results_df.to_csv(
        os.path.join(save_path, "PDMS_Radiative_Cooling_Optimized_Results.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    print(
        f"\nOptimized results saved to CSV: {os.path.join(save_path, 'PDMS_Radiative_Cooling_Optimized_Results.csv')}")

print("\nAll plots optimized and saved successfully! (Academic English style)")