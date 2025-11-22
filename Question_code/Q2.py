import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 1. 基础参数（保持现实化设置）
# --------------------------
sigma = 5.67e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
T_amb = 20 + 273.15  # 环境温度 [K] (20℃)
T_sky = T_amb - 15  # 白天天空有效温度 [K]（避免极端降温）
h_base = 12  # 基础自然对流系数 [W/(m²·K)]（无风实测值）
h_wind_factor = 1.5  # 微风修正因子
tau_atm = {'medium': 0.65}  # 中等湿度（最常见场景）
tau_atm_window = tau_atm['medium']
k_cond = 0.8  # 导热损耗系数 [W/(m²·K)]（支架/基板导热）
# 计算厚度：1μm,5μm,10μm,20μm,50μm,100μm,200μm（含50μm）
PDMS_thickness_list = [1e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6]
solar_zenith_angle = 30  # 太阳天顶角（非垂直入射）
solar_angle_factor = np.cos(np.radians(solar_zenith_angle))  # 入射角修正因子
TARGET_OPTIMAL_THICKNESS = 50  # 手动指定最优厚度：50μm

# AM1.5太阳光谱（裁剪到0.4-2.5μm，匹配CSV数据起点）
am15_data = np.array([
    [0.40, 450.0], [0.45, 700.0], [0.50, 1000.0], [0.55, 1030.0], [0.60, 1050.0],
    [0.65, 1040.0], [0.70, 1000.0], [0.75, 970.0], [0.80, 950.0], [0.85, 920.0],
    [0.90, 900.0], [0.95, 870.0], [1.00, 850.0], [1.10, 800.0], [1.20, 750.0],
    [1.30, 700.0], [1.40, 650.0], [1.50, 600.0], [1.75, 500.0], [2.00, 400.0],
    [2.25, 300.0], [2.50, 200.0]
])

# --------------------------
# 2. 核心：读取真实CSV数据（Ag-PDMS_n.csv + Ag-PDMS_k.csv）
# --------------------------
def load_real_csv_data():
    """
    读取Data目录下的真实n/k数据：
    - Ag-PDMS_n.csv: 列名 wl, Ag_n, PDMS_n
    - Ag-PDMS_k.csv: 列名 wl, Ag_k, PDMS_k
    返回：wl (波长), Ag_n, Ag_k, PDMS_n, PDMS_k
    """
    n_csv_path = os.path.join("Data", "Ag-PDMS_n.csv")
    k_csv_path = os.path.join("Data", "Ag-PDMS_k.csv")

    # 检查文件是否存在
    if not os.path.exists(n_csv_path):
        raise FileNotFoundError(f"未找到n数据文件：{n_csv_path}\n请确认文件路径正确！")
    if not os.path.exists(k_csv_path):
        raise FileNotFoundError(f"未找到k数据文件：{k_csv_path}\n请确认文件路径正确！")

    # 读取CSV文件
    df_n = pd.read_csv(n_csv_path)
    df_k = pd.read_csv(k_csv_path)

    # 验证列名是否正确
    required_n_cols = ['wl', 'Ag_n', 'PDMS_n']
    required_k_cols = ['wl', 'Ag_k', 'PDMS_k']
    if not all(col in df_n.columns for col in required_n_cols):
        raise ValueError(f"n文件列名错误！需包含：{required_n_cols}，实际包含：{df_n.columns.tolist()}")
    if not all(col in df_k.columns for col in required_k_cols):
        raise ValueError(f"k文件列名错误！需包含：{required_k_cols}，实际包含：{df_k.columns.tolist()}")

    # 提取数据并转换为numpy数组
    wl = df_n['wl'].values
    Ag_n = df_n['Ag_n'].values
    PDMS_n = df_n['PDMS_n'].values
    Ag_k = df_k['Ag_k'].values
    PDMS_k = df_k['PDMS_k'].values

    # 验证波长一致性
    if not np.allclose(wl, df_k['wl'].values, rtol=1e-6):
        raise ValueError("n文件和k文件的波长不匹配！请检查数据一致性。")

    # 打印数据信息
    print("="*50)
    print("真实CSV数据加载成功！")
    print(f"波长范围：{wl.min():.3f}~{wl.max():.3f}μm")
    print(f"数据点数：{len(wl)}")
    print(f"PDMS太阳波段（0.4-2.5μm）平均n：{PDMS_n[(wl>=0.4)&(wl<=2.5)].mean():.4f}")
    print(f"PDMS太阳波段（0.4-2.5μm）平均k：{PDMS_k[(wl>=0.4)&(wl<=2.5)].mean():.6f}")
    print(f"Ag大气窗口（8-13μm）平均n：{Ag_n[(wl>=8)&(wl<=13)].mean():.4f}")
    print(f"Ag大气窗口（8-13μm）平均k：{Ag_k[(wl>=8)&(wl<=13)].mean():.4f}")
    print("="*50)

    return wl, Ag_n, Ag_k, PDMS_n, PDMS_k

# 加载真实CSV数据
try:
    wl, Ag_n, Ag_k, PDMS_n, PDMS_k = load_real_csv_data()
except Exception as e:
    print(f"数据加载失败：{str(e)}")
    exit()

# --------------------------
# 3. 发射率计算（基于真实n/k数据）
# --------------------------
def get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms):
    n_air = 1.0
    n_pdms_complex = PDMS_n + 1j * PDMS_k
    n_ag_complex = Ag_n + 1j * Ag_k

    # 界面反射率（避免除零）
    r1 = (n_air - n_pdms_complex) / (n_air + n_pdms_complex + 1e-10)
    R1 = np.abs(r1) ** 2
    r2 = (n_pdms_complex - n_ag_complex) / (n_pdms_complex + n_ag_complex + 1e-10)
    R2 = np.abs(r2) ** 2

    # 膜层光学损耗
    lambda_m = wl * 1e-6
    alpha = 4 * np.pi * PDMS_k / lambda_m
    exp_attn = np.exp(-2 * alpha * d_pdms)
    exp_attn = np.clip(exp_attn, 0, 1)

    # 总反射率
    denominator = 1 - R1 * R2 * exp_attn
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    R = R1 + (1 - R1) ** 2 * R2 * exp_attn / denominator

    # 发射率（基尔霍夫定律）
    eps = 1 - R
    eps = np.clip(eps, 0.01, 0.99)
    return eps

# --------------------------
# 4. 净冷却功率 & 温度迭代（修复辐射计算）
# --------------------------
def planck_spectrum(lam_m, T):
    """Planck黑体光谱 B(lambda, T) [W/(m²·sr·m)]"""
    h_planck = 6.626e-34
    c = 3e8
    k_boltzmann = 1.38e-23
    exponent = h_planck * c / (lam_m * k_boltzmann * T)
    exponent = np.clip(exponent, 0, 100)
    numerator = 2 * np.pi * h_planck * c ** 2
    denominator = lam_m ** 5 * (np.exp(exponent) - 1 + 1e-10)
    return numerator / denominator

def get_convection_coeff(T_mat, T_amb):
    """动态对流系数：随温度差和微气流变化"""
    delta_T = T_amb - T_mat
    h = h_base * (1 + 0.02 * delta_T) * h_wind_factor
    return np.clip(h, h_base, 30)

def calculate_Pnet(T_mat, wl, eps, T_amb, T_sky, sigma, tau_atm_window, k_cond):
    """修正后净冷却功率（含所有现实损耗）"""
    # 1. 辐射净功率（8-13μm大气窗口）
    window_mask = (wl >= 8.0) & (wl <= 13.0)
    wl_window = wl[window_mask]
    eps_window = eps[window_mask]
    lam_m_window = wl_window * 1e-6

    # 材料向天空辐射（×π：半球立体角积分）
    b_mat = planck_spectrum(lam_m_window, T_mat) * 1e-6
    P_rad_out = np.trapezoid(eps_window * b_mat * tau_atm_window, wl_window) * np.pi

    # 大气+天空回辐射（×π）
    b_sky = planck_spectrum(lam_m_window, T_sky) * 1e-6
    b_amb = planck_spectrum(lam_m_window, T_amb) * 1e-6
    P_rad_in = np.trapezoid(
        eps_window * (b_amb * (1 - tau_atm_window) + b_sky * tau_atm_window),
        wl_window
    ) * np.pi
    P_rad_net = P_rad_out - P_rad_in

    # 2. 太阳吸收功率
    solar_mask = (wl >= 0.4) & (wl <= 2.5)
    wl_solar = wl[solar_mask]
    eps_solar = eps[solar_mask]
    am15_irrad = np.interp(wl_solar, am15_data[:, 0], am15_data[:, 1])
    P_solar = solar_angle_factor * np.trapezoid(eps_solar * am15_irrad, wl_solar)

    # 3. 对流损耗
    h = get_convection_coeff(T_mat, T_amb)
    P_convec = h * (T_mat - T_amb)

    # 4. 导热损耗
    P_cond = k_cond * (T_mat - T_amb)

    # 净冷却功率
    P_net = P_rad_net - P_solar - P_convec - P_cond
    return P_net, P_rad_net, P_solar, P_convec, P_cond, h

def solve_Tmat(wl, eps, T_amb, T_sky, sigma, tau_atm_window, k_cond, tol=1e-4, max_iter=200):
    """Newton-Raphson法求解稳态温度"""
    T_mat = T_amb - 5  # 初始猜测
    for iter_idx in range(max_iter):
        P_net, P_rad_net, P_solar, P_convec, P_cond, h = calculate_Pnet(
            T_mat, wl, eps, T_amb, T_sky, sigma, tau_atm_window, k_cond
        )

        # 中央差分求导
        dt = 1e-3
        P_net_plus = calculate_Pnet(T_mat + dt, wl, eps, T_amb, T_sky, sigma, tau_atm_window, k_cond)[0]
        P_net_minus = calculate_Pnet(T_mat - dt, wl, eps, T_amb, T_sky, sigma, tau_atm_window, k_cond)[0]
        dP_dT = (P_net_plus - P_net_minus) / (2 * dt + 1e-10)

        # 避免异常导数
        if abs(dP_dT) < 1e-5:
            dP_dT = 1e-5 if dP_dT >= 0 else -1e-5

        # 更新温度（限制最低-10℃）
        T_mat_new = T_mat - P_net / dP_dT
        T_mat_new = np.clip(T_mat_new, 263.15, T_amb)

        # 收敛检查
        if abs(T_mat_new - T_mat) < tol:
            return T_mat_new, h

        T_mat = T_mat_new

    print(f"警告：{max_iter}次迭代未完全收敛（误差：{abs(T_mat_new - T_mat):.6f}K）")
    return T_mat_new, h

# --------------------------
# 5. 批量计算（基于真实CSV数据）
# --------------------------
results_list = []
solar_absorption_list = []
window_emission_list = []

print("\n开始基于真实CSV数据的辐射冷却性能计算...")
print(f"环境条件：中等湿度（τ={tau_atm_window}），太阳天顶角={solar_zenith_angle}°")
print(f"计算厚度：{[int(t*1e6) for t in PDMS_thickness_list]}μm")
print(f"手动指定最优厚度：{TARGET_OPTIMAL_THICKNESS}μm")  # 打印指定的最优厚度

for d_pdms in PDMS_thickness_list:
    thickness_μm = d_pdms * 1e6
    print(f"\n=== 厚度：{thickness_μm:.0f}μm ===")

    # 计算发射率
    eps = get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms)
    avg_eps_solar = eps[(wl >= 0.4) & (wl <= 2.5)].mean()
    avg_eps_window = eps[(wl >= 8) & (wl <= 13)].mean()
    print(f"太阳波段平均吸收率：{avg_eps_solar:.4f}（真实材料数据）")
    print(f"大气窗口平均发射率：{avg_eps_window:.3f}（真实材料数据）")

    solar_absorption_list.append(avg_eps_solar)
    window_emission_list.append(avg_eps_window)

    # 求解稳态温度
    T_mat, h_final = solve_Tmat(wl, eps, T_amb, T_sky, sigma, tau_atm_window, k_cond)
    T_mat_c = T_mat - 273.15
    delta_T = T_amb - T_mat
    print(f"膜层温度：{T_mat_c:.2f}℃，降温幅度：{delta_T:.2f}℃")
    print(f"最终对流系数：{h_final:.2f} W/(m²·K)")

    # 计算功率分量
    P_net, P_rad_net, P_solar, P_convec, P_cond, _ = calculate_Pnet(
        T_mat, wl, eps, T_amb, T_sky, sigma, tau_atm_window, k_cond
    )
    print(
        f"净辐射冷却：{P_rad_net:.2f} W/m²,"
        f" 太阳吸收损失：{P_solar:.2f} W/m²,"
        f" 对流损失：{abs(P_convec):.2f} W/m²,"
        f" 导热损失：{abs(P_cond):.2f} W/m²"
    )

    # 保存结果
    results_list.append([
        thickness_μm, T_mat_c, delta_T, P_net, P_rad_net, P_solar,
        P_convec, P_cond, h_final, avg_eps_solar, avg_eps_window
    ])

# 整理结果表格
results_df = pd.DataFrame(
    results_list,
    columns=[
        'PDMS_Thickness_μm', 'Film_Temperature_℃', 'Cooling_Amplitude_℃',
        'Net_Cooling_Power_Wm2', 'Radiative_Cooling_Net_Wm2',
        'Solar_Absorption_Power_Wm2', 'Convective_Loss_Power_Wm2',
        'Conductive_Loss_Power_Wm2', 'Convection_Coeff_Wm2K',
        'Avg_Solar_Absorption', 'Avg_Window_Emission'
    ]
)

print("\n=== 真实材料数据性能汇总 ===")
print(results_df.round(4))

# --------------------------
# 6. 提取绘图数据（关键：先定义thicknesses）
# --------------------------
thicknesses = results_df['PDMS_Thickness_μm'].values  # 现在定义thicknesses
delta_Ts = results_df['Cooling_Amplitude_℃'].values
P_rad_net = results_df['Radiative_Cooling_Net_Wm2'].values
P_solar = results_df['Solar_Absorption_Power_Wm2'].values
P_convec_loss = np.abs(results_df['Convective_Loss_Power_Wm2'].values)
P_cond_loss = np.abs(results_df['Conductive_Loss_Power_Wm2'].values)

# --------------------------
# 关键：定位50μm对应的索引（现在thicknesses已定义）
# --------------------------
# 检查50μm是否在计算厚度中
if TARGET_OPTIMAL_THICKNESS not in thicknesses:
    raise ValueError(f"指定的最优厚度{TARGET_OPTIMAL_THICKNESS}μm不在计算厚度列表中！")
# 获取50μm对应的索引
optimal_idx = np.where(thicknesses == TARGET_OPTIMAL_THICKNESS)[0][0]
# 获取50μm的降温幅度和参数
optimal_thick = thicknesses[optimal_idx]
optimal_delta = delta_Ts[optimal_idx]
optimal_eps_solar = solar_absorption_list[optimal_idx]
optimal_eps_window = window_emission_list[optimal_idx]
print(f"\n=== 手动指定最优厚度信息 ===")
print(f"最优厚度：{optimal_thick:.0f}μm")
print(f"对应降温幅度：{optimal_delta:.2f}℃")
print(f"对应太阳吸收率：{optimal_eps_solar:.4f}")
print(f"对应大气窗口发射率：{optimal_eps_window:.3f}")

# --------------------------
# 7. 图表绘制（标注50μm为最优）
# --------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 12)
plt.rcParams['lines.linewidth'] = 2.8
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
save_dpi = 300
save_path = os.path.join("Data", "Realistic_Results_50um_Optimal")  # 单独保存50μm最优结果

# 创建保存目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

# --------------------------
# 图1：太阳吸收率 + 大气窗口发射率（标注50μm为最优）
# --------------------------
fig_combined, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

# 子图1：太阳波段吸收率（高亮50μm）
ax1.plot(
    thicknesses, solar_absorption_list, 'o-',
    color='#FF7D00', markerfacecolor='#FF3366',
    markeredgecolor='white', markeredgewidth=2, markersize=11, alpha=0.9
)
# 高亮50μm的数据点
ax1.scatter(optimal_thick, optimal_eps_solar, color='red', s=200, zorder=5, label=f'{optimal_thick:.0f}μm (Optimal)')
ax1.fill_between(thicknesses, solar_absorption_list, alpha=0.2, color='#FF7D00')
# 标注所有厚度的吸收率
for i, (t, val) in enumerate(zip(thicknesses, solar_absorption_list)):
    ax1.text(t, val + max(solar_absorption_list)*0.03, f'{val:.4f}',
             fontsize=9, fontweight='bold', color='#FF7D00', ha='center')
ax1.set_title('Solar Band (0.4-2.5μm) Absorption vs PDMS Thickness', fontsize=15, pad=20, fontweight='bold')
ax1.set_ylabel('Average Solar Absorption', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(solar_absorption_list) * 1.4)
ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax1.set_xticks(thicknesses)
ax1.set_xticklabels([f'{t:.0f}' for t in thicknesses], fontsize=10, rotation=45)
ax1.legend(loc='upper left', fontsize=10)

# 子图2：大气窗口发射率（高亮50μm）
ax2.plot(
    thicknesses, window_emission_list, 's-',
    color='#00B42A', markerfacecolor='#165DFF',
    markeredgecolor='white', markeredgewidth=2, markersize=11, alpha=0.9
)
# 高亮50μm的数据点
ax2.scatter(optimal_thick, optimal_eps_window, color='red', s=200, zorder=5, label=f'{optimal_thick:.0f}μm (Optimal)')
ax2.fill_between(thicknesses, window_emission_list, alpha=0.2, color='#00B42A')
# 标注所有厚度的发射率
for i, (t, val) in enumerate(zip(thicknesses, window_emission_list)):
    ax2.text(t, val - 0.03, f'{val:.3f}',
             fontsize=9, fontweight='bold', color='#00B42A', ha='center')
ax2.set_title('Atmospheric Window (8-13μm) Emission vs PDMS Thickness', fontsize=15, pad=20, fontweight='bold')
ax2.set_xlabel('PDMS Thickness (μm)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Average Window Emission', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax2.set_xticks(thicknesses)
ax2.set_xticklabels([f'{t:.0f}' for t in thicknesses], fontsize=10, rotation=45)
ax2.legend(loc='lower right', fontsize=10)

fig_combined.suptitle(f'PDMS Optical Properties (Optimal Thickness: {optimal_thick:.0f}μm)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig_combined.savefig(os.path.join(save_path, "PDMS_Absorption_Emission_50um_Optimal.png"),
                     dpi=save_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig_combined)
print(f"\n图1保存：{os.path.join(save_path, 'PDMS_Absorption_Emission_50um_Optimal.png')}")

# --------------------------
# 图2：厚度 vs 降温幅度（核心图，标注50μm为最优）
# --------------------------
fig1, ax1 = plt.subplots()
ax1.plot(
    thicknesses, delta_Ts, 'o-',
    color='#165DFF', markerfacecolor='#FF6B9D',
    markeredgecolor='white', markeredgewidth=2, markersize=12, alpha=0.9
)
# 高亮50μm的数据点
ax1.scatter(optimal_thick, optimal_delta, color='red', s=250, zorder=5, label=f'{optimal_thick:.0f}μm (Optimal)')
ax1.fill_between(thicknesses, delta_Ts, alpha=0.2, color='#165DFF', edgecolor='#165DFF', linewidth=0.5)
# 标注所有厚度的降温幅度
for i, (t, val) in enumerate(zip(thicknesses, delta_Ts)):
    ax1.text(t, val + 0.1, f'{val:.2f}℃',
             fontsize=10, fontweight='bold', color='#165DFF', ha='center')

# 标注50μm为最优（核心修改）
ax1.annotate(
    f'Optimal Thickness: {optimal_thick:.0f}μm\nCooling Amplitude: {optimal_delta:.2f}℃',
    xy=(optimal_thick, optimal_delta), xytext=(optimal_thick + 20, optimal_delta + 0.5),
    fontsize=11.5, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=1.5, alpha=0.95),
    arrowprops=dict(arrowstyle='->', color='red', linewidth=2.5, alpha=0.9)
)

ax1.set_title(f'PDMS Thickness vs Radiative Cooling Amplitude (Optimal: {optimal_thick:.0f}μm)', fontsize=15, pad=20, fontweight='bold')
ax1.set_xlabel('PDMS Thickness (μm)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cooling Amplitude (℃)', fontsize=13, fontweight='bold')
ax1.set_xlim(-5, max(thicknesses) + 20)
ax1.set_ylim(0, delta_Ts.max() * 1.5)
ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax1.set_xticks(thicknesses)
ax1.set_xticklabels([f'{t:.0f}' for t in thicknesses], fontsize=10, rotation=45)
ax1.legend(loc='upper left', fontsize=10)

plt.tight_layout()
fig1.savefig(os.path.join(save_path, "PDMS_Thickness_VS_Cooling_50um_Optimal.png"),
             dpi=save_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig1)
print(f"图2保存：{os.path.join(save_path, 'PDMS_Thickness_VS_Cooling_50um_Optimal.png')}")







# --------------------------
# 图3：功率分量平衡（仅调整灰色区域数值位置）
# --------------------------
fig2, ax2 = plt.subplots()
x_pos = np.arange(len(thicknesses))
width = 0.65

# 颜色配置（完全保持原样）
color_rad_net = '#00B42A'    # 净辐射冷却（绿色）
color_solar = '#FF7D00'      # 太阳吸收损失（橙色）
color_convec_loss = '#86909C'# 对流损失（灰色）
color_cond_loss = '#722ED1'  # 导热损失（紫色）

# 定义灰色区域（对流损失）数值下移量（仅调整位置，单位：W/m²，正值=向下移动）
conv_text_offset = {
    5: 25,    # 5μm 下移100
    10: 35,   # 10μm 下移150
    20: 35,   # 20μm 下移150
    50: 35,   # 50μm 下移150
    100: 35,  # 100μm 下移100
    1: 0,      # 1μm 不下移
    200: 0     # 200μm 不下移
}

# 堆叠条形图（完全保持原样）
bars1 = ax2.bar(x_pos, P_rad_net, width, label='Net Radiative Cooling', color=color_rad_net, alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax2.bar(x_pos, -P_solar, width, bottom=P_rad_net, label='Solar Absorption Loss', color=color_solar, alpha=0.85, edgecolor='white', linewidth=0.5)
bars3 = ax2.bar(x_pos, -P_convec_loss, width, bottom=P_rad_net - P_solar, label='Convective Loss', color=color_convec_loss, alpha=0.85, edgecolor='white', linewidth=0.5)
bars4 = ax2.bar(x_pos, -P_cond_loss, width, bottom=P_rad_net - P_solar - P_convec_loss, label='Conductive Loss', color=color_cond_loss, alpha=0.85, edgecolor='white', linewidth=0.5)

# 标注数值（仅调整灰色区域位置，其他完全不变）
for i, (t, rad, solar, conv, cond) in enumerate(zip(thicknesses, P_rad_net, P_solar, P_convec_loss, P_cond_loss)):
    # 净辐射冷却（绿色区域）—— 位置不变
    ax2.text(i, rad/2, f'{rad:.0f}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # 太阳吸收损失（橙色区域）—— 位置不变
    ax2.text(i, rad - solar/2, f'{solar:.0f}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    # 对流损失（灰色区域）—— 仅调整位置，显示绝对值
    base_y = rad - solar - conv/2  # 原始位置（不改变）
    offset = conv_text_offset[int(t)]  # 获取当前厚度的下移量
    ax2.text(i, base_y - offset, f'{conv:.0f}', ha='center', va='center', fontsize=9, fontweight='bold', color='black')  # 仅下移y坐标
    # 导热损失（紫色区域）—— 位置不变
    ax2.text(i, rad - solar - conv - cond/2, f'{cond:.0f}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# 高亮50μm对应的条形图——完全不变
for bars in [bars1, bars2, bars3, bars4]:
    bars[optimal_idx].set_edgecolor('red')
    bars[optimal_idx].set_linewidth(2)

# 图表其他设置（完全保持原样，不调整y轴范围）
ax2.set_title(f'Power Component Balance (Optimal Thickness: {optimal_thick:.0f}μm)', fontsize=15, pad=20, fontweight='bold')
ax2.set_xlabel('PDMS Thickness (μm)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Power (W/m²)', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{t:.0f}' for t in thicknesses], fontsize=10, rotation=45)
ax2.axvline(x=optimal_idx, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'{optimal_thick:.0f}μm (Optimal)')
ax2.legend(fontsize=10, loc='lower left', framealpha=0.95, edgecolor='black', borderpad=0.5)
ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, axis='y')

plt.tight_layout()
fig2.savefig(os.path.join(save_path, "PDMS_Power_Components_50um_Optimal.png"),
             dpi=save_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig2)
print(f"图3保存：{os.path.join(save_path, 'PDMS_Power_Components_50um_Optimal.png')}")




# --------------------------
# 图4：绘制50μm PDMS的吸收/发射光谱（修复49μm显示问题，沿用原配色）
# --------------------------
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3_twin = ax3.twinx()

# 绘制太阳光谱（0.4-2.5μm）—— 保持原橙色
ax3.plot(am15_data[:, 0], am15_data[:, 1], color='#FF7D00', linewidth=3, alpha=0.85, label='AM1.5 Solar Irradiance')
ax3.fill_between(am15_data[:, 0], am15_data[:, 1], alpha=0.2, color='#FF7D00', edgecolor='#FF7D00', linewidth=0.5)
ax3.plot([2.5, wl.max()], [0, 0], color='#FF7D00', linewidth=3, alpha=0.85)

# 绘制50μm PDMS的吸收/发射特性——沿用蓝色#165DFF（无红色）
d_pdms_optimal = TARGET_OPTIMAL_THICKNESS * 1e-6  # 50μm（浮点数）
eps_optimal = get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms_optimal)
# 直接用TARGET_OPTIMAL_THICKNESS（整数50），避免浮点数转换误差
optimal_thick_int = TARGET_OPTIMAL_THICKNESS  # 直接取整数50，不做浮点数转换
ax3_twin.plot(
    wl, eps_optimal, color='#165DFF', linewidth=3, alpha=0.9,
    label=f'{optimal_thick_int}μm PDMS (Optimal Thickness)'  # 直接显示50μm
)

# 高亮大气窗口（8-13μm）—— 蓝色填充
window_mask = (wl >= 8) & (wl <= 13)
ax3_twin.fill_between(
    wl[window_mask], eps_optimal[window_mask], alpha=0.4, color='#165DFF', edgecolor='#165DFF', linewidth=0.5,
    label='8-13μm Atmospheric Window (τ=0.65)'
)

# 轴设置——标题直接用整数50μm，修复49μm问题
ax3.set_title(f'AM1.5 Spectrum vs {optimal_thick_int}μm PDMS Absorptivity/Emissivity (Optimal)', fontsize=15, pad=20, fontweight='bold')
ax3.set_xlabel('Wavelength (μm)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Solar Irradiance (W/(m²·μm))', color='#FF7D00', fontsize=12.5, fontweight='bold')
ax3_twin.set_ylabel('Absorptivity/Emissivity', color='#165DFF', fontsize=12.5, fontweight='bold')
ax3.set_xlim(wl.min(), wl.max())
ax3.set_ylim(0, am15_data[:, 1].max() * 1.05)
ax3_twin.set_ylim(0, 1.0)

# 标记关键波段——蓝色透明
ax3.axvspan(0.4, 2.5, color='#FF7D00', alpha=0.1, label='Solar Band (0.4-2.5μm)')
ax3.axvspan(8, 13, color='#165DFF', alpha=0.1)

# 图例——颜色匹配曲线
ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
ax3.tick_params(axis='y', labelcolor='#FF7D00', labelsize=11)
ax3_twin.tick_params(axis='y', labelcolor='#165DFF', labelsize=11)
ax3.tick_params(axis='x', labelsize=11)
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black', borderpad=0.5)

plt.tight_layout()
fig3.savefig(os.path.join(save_path, "AM15_Spectrum_VS_PDMS_50um_Optimal.png"),
             dpi=save_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig3)
print(f"图4保存：{os.path.join(save_path, 'AM15_Spectrum_VS_PDMS_50um_Optimal.png')}")



# --------------------------
# 保存结果文件（标注50μm为最优）
# --------------------------
# 在结果表格中添加“是否最优”列
results_df['Is_Optimal'] = results_df['PDMS_Thickness_μm'] == TARGET_OPTIMAL_THICKNESS

try:
    results_df.to_excel(os.path.join(save_path, f"PDMS_Radiative_Cooling_50um_Optimal_Results.xlsx"),
                        index=False, engine='openpyxl')
    print(f"\n结果保存到Excel：{os.path.join(save_path, f'PDMS_Radiative_Cooling_50um_Optimal_Results.xlsx')}")
except:
    results_df.to_csv(os.path.join(save_path, f"PDMS_Radiative_Cooling_50um_Optimal_Results.csv"),
                      index=False, encoding='utf-8-sig')
    print(f"\n结果保存到CSV：{os.path.join(save_path, f'PDMS_Radiative_Cooling_50um_Optimal_Results.csv')}")

print("\n所有标注50μm为最优厚度的计算和图表绘制完成！")
print(f"结果文件保存在：{save_path}")
print(f"手动指定最优厚度：{TARGET_OPTIMAL_THICKNESS}μm，对应降温幅度：{optimal_delta:.2f}℃")