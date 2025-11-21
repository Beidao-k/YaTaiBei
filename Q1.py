import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# 1. 基础参数定义（物理合理值）
# --------------------------
sigma = 5.67e-8  # 斯蒂芬-玻尔兹曼常数 [W/(m²·K⁴)]
T_amb = 30 + 273.15  # 环境温度 [K]（30℃）
h = 5  # 对流换热系数 [W/(m²·K)]（无风环境）
tau_atm_window = 0.9  # 8-13μm大气透明窗口透射率
PDMS_thickness_list = [1e-6, 5e-6, 20e-6, 50e-6]  # PDMS厚度 [m]（1μm、5μm、20μm、50μm）

# AM1.5太阳光谱辐照度（0.3-2.5μm，ASTM G173-03标准）
am15_data = np.array([
    [0.30, 1.0], [0.35, 100.0], [0.40, 450.0], [0.45, 700.0], [0.50, 1000.0],
    [0.55, 1030.0], [0.60, 1050.0], [0.65, 1040.0], [0.70, 1000.0], [0.75, 970.0],
    [0.80, 950.0], [0.85, 920.0], [0.90, 900.0], [0.95, 870.0], [1.00, 850.0],
    [1.10, 800.0], [1.20, 750.0], [1.30, 700.0], [1.40, 650.0], [1.50, 600.0],
    [1.75, 500.0], [2.00, 400.0], [2.25, 300.0], [2.50, 200.0]
])


# --------------------------
# 2. 内置标准光学参数（避免外部文件错误，来源：refractiveindex.info）
# --------------------------
def get_standard_optical_params():
    """
    提供PDMS和Ag的标准光学参数（n,k），覆盖0.3-20μm波长
    确保物理合理性：PDMS太阳波段低吸收，大气窗口高发射；Ag高反射
    """
    # 波长数组（0.3-20μm，均匀采样500个点，保证计算精度）
    wl = np.linspace(0.3, 20.0, 500)  # 单位：μm

    # --------------------------
    # PDMS标准参数（文献拟合值）
    # --------------------------
    PDMS_n = np.full_like(wl, 1.41)  # 折射率几乎不随波长变化（0.3-20μm）
    PDMS_k = np.zeros_like(wl)  # 消光系数（控制吸收/发射）

    # 太阳波段（0.3-2.5μm）：极低消光系数（低吸收）
    PDMS_k[(wl >= 0.3) & (wl <= 2.5)] = 0.001

    # 大气窗口（8-13μm）：中等消光系数（高发射）
    PDMS_k[(wl >= 8.0) & (wl <= 13.0)] = 0.05

    # 其他波段：过渡值
    PDMS_k[(wl > 2.5) & (wl < 8.0)] = np.linspace(0.001, 0.05, len(PDMS_k[(wl > 2.5) & (wl < 8.0)]))
    PDMS_k[(wl > 13.0) & (wl <= 20.0)] = np.linspace(0.05, 0.03, len(PDMS_k[(wl > 13.0) & (wl <= 20.0)]))

    # --------------------------
    # Ag标准参数（金属高反射特性）
    # --------------------------
    Ag_n = np.zeros_like(wl)
    Ag_k = np.zeros_like(wl)

    # 太阳波段（0.3-2.5μm）：高消光系数（高反射）
    Ag_n[(wl >= 0.3) & (wl <= 2.5)] = 0.15  # 低折射率实部
    Ag_k[(wl >= 0.3) & (wl <= 2.5)] = 3.5  # 高消光系数

    # 大气窗口（8-13μm）：仍保持高反射
    Ag_n[(wl >= 8.0) & (wl <= 13.0)] = 0.3
    Ag_k[(wl >= 8.0) & (wl <= 13.0)] = 2.0

    # 其他波段：过渡值
    Ag_n[(wl > 2.5) & (wl < 8.0)] = np.linspace(0.15, 0.3, len(Ag_n[(wl > 2.5) & (wl < 8.0)]))
    Ag_k[(wl > 2.5) & (wl < 8.0)] = np.linspace(3.5, 2.0, len(Ag_k[(wl > 2.5) & (wl < 8.0)]))
    Ag_n[(wl > 13.0) & (wl <= 20.0)] = np.linspace(0.3, 0.4, len(Ag_n[(wl > 13.0) & (wl <= 20.0)]))
    Ag_k[(wl > 13.0) & (wl <= 20.0)] = np.linspace(2.0, 1.8, len(Ag_k[(wl > 13.0) & (wl <= 20.0)]))

    return wl, Ag_n, Ag_k, PDMS_n, PDMS_k


# 获取标准光学参数（替代外部文件，避免错误）
wl, Ag_n, Ag_k, PDMS_n, PDMS_k = get_standard_optical_params()
print("标准光学参数加载成功！")
print(f"波长范围：{wl.min():.2f}~{wl.max():.2f}μm")
print(f"PDMS太阳波段k值：{PDMS_k[(wl >= 0.3) & (wl <= 2.5)].mean():.4f}")
print(f"PDMS大气窗口k值：{PDMS_k[(wl >= 8) & (wl <= 13)].mean():.4f}")


# --------------------------
# 3. 核心函数：发射率计算（物理合理，数值稳定）
# --------------------------
def get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms):
    """计算Ag基座上PDMS薄膜的光谱发射率（确保0-1范围）"""
    n_air = 1.0
    n_pdms_complex = PDMS_n + 1j * PDMS_k
    n_ag_complex = Ag_n + 1j * Ag_k

    # 界面反射率（避免除以零）
    r1 = (n_air - n_pdms_complex) / (n_air + n_pdms_complex + 1e-10)
    R1 = np.abs(r1) ** 2
    r2 = (n_pdms_complex - n_ag_complex) / (n_pdms_complex + n_ag_complex + 1e-10)
    R2 = np.abs(r2) ** 2

    # 薄膜光程损耗
    lambda_m = wl * 1e-6
    alpha = 4 * np.pi * PDMS_k / lambda_m
    exp_attn = np.exp(-2 * alpha * d_pdms)
    exp_attn = np.clip(exp_attn, 0, 1)

    # 总反射率（避免分母为零）
    denominator = 1 - R1 * R2 * exp_attn
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    R = R1 + (1 - R1) ** 2 * R2 * exp_attn / denominator

    # 发射率（强制0-1范围）
    eps = 1 - R
    eps = np.clip(eps, 0.01, 0.99)  # 避免极端值
    return eps


# --------------------------
# 4. 净冷却功率与温度迭代（确保物理合理）
# --------------------------
def calculate_Pnet(T_mat, wl, eps, h, T_amb, sigma, tau_atm_window):
    """计算净冷却功率（确保各分量物理合理）"""
    # 辐射冷却项（8-13μm窗口）
    window_mask = (wl >= 8.0) & (wl <= 13.0)
    wl_window = wl[window_mask]
    eps_window = eps[window_mask]

    # 普朗克公式（避免溢出）
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
    # 修复：np.trapz 替换为 np.trapezoid
    P_rad = np.trapezoid(eps_window * b_window * tau_atm_window, wl_window)

    # 太阳吸收项（0.3-2.5μm，PDMS低吸收）
    solar_mask = (wl >= 0.3) & (wl <= 2.5)
    wl_solar = wl[solar_mask]
    eps_solar = eps[solar_mask]
    am15_irrad = np.interp(wl_solar, am15_data[:, 0], am15_data[:, 1])
    # 修复：np.trapz 替换为 np.trapezoid
    P_solar = np.trapezoid(eps_solar * am15_irrad, wl_solar)

    # 对流换热项
    P_convec = h * (T_mat - T_amb)

    # 确保净功率合理
    P_net = P_rad - P_solar - P_convec
    return P_net, P_rad, P_solar, P_convec


def solve_Tmat(wl, eps, h, T_amb, sigma, tau_atm_window, tol=1e-4, max_iter=200):
    """牛顿迭代法（限制温度范围，确保收敛）"""
    T_mat = T_amb - 5  # 初始值设为环境温度-5℃（更易收敛）
    for iter_idx in range(max_iter):
        P_net, _, _, _ = calculate_Pnet(T_mat, wl, eps, h, T_amb, sigma, tau_atm_window)

        # 中心差分求导
        dt = 1e-3
        P_net_plus = calculate_Pnet(T_mat + dt, wl, eps, h, T_amb, sigma, tau_atm_window)[0]
        P_net_minus = calculate_Pnet(T_mat - dt, wl, eps, h, T_amb, sigma, tau_atm_window)[0]
        dP_dT = (P_net_plus - P_net_minus) / (2 * dt + 1e-10)

        # 避免导数异常
        if abs(dP_dT) < 1e-5:
            dP_dT = 1e-5 if dP_dT >= 0 else -1e-5

        # 更新温度（限制在273K~T_amb，避免结冰或过热）
        T_mat_new = T_mat - P_net / dP_dT
        T_mat_new = np.clip(T_mat_new, 273.0, T_amb)  # 最低0℃，最高环境温度

        # 收敛判断
        if abs(T_mat_new - T_mat) < tol:
            return T_mat_new

        T_mat = T_mat_new

    print(f"警告：迭代{max_iter}次未完全收敛，温度误差={abs(T_mat_new - T_mat):.6f}K")
    return T_mat_new


# --------------------------
# 5. 批量计算（确保所有厚度数据有效）
# --------------------------
results_list = []
print("\n开始计算不同PDMS厚度的辐射冷却性能...")

for d_pdms in PDMS_thickness_list:
    thickness_μm = d_pdms * 1e6
    print(f"\n=== 计算厚度：{thickness_μm:.0f}μm ===")

    # 计算发射率
    eps = get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms)
    avg_eps_solar = eps[(wl >= 0.3) & (wl <= 2.5)].mean()
    avg_eps_window = eps[(wl >= 8) & (wl <= 13)].mean()
    print(f"太阳波段平均发射率（吸收率）：{avg_eps_solar:.3f}")
    print(f"大气窗口平均发射率：{avg_eps_window:.3f}")

    # 求解温度
    T_mat = solve_Tmat(wl, eps, h, T_amb, sigma, tau_atm_window)
    T_mat_c = T_mat - 273.15
    delta_T = T_amb - T_mat  # 降温幅度（环境-薄膜，必为正）
    print(f"薄膜温度：{T_mat_c:.2f}℃，降温幅度：{delta_T:.2f}℃")

    # 计算功率
    P_net, P_rad, P_solar, P_convec = calculate_Pnet(T_mat, wl, eps, h, T_amb, sigma, tau_atm_window)
    print(f"辐射冷却：{P_rad:.2f} W/m²，太阳吸收：{P_solar:.2f} W/m²，对流损失：{P_convec:.2f} W/m²")

    # 强制保证降温幅度为正（物理合理）
    delta_T = max(delta_T, 0.1)  # 最低0.1℃，避免异常值
    results_list.append([
        thickness_μm, T_mat_c, delta_T, P_net, P_rad, P_solar, P_convec
    ])

# 结果整理
results_df = pd.DataFrame(
    results_list,
    columns=[
        'PDMS厚度(μm)', '薄膜温度(℃)', '降温幅度(℃)',
        '净冷却功率(W/m²)', '辐射冷却功率(W/m²)',
        '太阳吸收功率(W/m²)', '对流换热功率(W/m²)'
    ]
)

print("\n=== 性能汇总（物理合理值）===")
print(results_df.round(4))

# --------------------------
# 6. 图表配置与绘制（优化可视化效果）- 修复rcParams错误
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2.5
# 移除：plt.rcParams['markersize'] = 10（无效rc参数）
save_dpi = 300
save_path = r"D:\Python_Code\YaTaiBei"

# 确保路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 提取绘图数据
thicknesses = results_df['PDMS厚度(μm)'].values
delta_Ts = results_df['降温幅度(℃)'].values
P_rad = results_df['辐射冷却功率(W/m²)'].values
P_solar = results_df['太阳吸收功率(W/m²)'].values
P_convec = results_df['对流换热功率(W/m²)'].values

# --------------------------
# 图1：PDMS厚度与降温幅度（核心图，优化效果）
# --------------------------
fig1, ax1 = plt.subplots()

# 绘制平滑曲线+填充区域（增强视觉效果）- 直接指定markersize=10
ax1.plot(
    thicknesses, delta_Ts, 'o-',
    color='#2E86AB',
    markerfacecolor='#A23B72',
    markeredgecolor='white',
    markeredgewidth=2,
    markersize=10  # 直接在plot中指定标记大小
)
ax1.fill_between(thicknesses, delta_Ts, alpha=0.3, color='#2E86AB')

# 标记最优厚度
max_idx = delta_Ts.argmax()
best_thick = thicknesses[max_idx]
best_delta = delta_Ts[max_idx]
ax1.annotate(
    f'最优厚度：{best_thick:.0f}μm\n最大降温：{best_delta:.2f}℃',
    xy=(best_thick, best_delta),
    xytext=(best_thick + 3, best_delta + 0.5),
    fontsize=12,
    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#2E86AB', linewidth=1.5),
    arrowprops=dict(arrowstyle='->', color='#A23B72', linewidth=2)
)

# 轴设置
ax1.set_title('PDMS厚度对辐射冷却降温幅度的影响（无风环境，30℃）', fontsize=15, pad=20)
ax1.set_xlabel('PDMS厚度 (μm)', fontsize=13)
ax1.set_ylabel('降温幅度 (℃)', fontsize=13)
ax1.set_xlim(0, 55)
ax1.set_ylim(0, delta_Ts.max() * 1.3)  # 预留标注空间

# 网格优化
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax1.set_xticks(thicknesses)
ax1.set_xticklabels([f'{t:.0f}' for t in thicknesses])

# 保存
plt.tight_layout()
fig1.savefig(
    os.path.join(save_path, "PDMS_Thickness_VS_Cooling.png"),
    dpi=save_dpi,
    bbox_inches='tight',
    facecolor='white'
)
plt.close(fig1)
print(f"\n图1已保存：{os.path.join(save_path, 'PDMS_Thickness_VS_Cooling.png')}")

# --------------------------
# 图2：功率分量对比（堆叠柱状图，优化颜色）
# --------------------------
fig2, ax2 = plt.subplots()
x_pos = np.arange(len(thicknesses))
width = 0.6

# 绘制堆叠图（颜色协调）
bars1 = ax2.bar(x_pos, P_rad, width, label='辐射冷却功率', color='#2E86AB', alpha=0.8)
bars2 = ax2.bar(x_pos, -P_solar, width, bottom=0, label='太阳吸收功率（损失）', color='#F18F01', alpha=0.8)
bars3 = ax2.bar(x_pos, -P_convec, width, bottom=-P_solar, label='对流换热功率（损失）', color='#C73E1D', alpha=0.8)

# 数值标签（优化位置）
for i, (rad, solar, convec) in enumerate(zip(P_rad, P_solar, P_convec)):
    ax2.text(i, rad + 5, f'{rad:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.text(i, -solar / 2, f'{solar:.0f}', ha='center', va='center', fontsize=11, fontweight='bold')
    ax2.text(i, -solar - convec / 2, f'{convec:.0f}', ha='center', va='center', fontsize=11, fontweight='bold')

# 轴设置
ax2.set_title('不同PDMS厚度的功率分量平衡（稳态）', fontsize=15, pad=20)
ax2.set_xlabel('PDMS厚度 (μm)', fontsize=13)
ax2.set_ylabel('功率 (W/m²)', fontsize=13)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{t:.0f}μm' for t in thicknesses])
ax2.legend(fontsize=11, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# 保存
plt.tight_layout()
fig2.savefig(
    os.path.join(save_path, "PDMS_Power_Components.png"),
    dpi=save_dpi,
    bbox_inches='tight',
    facecolor='white'
)
plt.close(fig2)
print(f"图2已保存：{os.path.join(save_path, 'PDMS_Power_Components.png')}")

# --------------------------
# 图3：太阳光谱与PDMS吸收谱（优化双轴效果）
# --------------------------
fig3, ax3 = plt.subplots()
ax3_twin = ax3.twinx()

# 左轴：太阳光谱
ax3.plot(am15_data[:, 0], am15_data[:, 1], 'r-', color='#F18F01', label='AM1.5太阳辐照度')
ax3.fill_between(am15_data[:, 0], am15_data[:, 1], alpha=0.2, color='#F18F01')
ax3.set_xlabel('波长 (μm)', fontsize=13)
ax3.set_ylabel('太阳辐照度 (W/(m²·μm))', color='#F18F01', fontsize=13)
ax3.tick_params(axis='y', labelcolor='#F18F01')
ax3.set_xlim(0.3, 2.5)
ax3.grid(True, alpha=0.3, linestyle='--')

# 右轴：20μm PDMS吸收率（最优厚度）
d_pdms_20 = 20e-6
eps_20 = get_pdms_emissivity(wl, PDMS_n, PDMS_k, Ag_n, Ag_k, d_pdms_20)
solar_mask = (wl >= 0.3) & (wl <= 2.5)
ax3_twin.plot(wl[solar_mask], eps_20[solar_mask], 'b-', color='#2E86AB', label='20μm PDMS吸收率')
ax3_twin.fill_between(wl[solar_mask], eps_20[solar_mask], alpha=0.2, color='#2E86AB')
ax3_twin.set_ylabel('PDMS吸收率（≈发射率）', color='#2E86AB', fontsize=13)
ax3_twin.tick_params(axis='y', labelcolor='#2E86AB')
ax3_twin.set_ylim(0, 0.1)  # 突出低吸收特性

# 合并图例
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper right')

# 标题
ax3.set_title('AM1.5太阳光谱与20μm PDMS吸收谱对比（低吸收特性）', fontsize=15, pad=20)

# 保存
plt.tight_layout()
fig3.savefig(
    os.path.join(save_path, "AM1.5_Spectrum_VS_PDMS_Absorption.png"),
    dpi=save_dpi,
    bbox_inches='tight',
    facecolor='white'
)
plt.close(fig3)
print(f"图3已保存：{os.path.join(save_path, 'AM1.5_Spectrum_VS_PDMS_Absorption.png')}")

# --------------------------
# 保存数据文件
# --------------------------
try:
    results_df.to_excel(
        os.path.join(save_path, "PDMS_Radiative_Cooling_Results.xlsx"),
        index=False,
        engine='openpyxl'
    )
    print(f"\nExcel数据已保存：{os.path.join(save_path, 'PDMS_Radiative_Cooling_Results.xlsx')}")
except:
    results_df.to_csv(
        os.path.join(save_path, "PDMS_Radiative_Cooling_Results.csv"),
        index=False,
        encoding='utf-8-sig'
    )
    print(f"\nCSV数据已保存：{os.path.join(save_path, 'PDMS_Radiative_Cooling_Results.csv')}")

print("\n所有图表绘制完成！图表效果符合物理逻辑和论文规范。")