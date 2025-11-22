import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from typing import List, Dict, Tuple
sigma = 5.67e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]
T_amb = 20 + 273.15  # 环境温度 [K] (20℃) —— P_net计算基准温度
T_sky = T_amb - 15  # 白天天空有效温度 [K]
h_base = 12  # 基础自然对流系数 [W/(m²·K)]
h_wind_factor = 1.5  # 微风修正因子
tau_atm = {'medium': 0.65}  # 中等湿度
tau_atm_window = tau_atm['medium']
k_cond = 0.8  # 导热损耗系数 [W/(m²·K)]（问题2支架/基板导热）
solar_zenith_angle = 30  # 太阳天顶角（问题2非垂直入射）
solar_angle_factor = np.cos(np.radians(solar_zenith_angle))  # 入射角修正因子
TARGET_SINGLE_THICKNESS = 50  # 单层PDMS基准厚度

am15_data = np.array([
    [0.40, 450.0], [0.45, 700.0], [0.50, 1000.0], [0.55, 1030.0], [0.60, 1050.0],
    [0.65, 1040.0], [0.70, 1000.0], [0.75, 970.0], [0.80, 950.0], [0.85, 920.0],
    [0.90, 900.0], [0.95, 870.0], [1.00, 850.0], [1.10, 800.0], [1.20, 750.0],
    [1.30, 700.0], [1.40, 650.0], [1.50, 600.0], [1.75, 500.0], [2.00, 400.0],
    [2.25, 300.0], [2.50, 200.0]
])

# 遗传算法参数
GA_PARAMS = {
    "pop_size": 60,  # 种群规模
    "max_gens": 200,  # 最大迭代次数
    "cx_prob": 0.7,  # 交叉概率
    "mut_prob": 0.2,  # 变异概率
    "elite_ratio": 0.15  # 精英保留比例
}

THICKNESS_CONSTRAINTS = {
    "Ag": (30e-9, 100e-9, 1e9),  # Ag基底: 30-100nm
    "Al": (30e-9, 100e-9, 1e9),  # 新增Al基底: 30-100nm
    "SIO2": (50e-9, 500e-9, 1e9),  # SiO2介质层: 50-500nm
    "Al2O3": (50e-9, 500e-9, 1e9),  # Al2O3介质层: 50-500nm
    "PDMS": (10e-6, 200e-6, 1e6)  # PDMS功能层: 10-200μm
}

def load_real_csv_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[
        str, np.ndarray]]:

    n_csv_path = os.path.join("Data", "Ag-PDMS_n.csv")
    k_csv_path = os.path.join("Data", "Ag-PDMS_k.csv")
    sio2_csv_path = os.path.join("Data", "SiO2_nk.csv")
    al2o3_csv_path = os.path.join("Data", "Al2O3_nk.csv")
    al_csv_path = os.path.join("Data", "Al_nk.csv")

    required_files = [n_csv_path, k_csv_path, sio2_csv_path, al2o3_csv_path, al_csv_path]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"未找到数据文件：{file}\n请确认Data文件夹路径正确！")

    df_n = pd.read_csv(n_csv_path)
    df_k = pd.read_csv(k_csv_path)
    required_n_cols = ['wl', 'Ag_n', 'PDMS_n']
    required_k_cols = ['wl', 'Ag_k', 'PDMS_k']
    if not all(col in df_n.columns for col in required_n_cols):
        raise ValueError(f"n文件列名错误！需包含：{required_n_cols}，实际：{df_n.columns.tolist()}")
    if not all(col in df_k.columns for col in required_k_cols):
        raise ValueError(f"k文件列名错误！需包含：{required_k_cols}，实际：{df_k.columns.tolist()}")

    df_sio2 = pd.read_csv(sio2_csv_path)
    df_al2o3 = pd.read_csv(al2o3_csv_path)
    required_sio2_cols = ['wl', 'SIO2_n', 'SIO2_k']
    required_al2o3_cols = ['wl', 'Al2O3_n', 'Al2O3_k']
    if not all(col in df_sio2.columns for col in required_sio2_cols):
        raise ValueError(f"SiO2文件列名错误！需包含：{required_sio2_cols}，实际：{df_sio2.columns.tolist()}")
    if not all(col in df_al2o3.columns for col in required_al2o3_cols):
        raise ValueError(f"Al2O3文件列名错误！需包含：{required_al2o3_cols}，实际：{df_al2o3.columns.tolist()}")

    df_al = pd.read_csv(al_csv_path)
    required_al_cols = ['wl', 'Al_n', 'Al_k']
    if not all(col in df_al.columns for col in required_al_cols):
        raise ValueError(f"Al文件列名错误！需包含：{required_al_cols}，实际：{df_al.columns.tolist()}")

    wl = df_n['wl'].values
    Ag_n = df_n['Ag_n'].values
    PDMS_n = df_n['PDMS_n'].values
    Ag_k = df_k['Ag_k'].values
    PDMS_k = df_k['PDMS_k'].values
    SiO2_n = df_sio2['SIO2_n'].values
    SiO2_k = df_sio2['SIO2_k'].values
    Al2O3_n = df_al2o3['Al2O3_n'].values
    Al2O3_k = df_al2o3['Al2O3_k'].values
    Al_n = df_al['Al_n'].values
    Al_k = df_al['Al_k'].values

    if not np.allclose(wl, df_k['wl'].values, rtol=1e-6) or \
            not np.allclose(wl, df_sio2['wl'].values, rtol=1e-6) or \
            not np.allclose(wl, df_al2o3['wl'].values, rtol=1e-6) or \
            not np.allclose(wl, df_al['wl'].values, rtol=1e-6):
        raise ValueError("所有n/k文件波长不匹配！请检查数据一致性。")

    # 整理字典（新增Al的n、k到字典）
    n_dict = {"Ag": Ag_n, "Al": Al_n, "PDMS": PDMS_n, "SIO2": SiO2_n, "Al2O3": Al2O3_n}
    k_dict = {"Ag": Ag_k, "Al": Al_k, "PDMS": PDMS_k, "SIO2": SiO2_k, "Al2O3": Al2O3_k}

    # 打印数据信息（新增Al数据提示）
    print("=" * 60)
    print("✅ 数据加载成功（含Al基底+多层膜介质层数据）")
    print(f"波长范围：{wl.min():.3f}~{wl.max():.3f}μm | 数据点数：{len(wl)}")
    print(f"PDMS太阳波段（0.4-2.5μm）平均n：{PDMS_n[(wl >= 0.4) & (wl <= 2.5)].mean():.4f}")
    print(f"Ag大气窗口（8-13μm）平均k：{Ag_k[(wl >= 8) & (wl <= 13)].mean():.4f}")
    print(f"Al大气窗口（8-13μm）平均k：{Al_k[(wl >= 8) & (wl <= 13)].mean():.4f}")
    print("=" * 60)

    return wl, Ag_n, Ag_k, Al_n, Al_k, PDMS_n, PDMS_k, SiO2_n, SiO2_k, Al2O3_n, Al2O3_k, n_dict, k_dict


# 加载数据（适配新增的Al返回值，其余逻辑不变）
try:
    wl, Ag_n, Ag_k, Al_n, Al_k, PDMS_n, PDMS_k, SiO2_n, SiO2_k, Al2O3_n, Al2O3_k, n_dict, k_dict = load_real_csv_data()
except Exception as e:
    print(f"❌ 数据加载失败：{str(e)}")
    exit()


def planck_spectrum(lam_m: np.ndarray, T: float) -> np.ndarray:
    """Planck黑体光谱（问题2同款）[W/(m²·sr·m)]"""
    h_planck = 6.626e-34
    c = 3e8
    k_boltzmann = 1.38e-23
    exponent = h_planck * c / (lam_m * k_boltzmann * T)
    exponent = np.clip(exponent, 0, 100)
    numerator = 2 * np.pi * h_planck * c ** 2
    denominator = lam_m ** 5 * (np.exp(exponent) - 1 + 1e-10)
    return numerator / denominator


def get_convection_coeff(T_mat: float, T_amb: float) -> float:

    delta_T = T_amb - T_mat
    h = h_base * (1 + 0.02 * delta_T) * h_wind_factor
    return np.clip(h, h_base, 30)


def calculate_Pnet(T_mat: float, wl: np.ndarray, eps: np.ndarray) -> Tuple[float, float, float, float, float, float]:


    window_mask = (wl >= 8.0) & (wl <= 13.0)
    wl_window = wl[window_mask]
    eps_window = eps[window_mask]
    lam_m_window = wl_window * 1e-6  # μm→m


    b_mat = planck_spectrum(lam_m_window, T_mat) * 1e-6  # 转换为W/(m²·μm)
    P_rad_out = np.trapezoid(eps_window * b_mat * tau_atm_window, wl_window) * np.pi


    b_sky = planck_spectrum(lam_m_window, T_sky) * 1e-6
    b_amb = planck_spectrum(lam_m_window, T_amb) * 1e-6
    P_rad_in = np.trapezoid(
        eps_window * (b_amb * (1 - tau_atm_window) + b_sky * tau_atm_window),
        wl_window
    ) * np.pi
    P_rad_net = P_rad_out - P_rad_in  # 辐射净散热（正值合理）


    solar_mask = (wl >= 0.4) & (wl <= 2.5)
    wl_solar = wl[solar_mask]
    eps_solar = eps[solar_mask]
    am15_irrad = np.interp(wl_solar, am15_data[:, 0], am15_data[:, 1])
    P_solar = solar_angle_factor * np.trapezoid(eps_solar * am15_irrad, wl_solar)


    h = get_convection_coeff(T_mat, T_amb)
    P_convec = h * (T_mat - T_amb)

    P_cond = k_cond * (T_mat - T_amb)

    P_net = P_rad_net - P_solar - P_convec - P_cond
    return P_net, P_rad_net, P_solar, P_convec, P_cond, h


def solve_Tmat(wl: np.ndarray, eps: np.ndarray, tol: float = 1e-4, max_iter: int = 200) -> Tuple[float, float, float]:

    T_mat = T_amb - 5
    for iter_idx in range(max_iter):
        P_net, _, _, _, _, h = calculate_Pnet(T_mat, wl, eps)


        dt = 1e-3
        P_net_plus = calculate_Pnet(T_mat + dt, wl, eps)[0]
        P_net_minus = calculate_Pnet(T_mat - dt, wl, eps)[0]
        dP_dT = (P_net_plus - P_net_minus) / (2 * dt + 1e-10)

        if abs(dP_dT) < 1e-5:
            dP_dT = 1e-5 if dP_dT >= 0 else -1e-5

        T_mat_new = T_mat - P_net / dP_dT
        T_mat_new = np.clip(T_mat_new, 263.15, T_amb)  # 263.15K = -10℃

        if abs(T_mat_new - T_mat) < tol:
            delta_T = T_amb - T_mat_new
            return T_mat_new, delta_T, h

        T_mat = T_mat_new

    print(f"⚠ 稳态温度迭代{max_iter}次未完全收敛（误差：{abs(T_mat_new - T_mat):.6f}K）")
    delta_T = T_amb - T_mat_new
    return T_mat_new, delta_T, h


def get_single_pdms_perf(d_pdms: float) -> Dict:

    n_air = 1.0
    n_pdms_complex = PDMS_n + 1j * PDMS_k
    n_ag_complex = Ag_n + 1j * Ag_k

    r1 = (n_air - n_pdms_complex) / (n_air + n_pdms_complex + 1e-10)
    R1 = np.abs(r1) ** 2
    r2 = (n_pdms_complex - n_ag_complex) / (n_pdms_complex + n_ag_complex + 1e-10)
    R2 = np.abs(r2) ** 2

    lambda_m = wl * 1e-6
    alpha = 4 * np.pi * PDMS_k / lambda_m
    exp_attn = np.exp(-2 * alpha * d_pdms)
    exp_attn = np.clip(exp_attn, 0, 1)

    denominator = 1 - R1 * R2 * exp_attn
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    R = R1 + (1 - R1) ** 2 * R2 * exp_attn / denominator
    eps = 1 - R
    eps = np.clip(eps, 0.01, 0.99)

    P_net_tamb, P_rad_net, P_solar, _, _, _ = calculate_Pnet(T_amb, wl, eps)

    T_mat_steady, delta_T, h_steady = solve_Tmat(wl, eps)

    P_net_steady, _, _, _, _, _ = calculate_Pnet(T_mat_steady, wl, eps)

    solar_mask = (wl >= 0.4) & (wl <= 2.5)
    window_mask = (wl >= 8) & (wl <= 13)
    avg_eps_solar = eps[solar_mask].mean()
    avg_eps_window = eps[window_mask].mean()

    return {
        "thickness_μm": d_pdms * 1e6,
        "eps": eps,
        "T_mat_c_steady": T_mat_steady - 273.15,
        "delta_T": delta_T,
        "P_net_tamb": P_net_tamb,
        "P_net_steady": P_net_steady,
        "P_rad_net": P_rad_net,
        "P_solar": P_solar,
        "avg_eps_solar": avg_eps_solar,
        "avg_eps_window": avg_eps_window
    }


single_pdms_perf = get_single_pdms_perf(TARGET_SINGLE_THICKNESS * 1e-6)
print(f"\n📊 问题2基准性能（单层PDMS {TARGET_SINGLE_THICKNESS}μm + Ag基底）：")
print(f"   - 稳态降温幅度：{single_pdms_perf['delta_T']:.2f}℃")
print(f"   - 非稳态P_net（T_amb下）：{single_pdms_perf['P_net_tamb']:.2f} W/m²（核心对比指标）")
print(f"   - 稳态P_net：{single_pdms_perf['P_net_steady']:.2f} W/m²（必然为0，仅参考）")
print(f"   - 太阳吸收率：{single_pdms_perf['avg_eps_solar']:.4f}")
print(f"   - 大气窗口发射率：{single_pdms_perf['avg_eps_window']:.3f}")


def calculate_multilayer_emissivity(
        materials: List[str],
        thicknesses: List[float]
) -> np.ndarray:

    n_air = 1.0
    eps = np.zeros_like(wl, dtype=np.float64)
    substrate = materials[-1]

    for idx, lam in enumerate(wl):

        total_matrix = np.eye(2, dtype=np.complex128)

        for mat, d in zip(materials, thicknesses):

            n = n_dict[mat][idx]
            k = k_dict[mat][idx]
            n_complex = n + 1j * k


            lam_m = lam * 1e-6
            delta = 2 * np.pi * n_complex * d / lam_m


            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            layer_matrix = np.array([
                [cos_d, -1j * sin_d / n_complex],
                [-1j * n_complex * sin_d, cos_d]
            ], dtype=np.complex128)


            total_matrix = total_matrix @ layer_matrix


        n_substrate = n_dict[substrate][idx] + 1j * k_dict[substrate][idx]

        M11, M12, M21, M22 = total_matrix[0, 0], total_matrix[0, 1], total_matrix[1, 0], total_matrix[1, 1]
        numerator = n_air * M11 + n_air * n_substrate * M12 - M21 - n_substrate * M22
        denominator = n_air * M11 + n_air * n_substrate * M12 + M21 + n_substrate * M22
        if abs(denominator) < 1e-10:
            r = 1.0
        else:
            r = numerator / denominator

        R = np.abs(r) ** 2
        eps[idx] = np.clip(1 - R, 0.01, 0.99)

    return eps

def init_individual() -> Dict[str, List]:
    dielectric_mat = np.random.choice(["SIO2", "Al2O3"])  # 介质层随机
    substrate_mat = np.random.choice(["Ag", "Al"])  # 新增：基底随机选择Ag/Al
    materials = ["PDMS", dielectric_mat, substrate_mat]  # 层序：PDMS→介质→基底
    thicknesses = []
    for mat in materials:
        min_d, max_d, _ = THICKNESS_CONSTRAINTS[mat]
        d = np.random.uniform(min_d, max_d)  # 均匀分布采样厚度
        thicknesses.append(d)
    return {"materials": materials, "thicknesses": thicknesses}


def calculate_fitness(individual: Dict[str, List]) -> Tuple[float, float, float, float, float]:
    eps = calculate_multilayer_emissivity(
        individual["materials"], individual["thicknesses"]
    )

    P_net_tamb, P_rad_net, P_solar, _, _, _ = calculate_Pnet(T_amb, wl, eps)

    _, delta_T, _ = solve_Tmat(wl, eps)

    solar_mask = (wl >= 0.4) & (wl <= 2.5)
    window_mask = (wl >= 8) & (wl <= 13)
    avg_eps_solar = eps[solar_mask].mean()
    avg_eps_window = eps[window_mask].mean()

    solar_penalty = max(0, avg_eps_solar - 0.10) * 50
    window_penalty = max(0, 0.80 - avg_eps_window) * 50
    total_penalty = solar_penalty + window_penalty

    fitness = P_net_tamb - total_penalty
    fitness = max(fitness, 1e-10)

    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back and 'gen' in frame.f_back.f_locals:
        gen = frame.f_back.f_locals['gen']
        fitness_list = frame.f_back.f_locals.get('fitness_list', [])
        substrate = individual["materials"][-1]  # 显示当前基底
        if gen == 0 and len(fitness_list) < 3:
            print(
                f"  [调试] 基底:{substrate} | P_net（T_amb下）={P_net_tamb:.2f}, α_solar={avg_eps_solar:.4f}, ε_window={avg_eps_window:.4f}, "
                f"惩罚={total_penalty:.2f}, 适应度={fitness:.2f}, 稳态ΔT={delta_T:.2f}℃")

    return fitness, delta_T, P_net_tamb, avg_eps_solar, avg_eps_window


def selection(population: List[Dict], fitness_list: List[float], elite_size: int) -> List[Dict]:
    fitness_arr = np.array(fitness_list, dtype=np.float64)
    fitness_arr = np.nan_to_num(fitness_arr, nan=1e-10)
    fitness_arr[fitness_arr <= 0] = 1e-10

    sorted_idx = np.argsort(fitness_arr)[::-1]
    elites = [deepcopy(population[i]) for i in sorted_idx[:elite_size]]

    probs = fitness_arr / fitness_arr.sum()
    selected_indices = np.random.choice(
        len(population),
        size=len(population) - elite_size,
        p=probs,
        replace=True
    )
    selected = [deepcopy(population[i]) for i in selected_indices]

    return elites + selected


def crossover(parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
    child1, child2 = deepcopy(parent1), deepcopy(parent2)

    if np.random.random() < GA_PARAMS["cx_prob"]:
        if np.random.random() < 0.5:
            child1["materials"][1], child2["materials"][1] = child2["materials"][1], child1["materials"][1]

        if np.random.random() < 0.5:
            child1["materials"][2], child2["materials"][2] = child2["materials"][2], child1["materials"][2]

        cx_point = np.random.randint(1, len(parent1["thicknesses"]))
        child1["thicknesses"][cx_point:], child2["thicknesses"][cx_point:] = \
            child2["thicknesses"][cx_point:], child1["thicknesses"][cx_point:]

    return child1, child2


def mutate(individual: Dict) -> Dict:
    mutated = deepcopy(individual)

    if np.random.random() < GA_PARAMS["mut_prob"]:
        current_mat = mutated["materials"][1]
        mutated["materials"][1] = "Al2O3" if current_mat == "SIO2" else "SIO2"

    if np.random.random() < GA_PARAMS["mut_prob"]:
        current_substrate = mutated["materials"][2]
        mutated["materials"][2] = "Al" if current_substrate == "Ag" else "Ag"

    for i, (mat, d) in enumerate(zip(mutated["materials"], mutated["thicknesses"])):
        if np.random.random() < GA_PARAMS["mut_prob"]:
            min_d, max_d, _ = THICKNESS_CONSTRAINTS[mat]
            std = 0.15 * (max_d - min_d)
            new_d = np.random.normal(loc=d, scale=std)
            mutated["thicknesses"][i] = np.clip(new_d, min_d, max_d)

    return mutated


def genetic_algorithm_optimization() -> Tuple[Dict, List[float], List[float], List[float]]:

    pop_size = GA_PARAMS["pop_size"]
    max_gens = GA_PARAMS["max_gens"]
    elite_size = int(pop_size * GA_PARAMS["elite_ratio"])

    population = [init_individual() for _ in range(pop_size)]
    best_fitness_hist = []
    best_delta_T_hist = []
    best_pnet_tamb_hist = []
    best_fitness = 0.0
    no_improvement_count = 0

    print(f"\n🚀 开始GA优化（种群:{pop_size}, 最大代数:{max_gens}，基底选项：Ag/Al）")
    for gen in range(max_gens):
        fitness_list = []
        delta_T_list = []
        pnet_tamb_list = []
        for ind in population:
            fitness, delta_T, p_net_tamb, _, _ = calculate_fitness(ind)
            fitness_list.append(fitness)
            delta_T_list.append(delta_T)
            pnet_tamb_list.append(p_net_tamb)

        # 找到当前代最优个体
        best_idx = np.argmax(fitness_list)
        current_best_fitness = fitness_list[best_idx]
        current_best_delta_T = delta_T_list[best_idx]
        current_best_pnet = pnet_tamb_list[best_idx]
        current_best_ind = population[best_idx]
        current_best_substrate = current_best_ind["materials"][-1]

        best_fitness_hist.append(current_best_fitness)
        best_delta_T_hist.append(current_best_delta_T)
        best_pnet_tamb_hist.append(current_best_pnet)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= 40:
            print(f"\n✓ GA早停于第{gen + 1}代（连续40代无适应度改进）")
            break

        if (gen + 1) % 20 == 0 or gen == 0:
            mats = current_best_ind["materials"]
            thicks = [d * THICKNESS_CONSTRAINTS[m][2] for m, d in zip(mats, current_best_ind["thicknesses"])]
            units = ["μm" if THICKNESS_CONSTRAINTS[m][2] == 1e6 else "nm" for m in mats]
            print(
                f"第{gen + 1:3d}代 | 基底:{current_best_substrate} | P_net（T_amb下）={current_best_pnet:6.2f}W/m² | 适应度={current_best_fitness:6.2f} | ΔT={current_best_delta_T:4.2f}℃ | "
                f"{mats[0]}={thicks[0]:.1f}{units[0]}, {mats[1]}={thicks[1]:.0f}{units[1]}, {mats[2]}={thicks[2]:.0f}{units[2]}")

        selected_pop = selection(population, fitness_list, elite_size)
        offspring = []
        for i in range(0, len(selected_pop), 2):
            if i + 1 < len(selected_pop):
                c1, c2 = crossover(selected_pop[i], selected_pop[i + 1])
                offspring.extend([c1, c2])
            else:
                offspring.append(selected_pop[i])
        population = [mutate(ind) for ind in offspring[:pop_size]]

    final_fitness = []
    final_delta_T = []
    final_pnet = []
    for ind in population:
        fit, dt, pnet, _, _ = calculate_fitness(ind)
        final_fitness.append(fit)
        final_delta_T.append(dt)
        final_pnet.append(pnet)
    best_idx = np.argmax(final_fitness)
    best_ind = population[best_idx]
    best_ind_substrate = best_ind["materials"][-1]
    best_ind_perf = {
        "delta_T": final_delta_T[best_idx],
        "P_net_tamb": final_pnet[best_idx],
        "fitness": final_fitness[best_idx],
        "substrate": best_ind_substrate
    }
    print(f"\n✅ GA优化完成！最终最优个体性能：")
    print(f"   - 多层结构：{' → '.join(best_ind['materials'])}")
    print(f"   - 稳态降温幅度：{best_ind_perf['delta_T']:.2f}℃（vs 问题2基准（Ag基底）{single_pdms_perf['delta_T']:.2f}℃）")
    print(
        f"   - 非稳态P_net（T_amb下）：{best_ind_perf['P_net_tamb']:.2f} W/m²（vs 问题2基准（Ag基底）{single_pdms_perf['P_net_tamb']:.2f} W/m²）")

    return best_ind, best_fitness_hist, best_delta_T_hist, best_pnet_tamb_hist

def analyze_best_structure(best_ind: Dict) -> Dict:

    multi_eps = calculate_multilayer_emissivity(best_ind["materials"], best_ind["thicknesses"])
    multi_P_net_tamb, multi_P_rad_net, multi_P_solar, _, _, _ = calculate_Pnet(T_amb, wl, multi_eps)
    _, multi_delta_T, _ = solve_Tmat(wl, multi_eps)

    solar_mask = (wl >= 0.4) & (wl <= 2.5)
    window_mask = (wl >= 8) & (wl <= 13)
    multi_avg_eps_solar = multi_eps[solar_mask].mean()
    multi_avg_eps_window = multi_eps[window_mask].mean()

    single_delta_T = single_pdms_perf["delta_T"]
    single_P_net_tamb = single_pdms_perf["P_net_tamb"]
    single_avg_eps_solar = single_pdms_perf["avg_eps_solar"]
    single_avg_eps_window = single_pdms_perf["avg_eps_window"]
    single_P_rad_net = single_pdms_perf["P_rad_net"]
    single_P_solar = single_pdms_perf["P_solar"]

    delta_T_improve = (multi_delta_T - single_delta_T) / single_delta_T * 100
    P_net_improve = (multi_P_net_tamb - single_P_net_tamb) / abs(single_P_net_tamb) * 100
    alpha_reduce = (single_avg_eps_solar - multi_avg_eps_solar) / single_avg_eps_solar * 100
    window_improve = (multi_avg_eps_window - single_avg_eps_window) / single_avg_eps_window * 100

    print("\n" + "=" * 90)
    print(f"📊 最优多层膜 vs 问题2基准（单层PDMS {TARGET_SINGLE_THICKNESS}μm + Ag基底）")
    print("=" * 90)
    # 格式化多层膜结构信息（含基底）
    mats = best_ind["materials"]
    thicks = [d * THICKNESS_CONSTRAINTS[m][2] for m, d in zip(mats, best_ind["thicknesses"])]
    units = ["μm" if THICKNESS_CONSTRAINTS[m][2] == 1e6 else "nm" for m in mats]
    struct_info = f"多层结构：{' → '.join(mats)} | 厚度：{mats[0]}={thicks[0]:.1f}{units[0]}, {mats[1]}={thicks[1]:.0f}{units[1]}, {mats[2]}={thicks[2]:.0f}{units[2]}"
    print(struct_info)
    # 对比数据（核心：非稳态P_net）
    print(f"{'指标':<30} {'多层膜（' + mats[-1] + '基底）':<22} {'单层PDMS（Ag基底）':<22} {'变化幅度':<12}")
    print(f"{'稳态降温幅度 (℃)':<30} {multi_delta_T:<22.2f} {single_delta_T:<22.2f} ↑{delta_T_improve:<10.1f}%")
    print(
        f"{'非稳态P_net（T_amb下） (W/m²)':<30} {multi_P_net_tamb:<22.2f} {single_P_net_tamb:<22.2f} ↑{P_net_improve:<10.1f}%")
    print(f"{'太阳吸收率':<30} {multi_avg_eps_solar:<22.4f} {single_avg_eps_solar:<22.4f} ↓{alpha_reduce:<10.1f}%")
    print(
        f"{'大气窗口发射率':<30} {multi_avg_eps_window:<22.4f} {single_avg_eps_window:<22.4f} ↑{window_improve:<10.1f}%")
    print(f"{'辐射净散热（T_amb下） (W/m²)':<30} {multi_P_rad_net:<22.2f} {single_P_rad_net:<22.2f} -")
    print(f"{'太阳吸收功率 (W/m²)':<30} {multi_P_solar:<22.2f} {single_P_solar:<22.2f} -")
    print("=" * 90)

    return {
        "multi_materials": mats,
        "multi_thicknesses": thicks,
        "multi_units": units,
        "multi_delta_T": multi_delta_T,
        "multi_P_net_tamb": multi_P_net_tamb,
        "multi_avg_eps_solar": multi_avg_eps_solar,
        "multi_avg_eps_window": multi_avg_eps_window,
        "single_delta_T": single_delta_T,
        "single_P_net_tamb": single_pdms_perf["P_net_tamb"],
        "single_avg_eps_solar": single_pdms_perf["avg_eps_solar"],
        "single_avg_eps_window": single_pdms_perf["avg_eps_window"],
        "delta_T_improve": delta_T_improve,
        "P_net_improve": P_net_improve,
        "multi_eps": multi_eps,
        "single_eps": single_pdms_perf["eps"]
    }


def plot_results(best_result: Dict, best_fitness_hist: List, best_delta_T_hist: List, best_pnet_hist: List):
    save_dir = os.path.join("Data", "MultiLayer_Results_Fixed_Pnet")
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (14, 12)
    plt.rcParams['lines.linewidth'] = 2.8
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    save_dpi = 300

    fig1, ax1 = plt.subplots()
    gens = range(1, len(best_fitness_hist) + 1)
    line1 = ax1.plot(gens, best_pnet_hist, 'o-', color='#165DFF', markerfacecolor='#6B9DFF',
                     markeredgecolor='white', markeredgewidth=2, markersize=8, alpha=0.9,
                     label='Best Unsteady-state P_net per Gen (at T_amb)')
    ax1.set_xlabel('GA Iteration', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Unsteady-state P_net (W/m²)', color='#165DFF', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#165DFF')
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)

    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(gens, best_delta_T_hist, 's-', color='#FF7D00', markerfacecolor='#FFB380',
                          markeredgecolor='white', markeredgewidth=2, markersize=8, alpha=0.9, label='Best ΔT per Gen')
    line3 = ax1_twin.plot(gens, best_fitness_hist, '^-', color='#00B42A', markerfacecolor='#66E080',
                          markeredgecolor='white', markeredgewidth=2, markersize=8, alpha=0.9,
                          label='Best Fitness per Gen')
    ax1_twin.set_ylabel('Temperature Drop (℃) / Fitness (W/m²)', fontweight='bold', fontsize=12)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.95)
    ax1.set_title('GA Optimization Convergence Curve (Based on Unsteady-state P_net)', fontsize=14, fontweight='bold',
                  pad=20)

    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, "GA_Convergence_Curve_Fixed.png"),
                 dpi=save_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig1)
    print(f"\nFigure 1 saved to: {os.path.join(save_dir, 'GA_Convergence_Curve_Fixed.png')}")

    fig2, ax2 = plt.subplots()
    # Multi-layer emissivity (blue) —— 自动显示基底名称
    ax2.plot(wl, best_result["multi_eps"], color='#165DFF', linewidth=3, alpha=0.9,
             label=f'Multi-layer ({best_result["multi_materials"][0]}→{best_result["multi_materials"][1]}→{best_result["multi_materials"][2]})')
    # Single-layer PDMS emissivity (orange dashed line, Problem 2 baseline)
    ax2.plot(wl, best_result["single_eps"], color='#FF7D00', linewidth=2.5, linestyle='--', alpha=0.8,
             label=f'Single-layer PDMS ({TARGET_SINGLE_THICKNESS}μm, Ag Substrate)')
    # Highlight key wavelength bands
    ax2.axvspan(0.4, 2.5, color='#FF7D00', alpha=0.1, label='Solar Band (0.4-2.5μm)')
    ax2.axvspan(8, 13, color='#165DFF', alpha=0.1, label='Atmospheric Window (8-13μm)')

    # Axis settings
    ax2.set_xlabel('Wavelength (μm)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Emissivity', fontweight='bold', fontsize=12)
    ax2.set_xlim(wl.min(), wl.max())
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax2.set_title('Emissivity Spectrum: Multi-layer vs Single-layer PDMS', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, "Emissivity_Spectrum_Comparison.png"),
                 dpi=save_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig2)
    print(f"Figure 2 saved to: {os.path.join(save_dir, 'Emissivity_Spectrum_Comparison.png')}")

    fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(14, 6))
    categories = [f'Multi-layer ({best_result["multi_materials"][-1]} Substrate)', 'Single-layer PDMS (Ag Substrate)']
    delta_T_data = [best_result["multi_delta_T"], best_result["single_delta_T"]]
    P_net_data = [best_result["multi_P_net_tamb"], best_result["single_P_net_tamb"]]

    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax3_1.bar(x - width / 2, delta_T_data, width, label='Steady-state Temp Drop (℃)', color='#165DFF',
                      alpha=0.8)
    bars2 = ax3_1.bar(x + width / 2, P_net_data, width, label='Unsteady-state P_net (W/m²)', color='#00B42A', alpha=0.8)

    for bar, val in zip(bars1, delta_T_data):
        ax3_1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{val:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars2, P_net_data):
        ax3_1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax3_1.set_xlabel('Structure Type', fontweight='bold', fontsize=12)
    ax3_1.set_ylabel('Value', fontweight='bold', fontsize=12)
    ax3_1.set_title('Temp Drop & Unsteady-state P_net Comparison', fontsize=13, fontweight='bold')
    ax3_1.set_xticks(x)
    ax3_1.set_xticklabels(categories, rotation=15)
    ax3_1.legend(fontsize=10)
    ax3_1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y')

    alpha_data = [best_result["multi_avg_eps_solar"], best_result["single_avg_eps_solar"]]
    window_data = [best_result["multi_avg_eps_window"], best_result["single_avg_eps_window"]]

    bars3 = ax3_2.bar(x - width / 2, alpha_data, width, label='Solar Absorptivity', color='#FF7D00', alpha=0.8)
    bars4 = ax3_2.bar(x + width / 2, window_data, width, label='Atmospheric Window Emissivity', color='#722ED1',
                      alpha=0.8)

    # Label values on bars
    for bar, val in zip(bars3, alpha_data):
        ax3_2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f'{val:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
    for bar, val in zip(bars4, window_data):
        ax3_2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.4f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax3_2.set_xlabel('Structure Type', fontweight='bold', fontsize=12)
    ax3_2.set_ylabel('Value', fontweight='bold', fontsize=12)
    ax3_2.set_title('Solar Absorptivity & Window Emissivity Comparison', fontsize=13, fontweight='bold')
    ax3_2.set_xticks(x)
    ax3_2.set_xticklabels(categories, rotation=15)  # 旋转标签避免重叠
    ax3_2.legend(fontsize=10)
    ax3_2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, axis='y')

    plt.tight_layout()
    fig3.savefig(os.path.join(save_dir, "Performance_Comparison_Bar_Fixed.png"),
                 dpi=save_dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig3)
    print(f"Figure 3 saved to: {os.path.join(save_dir, 'Performance_Comparison_Bar_Fixed.png')}")

    result_df = pd.DataFrame({
        "Optimization Indicator": [
            "Multi-layer Structure Materials",
            f"{best_result['multi_materials'][0]} Thickness",
            f"{best_result['multi_materials'][1]} Thickness",
            f"{best_result['multi_materials'][2]} Substrate Thickness",  # 新增：基底厚度标签
            "Multi-layer Steady Temp Drop (℃)",
            "Single-layer PDMS (Ag Substrate) Steady Temp Drop (℃)",
            "Temp Drop Improvement",
            "Multi-layer Unsteady P_net (W/m²)",
            "Single-layer PDMS (Ag Substrate) Unsteady P_net (W/m²)",
            "Unsteady P_net Improvement",
            "Multi-layer Solar Absorptivity",
            "Single-layer PDMS (Ag Substrate) Solar Absorptivity",
            "Multi-layer Window Emissivity",
            "Single-layer PDMS (Ag Substrate) Window Emissivity",
            "Total GA Iterations",
            "GA Early Stopping"
        ],
        "Value": [
            ' → '.join(best_result["multi_materials"]),
            f'{best_result["multi_thicknesses"][0]:.1f}{best_result["multi_units"][0]}',
            f'{best_result["multi_thicknesses"][1]:.0f}{best_result["multi_units"][1]}',
            f'{best_result["multi_thicknesses"][2]:.0f}{best_result["multi_units"][2]}',
            f'{best_result["multi_delta_T"]:.2f}',
            f'{best_result["single_delta_T"]:.2f}',
            f'{best_result["delta_T_improve"]:.1f}%',
            f'{best_result["multi_P_net_tamb"]:.2f}',
            f'{best_result["single_P_net_tamb"]:.2f}',
            f'{best_result["P_net_improve"]:.1f}%',
            f'{best_result["multi_avg_eps_solar"]:.4f}',
            f'{best_result["single_avg_eps_solar"]:.4f}',
            f'{best_result["multi_avg_eps_window"]:.4f}',
            f'{best_result["single_avg_eps_window"]:.4f}',
            f'{len(best_fitness_hist)}',
            "Yes" if len(best_fitness_hist) < GA_PARAMS["max_gens"] else "No"
        ]
    })

    # Save to Excel/CSV
    try:
        result_df.to_excel(os.path.join(save_dir, "MultiLayer_VS_Problem2_Results_Fixed.xlsx"),
                           index=False, engine='openpyxl')
        print(f"Results saved to Excel: {os.path.join(save_dir, 'MultiLayer_VS_Problem2_Results_Fixed.xlsx')}")
    except:
        result_df.to_csv(os.path.join(save_dir, "MultiLayer_VS_Problem2_Results_Fixed.csv"),
                         index=False, encoding='utf-8-sig')
        print(f"Results saved to CSV: {os.path.join(save_dir, 'MultiLayer_VS_Problem2_Results_Fixed.csv')}")

    print(f"\nAll result files saved to: {os.path.abspath(save_dir)}")


def main():
    try:
        print("=" * 90)
        print("问题3：多层膜辐射冷却结构优化（P_net=0问题已修复，基底选项：Ag/Al）")
        print("=" * 90)

        best_ind, best_fitness_hist, best_delta_T_hist, best_pnet_hist = genetic_algorithm_optimization()


        print("\n📈 正在分析最优多层膜结构性能...")
        best_result = analyze_best_structure(best_ind)


        print("\n🎨 正在绘制结果图表...")
        plot_results(best_result, best_fitness_hist, best_delta_T_hist, best_pnet_hist)

        print("\n" + "=" * 90)
        print("✓ 问题3多层膜优化任务完成（P_net=0问题已修复！）")
        print("=" * 90)
        print(
            f"核心结论：多层膜结构（{'→'.join(best_ind['materials'])}）较问题2基准（单层PDMS {TARGET_SINGLE_THICKNESS}μm + Ag基底）")
        print(
            f"   - 稳态降温幅度提升 {best_result['delta_T_improve']:.1f}%（{best_result['multi_delta_T']:.2f}℃ → {best_result['single_delta_T']:.2f}℃）")
        print(
            f"   - 非稳态P_net提升 {best_result['P_net_improve']:.1f}%（{best_result['multi_P_net_tamb']:.2f} → {best_result['single_P_net_tamb']:.2f} W/m²）")
        print(
            f"   - 太阳吸收率降低 {abs((best_result['multi_avg_eps_solar'] - best_result['single_avg_eps_solar']) * 100):.2f}%")
        print(
            f"   - 大气窗口发射率提升 {abs((best_result['multi_avg_eps_window'] - best_result['single_avg_eps_window']) * 100):.2f}%")
        print("=" * 90)

    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()