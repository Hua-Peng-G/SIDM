# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import time
import math
import warnings
from numpy.lib.format import open_memmap
# 这里已经去掉了 compute_current_P_fac 的导入
from kernel import run_simulation_chunk_numba

# --- 两行解决指定文件名 ---
CONFIG_FILENAME = 'config2.txt' 
# -------------------------

warnings.simplefilter('ignore')

def load_config(config_file):
    cfg = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if '=' not in line: continue
            key, val = line.split('=', 1)
            key, val = key.strip(), val.strip()
            if val.lower() == 'true': val = True
            elif val.lower() == 'false': val = False
            else:
                try: val = int(val)
                except ValueError:
                    try: val = float(val)
                    except ValueError: pass
            cfg[key] = val
    return cfg

def load_parameters(param_file='parameter.txt'):
    params = {}
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            key, val = line.split('=')
            params[key.strip()] = float(val.strip())
    return params

def main():
    print(f"Reading configuration: {CONFIG_FILENAME}")
    config = load_config(CONFIG_FILENAME)
    params = load_parameters('parameter.txt')

    # 读取物理参数
    Rs, M, sigma0 = params['Rs'], params['M'], params['sigma0']
    a, b, gm, ephib, val_C, w = params['a'], params['b'], params['gm'], params['ephib'], params['C'], params['w']
    dt_t0, total_steps = float(config['dt_t0']), int(config['total_steps'])
    save_interval, freeze_radius = int(config['save_interval']), float(config['freeze_radius'])

    FLAG_HSE, FLAG_HEAT = bool(config['FLAG_HSE']), bool(config['FLAG_HEAT'])
    ENABLE_PLOTTING, plot_x_key, plot_y_key = bool(config['ENABLE_PLOTTING']), config['plot_x'], config['plot_y']
    plot_log_y, filename = bool(config['plot_log_y']), config['filename']

    FLAG_BH_FORMED = bool(config.get('FLAG_BH_FORMED', False))
    EPS_EXC = float(config.get('EPS_EXC', 1.0))
    INITIAL_BH_IDX = int(config.get('INITIAL_BH_IDX', 0))
    FLAG_DAMP, DAMP_COEFF = bool(config.get('FLAG_DAMP', False)), float(config.get('DAMP_COEFF', 1e-7))
    FLAG_SM, v_limit_factor = bool(config.get('FLAG_SM', False)), float(config.get('v_limit_factor', 10000.0))

    # 单位转换
    term_R, term_M = Rs / 2.6, M / 6.3e9
    R0_scale = (term_R / term_M) * 8.5e6
    sigma = sigma0 * (2e33 * M) / ((1.48e5 * M)**2)
    dt = dt_t0 * 1.35e12 * (sigma0**-1) * (term_M**(-2.5)) * (term_R**3.5)
    fq = -1.5 * (gm - 1.0)**1.5 * a

    # 读取数据
    init_file = config.get('initial_filename', 'initial.npy')
    grid_file = config.get('grid_filename', 'A.npy')
    A_raw = np.load(grid_file).astype(np.float64).flatten()
    initial_raw = np.load(init_file).astype(np.float64)
    if initial_raw.ndim == 1: initial_raw = initial_raw.reshape(11, -1)

    A_arr = np.ascontiguousarray(A_raw[:initial_raw.shape[1]])
    start = initial_raw[:, :A_arr.size]
    Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt = [np.ascontiguousarray(start[i]) for i in range(11)]
    idx_bh_arr = np.array([INITIAL_BH_IDX], dtype=np.int64)
    N_grid = A_arr.size

    # 模拟准备
    SUB_STEPS, SPLIT_INDEX = 1, 1
    real_total_steps = total_steps // SUB_STEPS
    num_frames = math.ceil(real_total_steps / save_interval) + 1
    if os.path.exists(filename): os.remove(filename)
    
    # 因为去掉了 P_fac，输出维度恢复为 11 层
    fp = open_memmap(filename, mode='w+', dtype='float64', shape=(num_frames, 11, N_grid))

    state_out = np.zeros((11, N_grid))
    state_out[0:11, :] = [Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt]
    fp[0] = state_out
    fp.flush()

    print(f"Starting simulation...")
    start_time = time.time()
    current_step, frame_idx = 0, 1

    while current_step < real_total_steps:
        steps_to_run = min(save_interval, real_total_steps - current_step)
        run_simulation_chunk_numba(
            Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
            A_arr, dt, 1.0, R0_scale, sigma, fq, a, b, gm, ephib, w,
            steps_to_run, SPLIT_INDEX, SUB_STEPS,
            freeze_radius, FLAG_HSE, FLAG_HEAT,
            FLAG_BH_FORMED, idx_bh_arr, EPS_EXC,
            FLAG_DAMP, DAMP_COEFF, FLAG_SM, v_limit_factor
        )
        current_step += steps_to_run
        
        if frame_idx < num_frames:
            state_out[0:11, :] = [Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt]
            fp[frame_idx] = state_out
            frame_idx += 1
        
        # 终端监控
        curr_idx = idx_bh_arr[0]
        v_monitor = math.sqrt(max(0.0, 2.0 * ept[curr_idx] / (3.0 * R0_scale))) * 3.0e5
        M_bh = mt[curr_idx] if FLAG_BH_FORMED else 0.0
        print(f"Step {current_step/1000:.1f}k/{real_total_steps/1000:.1f}k | rho_in={rhot[curr_idx]:.4e} | v_in={v_monitor:.2f} km/s | BH_Idx={curr_idx} | M_BH={M_bh:.4e} ", end='\r', flush=True)

    fp.flush()
    print(f"\nSimulation complete. Total time: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()