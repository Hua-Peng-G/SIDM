import numpy as np
import os
import time
import math
import sys
from numpy.lib.format import open_memmap
import warnings

# Import calculation kernel
try:
    from kernel import run_simulation_chunk_numba, compute_current_P_fac
except ImportError:
    print("Error: 'kernel.py' not found. Please ensure it is in the same directory.")
    sys.exit(1)

warnings.resetwarnings()
warnings.simplefilter('ignore')

def load_config(config_file='config.txt'):
    cfg = {}
    with open(config_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            key, val = line.split('=')
            key = key.strip()
            val = val.strip()
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

# --- 1. Read external files ---
print("Reading configuration files...")
config = load_config('config.txt')
params = load_parameters('parameter.txt')

# Read new physical parameters
Rs = params['Rs']
M = params['M']
sigma0 = params['sigma0']

# Read other parameters
a = params['a']
b = params['b']
gm = params['gm']
ephib = params['ephib']
val_C = params['C']
Cq_viscosity = 1.0

# Read configuration
dt_t0 = float(config['dt_t0']) 
N_grid_config = int(config['N'])
total_steps = int(config['total_steps'])
save_interval = int(config['save_interval'])
FLAG_HSE = bool(config['FLAG_HSE'])
FLAG_P_FAC = bool(config['FLAG_P_FAC'])
FLAG_HEAT = bool(config['FLAG_HEAT'])
ENABLE_PLOTTING = bool(config['ENABLE_PLOTTING'])
plot_x_key = config['plot_x']
plot_y_key = config['plot_y']
plot_log_y = bool(config['plot_log_y'])
filename = config['filename']
freeze_radius = float(config['freeze_radius'])

# --- 2. Parameter conversion (Unit conversion layer) ---
term_R = Rs / 2.6
term_M = M / 6.3e9

# Calculate R0_scale
R0_scale = (term_R / term_M) * 8.5e6

# Calculate sigma
sigma = sigma0 * (2e33 * M) / ((1.48e5 * M)**2)

# Calculate dt (Physical time step used inside the program)
dt = dt_t0 * 1.35e12 * (sigma0**-1) * (term_M**(-2.5)) * (term_R**3.5)

# Calculate t_0 (Only for displaying information)
t_0_gyr = 1.33 * (sigma0**-1) * (term_M**(-1.5)) * (term_R**3.5)

# Calculate fq
fq = -1.5 * (gm - 1.0)**1.5 * a

print("-" * 40)
print(f"Physical Parameters:")
print(f"  Rs     = {Rs:.4e} kpc")
print(f"  M      = {M:.4e} solar mass")
print(f"  sigma = {sigma0:.4e} cm^2/g")
print("-" * 40)
print(f"Calculated Simulation Units:(natural units)")
print(f"  R0_scale = {R0_scale:.4e}")
print(f"  sigma    = {sigma:.4e}")
print(f"  dt (sim) = {dt:.4e}")
print("-" * 40)
print(f"Time Scale Info:")
print(f"  dt_t0    = {dt_t0:.1e} (Step size in t0 units)")
print(f"  t_0      = {t_0_gyr:.6f} Gyr")
print("-" * 40)


# --- 3. Read and clean data ---
print("Loading data arrays...")
try:
    A_raw = np.load('A.npy').astype(np.float64).flatten()
    initial_raw = np.load('initial.npy').astype(np.float64)
    
    if initial_raw.ndim == 1:
        if initial_raw.size % 11 == 0:
            initial_raw = initial_raw.reshape(11, -1)
        else:
            raise ValueError(f"initial.npy size {initial_raw.size} is not divisible by 11.")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)

# --- 4. Strict alignment ---
len_A = A_raw.shape[0]
len_Init = initial_raw.shape[1]
common_len = min(len_A, len_Init)

if len_A != len_Init:
    print(f"Warning: Dimensions mismatch! A={len_A}, Initial={len_Init}")
    print(f"-> Auto-trimming data to length {common_len}")

A_arr = np.ascontiguousarray(A_raw[:common_len])

start = initial_raw[:, :common_len]
Ut = np.ascontiguousarray(start[0])
Rt = np.ascontiguousarray(start[1])
rhot = np.ascontiguousarray(start[2])
ept = np.ascontiguousarray(start[3])
Pt = np.ascontiguousarray(start[4])
wt = np.ascontiguousarray(start[5])
ephit = np.ascontiguousarray(start[6])
mt = np.ascontiguousarray(start[7])
Gammat = np.ascontiguousarray(start[8])
eA = np.ascontiguousarray(start[9])
qt = np.ascontiguousarray(start[10])

N_grid = common_len
print(f"Data ready. Grid size: {N_grid}")

# --- 5. Output file preparation ---
SUB_STEPS = 1
real_total_steps = total_steps // SUB_STEPS
if real_total_steps == 0: real_total_steps = 1
num_frames = math.ceil(real_total_steps / save_interval) + 1

if os.path.exists(filename): os.remove(filename)
print(f"Initializing output file: {filename} with {num_frames} frames...")
fp = open_memmap(filename, mode='w+', dtype='float64', shape=(num_frames, 12, N_grid))

state_out = np.zeros((12, N_grid))
p_fac_initial = compute_current_P_fac(Ut, Rt, rhot, ept, R0_scale, a, sigma)
state_out[0:11, :] = [Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt]
state_out[11, :] = p_fac_initial
fp[0] = state_out
fp.flush()

# --- 6. Plotting initialization ---
fig = None; ax = None; sc = None

if ENABLE_PLOTTING:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.ion()
    
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    
    ax.set_xscale('log')
    if plot_log_y: ax.set_yscale('log')
    ax.set_xlabel(plot_x_key)
    ax.set_ylabel(plot_y_key)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    var_map = {'Rt': Rt, 'rhot': rhot, 'Ut': Ut, 'ept': ept, 'qt': qt}
    x_data = var_map.get(plot_x_key, Rt)
    y_data = var_map.get(plot_y_key, rhot)

    norm = Normalize(vmin=0, vmax=real_total_steps * dt_t0 * SUB_STEPS)
    current_time_vals = np.zeros(N_grid)
    
    sc = ax.scatter(x_data, y_data, c=current_time_vals, cmap='gist_rainbow', s=0.5, norm=norm, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Time (t/t0)')
    
    plt.show()
    print("Plotting enabled (Overlay Mode).")
    print(f"Mode: HSE={FLAG_HSE}, P_FAC={FLAG_P_FAC}, HEAT={FLAG_HEAT}")

# --- 7. Main loop ---
print(f"Starting simulation...")
start_time = time.time()
current_step = 0
frame_idx = 1

while current_step < real_total_steps:
    steps_to_run = min(save_interval, real_total_steps - current_step)

    run_simulation_chunk_numba(
        Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
        A_arr, dt, Cq_viscosity, 
        R0_scale, sigma, fq, a, b, gm, ephib,
        steps_to_run, SUB_STEPS,
        freeze_radius, FLAG_HSE, FLAG_P_FAC, FLAG_HEAT, val_C
    )
    
    current_step += steps_to_run
    
    if frame_idx < num_frames:
        p_fac_now = compute_current_P_fac(Ut, Rt, rhot, ept, R0_scale, a, sigma)
        state_out[0:11, :] = [Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt]
        state_out[11, :] = p_fac_now
        fp[frame_idx] = state_out
        
        if ENABLE_PLOTTING:
            try:
                var_map = {'Rt': Rt, 'rhot': rhot, 'Ut': Ut, 'ept': ept, 'qt': qt}
                x_data = var_map.get(plot_x_key, Rt)
                y_data = var_map.get(plot_y_key, rhot)
                current_time_vals[:] = current_step * dt_t0 * SUB_STEPS

                ax.scatter(x_data, y_data, c=current_time_vals, cmap='gist_rainbow', s=0.5, norm=norm, alpha=0.3)
                
                plt.draw()
                plt.pause(0.001)
                
            except Exception as e:
                print(f"\nPlotting error (disabled): {e}")
                ENABLE_PLOTTING = False
        
        frame_idx += 1
    
    if current_step % save_interval == 0:

        print(f"Step {current_step/1000:.1f}k/{real_total_steps/1000:.1f}k | rho_c={rhot[0]:.4e}", end='\r', flush=True)

fp.flush()
print(f"\nSimulation complete. Time: {time.time()-start_time:.2f}s")

if ENABLE_PLOTTING:
    plt.ioff()
    plt.show()