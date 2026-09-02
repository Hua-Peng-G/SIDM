# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

# --- 1. Low-level auxiliary functions ---

# Calculate cell-centered coordinates
@njit(cache=True)
def calc_Ac(A):
    return 0.5 * (A[:-1] + A[1:])

# Non-uniform grid averaging: cell-centered to face-centered
@njit(cache=True)
def avg_c2f_non(v_cell, A):
    N_face = len(A)
    out = np.zeros(N_face, dtype=np.float64)
    out[0] = v_cell[0]
    out[-1] = v_cell[-1]
    Ac = calc_Ac(A)
    for i in range(1, N_face - 1):
        d_left = A[i] - Ac[i-1]
        d_right = Ac[i] - A[i]
        out[i] = (v_cell[i-1] * d_right + v_cell[i] * d_left) / max(d_left + d_right, 1e-180)
    return out

# Non-uniform grid averaging: face-centered to cell-centered
@njit(cache=True)
def avg_f2c_non(v_face, A):
    N_cell = len(A) - 1
    out = np.zeros(N_cell, dtype=np.float64)
    Ac = calc_Ac(A)
    for j in range(N_cell):
        d_left = Ac[j] - A[j]
        d_right = A[j+1] - Ac[j]
        out[j] = (v_face[j] * d_right + v_face[j+1] * d_left) / max(d_left + d_right, 1e-180)
    return out

# Non-uniform grid derivative: cell-centered to face-centered
@njit(cache=True)
def deriv_c2f_non(v_cell, A):
    N_face = len(A)
    grad = np.zeros(N_face, dtype=np.float64)
    Ac = calc_Ac(A)
    for i in range(1, N_face - 1):
        grad[i] = (v_cell[i] - v_cell[i-1]) / max(Ac[i] - Ac[i-1], 1e-180)
    grad[0] = grad[1]
    grad[-1] = grad[-2]
    return grad

# Non-uniform grid derivative: face-centered to cell-centered
@njit(cache=True)
def deriv_f2c_non(v_face, A):
    N_cell = len(A) - 1
    grad = np.zeros(N_cell, dtype=np.float64)
    for j in range(N_cell):
        grad[j] = (v_face[j+1] - v_face[j]) / max(A[j+1] - A[j], 1e-180)
    return grad


# --- 2. Core evolution kernel ---
@njit(cache=True)
def evolve_kernel_unified(
    Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
    A, dt, Cq_viscosity, 
    R0_scale, sigma, fq, a, b, gm, ephib, w,
    freeze_radius,
    flag_hse, flag_heat,
    flag_bh_formed, idx_bh_arr, eps_exc,
    flag_damp, damp_coeff,
    flag_smooth_cell,
    v_limit_factor
):
    N_face = len(Ut)
    N_cell = N_face - 1
    max_hse_iter, rel_tol, pseudo_time_factor = 1000, 1e-5, 0.5
    
    v_dep_const = (2.0 / (3.0 * R0_scale)) * ((3e5/w)**2)
    
    rhot_cell = rhot[:-1]
    ept_cell = ept[:-1]
    Pt_cell = Pt[:-1]

    Rt_iter = Rt.copy()
    Gammat_iter = Gammat.copy()
    ephit_iter = ephit.copy()
    rhot_iter = rhot_cell.copy()
    ept_iter = ept_cell.copy()
    Pt_iter = Pt_cell.copy()
    wt_iter = wt[:-1].copy()
    
    qt_new = np.zeros(N_face, dtype=np.float64)
    delta_e_rad = np.zeros(N_cell, dtype=np.float64)
    dphidA_arr = np.zeros(N_face, dtype=np.float64)
    phit_new_arr = np.zeros(N_face, dtype=np.float64)

    # Determine boundary based on freeze_radius
    idx_bh = idx_bh_arr[0] if flag_bh_formed else 0
    limit_idx = N_face
    for i in range(N_face):
        if Rt[i] > freeze_radius:
            limit_idx = i
            break
    idx_end = max(5, min(limit_idx, N_face))
    cell_end = min(N_cell, idx_end)
    cell_start = idx_bh if flag_bh_formed else 0
    loop_start = max(1, idx_bh + 1) if flag_bh_formed else 1

    if flag_hse:
        # =========================================================
        # HSE (Hydrostatic Equilibrium) branch
        # =========================================================
        c_sigma_arr = np.zeros(N_cell, dtype=np.float64)
        for j in range(N_cell):
            c_sigma_arr[j] = 1.0 / (1.0 + v_dep_const * ept_cell[j])**2
        c_sigma_f = avg_c2f_non(c_sigma_arr, A)
        
        for k in range(max_hse_iter):
            for i in range(loop_start, idx_end):
                val = 1.0 - 2.0 * mt[i] / (Rt_iter[i] * R0_scale)
                Gammat_iter[i] = np.sqrt(val)
            Gam_cell = avg_f2c_non(Gammat_iter, A)

            # Starting point for density calculation
            j_start = cell_start if not flag_bh_formed else cell_start + 1
            for j in range(j_start, cell_end):
                dVol = (4.0/3.0) * np.pi * (Rt_iter[j+1]**3 - Rt_iter[j]**3)
                rhot_iter[j] = (Gam_cell[j] * (A[j+1] - A[j])) / dVol
            if flag_bh_formed:
                rhot_iter[cell_start] = rhot_iter[cell_start + 1]

            if flag_heat and cell_end > cell_start:
                ephit_c_avg = avg_f2c_non(ephit_iter, A)
                ep_f = avg_c2f_non(ept_iter, A)
                Pt_f = avg_c2f_non(Pt_iter, A)
                Vt_f = avg_c2f_non(1.0/rhot_iter, A)
                deeda = deriv_c2f_non(ept_iter * ephit_c_avg, A)
                
                for i in range(loop_start, min(N_face, idx_end + 1)):
                    denom_rad = (4.0/3.0 + a/b * (sigma * c_sigma_f[i])**2 / (4 * np.pi * R0_scale**4) * Pt_f[i])
                    qt_new[i] = fq * Gammat_iter[i] * np.sqrt(ep_f[i]) * Pt_f[i] / Vt_f[i] * Rt_iter[i]**2 / ephit_iter[i] * deeda[i] / denom_rad
                
                Q_f = np.zeros(N_face, dtype=np.float64)
                for i in range(idx_bh, min(N_face, idx_end + 1)):
                    Q_f[i] = 4 * np.pi * Rt_iter[i]**2 * qt_new[i] * ephit_iter[i]**2
                d4q = deriv_f2c_non(Q_f, A)
                for j in range(cell_start, cell_end):
                    delta_e_rad[j] = -1.0 * (d4q[j] / (R0_scale**3.5) * (sigma * c_sigma_arr[j]) / ephit_c_avg[j]) * dt

            # Starting point for energy update
            for j in range(j_start, cell_end):
                vol_old = 1.0 / rhot_cell[j]
                vol_new = 1.0 / rhot_iter[j]
                dV = vol_new - vol_old
                denom_fac = 1.0 + 0.5 * (gm - 1.0) * rhot_iter[j] * dV
                ept_iter[j] = (ept_cell[j] + delta_e_rad[j] - 0.5 * Pt_cell[j] * dV) / denom_fac
                Pt_iter[j] = (gm - 1.0) * ept_iter[j] * rhot_iter[j]
                wt_iter[j] = 1.0 + (ept_iter[j] + Pt_iter[j]/rhot_iter[j]) / R0_scale
            if flag_bh_formed:
                ept_iter[cell_start], Pt_iter[cell_start], wt_iter[cell_start] = ept_iter[cell_start+1], Pt_iter[cell_start+1], wt_iter[cell_start+1]

            dPdA_h = deriv_c2f_non(Pt_iter, A)
            wt_f_h = avg_c2f_non(wt_iter, A)
            rf_f_h = avg_c2f_non(rhot_iter, A)
            for i in range(loop_start, idx_end):
                dphidA_arr[i] = (-1.0 / wt_f_h[i]) * (dPdA_h[i] / rf_f_h[i])
            
            phit_new_arr[-1] = np.log(ephib) * R0_scale
            for i in range(N_face - 2, max(-1, idx_bh - 1), -1):
                phit_new_arr[i] = phit_new_arr[i+1] - 0.5 * (dphidA_arr[i] + dphidA_arr[i+1]) * (A[i+1] - A[i])
            for i in range(N_face): 
                ephit_iter[i] = np.exp(phit_new_arr[i] / R0_scale)

            max_rel_change = 0.0
            rf_f_it = avg_c2f_non(rhot_iter, A)
            Pt_f_it = avg_c2f_non(Pt_iter, A)
            if k < 5: current_factor = 1 
            elif k < 15: current_factor = 0.5
            else: current_factor = 0.1

            for i in range(loop_start, idx_end): 
                term1 = -1.0 * Gammat_iter[i]**2 * dphidA_arr[i] * 4 * np.pi * Rt_iter[i]**2 * rf_f_it[i] / ephit_iter[i]
                term2 = mt[i] / (Rt_iter[i]**2)
                term3 = 4 * np.pi * Pt_f_it[i] * Rt_iter[i] / R0_scale
                acc_val = (-1.0 * ephit_iter[i] * (term1 + term2 + term3)) / R0_scale
                t_dyn_sq = 1.0 if mt[i] == 0.0 else Rt_iter[i]**3 / mt[i]
                step_val = current_factor * t_dyn_sq * acc_val
                limit = 0.2 * (Rt_iter[i] - Rt_iter[i-1]) if i > 0 else 0.2 * Rt_iter[i]
                step_val = max(-limit, min(limit, step_val))
                Rt_iter[i] += step_val
                if i > 0 and Rt_iter[i] <= Rt_iter[i-1]: Rt_iter[i] = Rt_iter[i-1] * (1.0 + 1e-12)
                rel = np.abs(step_val) / Rt_iter[i]
                if rel > max_rel_change: max_rel_change = rel

            # Synchronous excision of the black hole horizon in HSE mode
            if flag_bh_formed:
                while idx_bh < N_face - 2:
                    val_g_hse = 1.0 - 2.0 * mt[idx_bh+1] / (Rt_iter[idx_bh+1] * R0_scale)
                    if Rt_iter[idx_bh+1] <= 2.0 * mt[idx_bh+1] / R0_scale * (1.0 + eps_exc) or val_g_hse < 1e-8:
                        idx_bh += 1
                        idx_bh_arr[0], Rt_iter[idx_bh] = idx_bh, Rt[idx_bh]
                        loop_start, cell_start = idx_bh + 1, idx_bh
                    else: break

            if max_rel_change < rel_tol: break

        for i in range(idx_end):
            Rt[i], Gammat[i], ephit[i], Ut[i] = Rt_iter[i], Gammat_iter[i], ephit_iter[i], 0.0
        for j in range(cell_end):
            rhot[j], ept[j], Pt[j], wt[j] = rhot_iter[j], ept_iter[j], Pt_iter[j], wt_iter[j]

    else:
        # =========================================================
        # Dynamic branch
        # =========================================================
        c_sigma_arr = np.zeros(N_cell, dtype=np.float64)
        for j in range(N_cell):
            c_sigma_arr[j] = 1.0 / (1.0 + v_dep_const * ept_cell[j])**2
        c_sigma_f = avg_c2f_non(c_sigma_arr, A)
        
        rhot_face_avg, P_tot_face_avg = avg_c2f_non(rhot_cell, A), avg_c2f_non(Pt_cell, A)
        dPdA_tot, wt_face_avg = deriv_c2f_non(Pt_cell, A), avg_c2f_non(wt, A)
        dUdt = np.zeros(N_face, dtype=np.float64)

        if flag_heat:
            ephit_c_avg = avg_f2c_non(ephit, A)
            ep_f, Pt_f = avg_c2f_non(ept_cell, A), avg_c2f_non(Pt_cell, A)
            Vt_f = avg_c2f_non(1.0/rhot_cell, A)
            deeda = deriv_c2f_non(ept_cell * ephit_c_avg, A)
            for i in range(loop_start, min(N_face, idx_end + 1)):
                denom_r = (4.0/3.0 + a/b * (sigma * c_sigma_f[i])**2 / (4 * np.pi * R0_scale**4) * Pt_f[i])
                qt_new[i] = fq * np.sqrt(ep_f[i]) * Pt_f[i] / Vt_f[i] * Rt[i]**2 / ephit[i] * deeda[i] / denom_r
            Q_f = np.zeros(N_face, dtype=np.float64)
            for i in range(idx_bh, min(N_face, idx_end + 1)): Q_f[i] = 4 * np.pi * Rt[i]**2 * qt_new[i] * ephit[i]**2
            d4q = deriv_f2c_non(Q_f, A)
            for j in range(cell_start, cell_end):
                delta_e_rad[j] = -1.0 * (d4q[j] / (R0_scale**3.5) * (sigma * c_sigma_arr[j]) / ephit_c_avg[j]) * dt

        for i in range(loop_start, idx_end):
            denom_eA = Rt[i]**2 * rhot_face_avg[i]**2
            eA_term = qt_new[i] / (4 * np.pi * denom_eA) 
            term_rad = ((sigma * c_sigma_f[i]) / (R0_scale**1.5)) / ephit[i] * (eA_term - eA[i]) / dt
            dphidA_val = (-1.0 / wt_face_avg[i]) * (dPdA_tot[i] / rhot_face_avg[i] + term_rad)
            dUdt[i] = (-1.0 * ephit[i] * (-1.0 * Gammat[i]**2 * dphidA_val * 4 * np.pi * Rt[i]**2 * rhot_face_avg[i] / ephit[i] + mt[i] / (Rt[i]**2) + 4 * np.pi * P_tot_face_avg[i] * Rt[i] / R0_scale)) / R0_scale
            if flag_damp: dUdt[i] -= damp_coeff * Ut[i]
                
        Ut_new, Rt_new = Ut.copy(), Rt.copy()
        for i in range(loop_start, idx_end): Ut_new[i] = Ut[i] + dUdt[i] * dt

        Gammat_new = Gammat.copy()
        for i in range(loop_start, idx_end):
            val_g = 1.0 + (Ut_new[i]/R0_scale)**2 - 2 * mt[i] / (Rt_new[i] * R0_scale)
            Gammat_new[i] = np.sqrt(val_g)

        if flag_bh_formed:
            while idx_bh < N_face - 2:
                if Rt_new[idx_bh+1] <= 2.0 * mt[idx_bh+1] / R0_scale * (1.0 + eps_exc) or Gammat_new[idx_bh+1] < 1e-4:
                    idx_bh += 1
                    idx_bh_arr[0], Ut_new[idx_bh], Rt_new[idx_bh] = idx_bh, 0.0, Rt[idx_bh]
                    loop_start, cell_start = idx_bh + 1, idx_bh
                else: break

        if idx_end == N_face:
            start_smooth = max(loop_start, N_face - 5)
            for _ in range(2): 
                temp_last = Ut_new[start_smooth-1]
                for i in range(start_smooth, N_face - 1):
                    val_smooth = 0.25 * temp_last + 0.5 * Ut_new[i] + 0.25 * Ut_new[i+1]
                    temp_last, Ut_new[i] = Ut_new[i], val_smooth
            extrap_val = 2.0 * Ut_new[-2] - Ut_new[-3]
            Ut_new[-1] = min(extrap_val, Ut_new[-2]) if extrap_val > Ut_new[-2] else extrap_val

        cs_f_d = avg_c2f_non(np.sqrt(gm * Pt_cell / rhot_cell), A)
        for i in range(loop_start, idx_end):
            v_lim = v_limit_factor * cs_f_d[i]
            Ut_new[i] = max(-v_lim, min(v_lim, Ut_new[i]))
            Rt_new[i] = Rt[i] + ephit[i] * Ut_new[i] * dt / (R0_scale**2)

        Gam_cell_d = avg_f2c_non(Gammat_new, A)
        rhot_new, ept_new, Pt_new, wt_new = rhot.copy(), ept.copy(), Pt.copy(), wt.copy()
        
        # Fix for density and energy update skipping Cell 0
        j_start_dyn = cell_start if not flag_bh_formed else cell_start + 1
        for j in range(j_start_dyn, cell_end):
            dVol = (4.0/3.0) * np.pi * (Rt_new[j+1]**3 - Rt_new[j]**3)
            rhot_new[j] = 1.0 / (dVol / (Gam_cell_d[j] * (A[j+1] - A[j])))
            
            dv = 1.0/rhot_new[j] - 1.0/rhot[j]
            den = 1.0 + 0.5 * (gm - 1.0) * rhot_new[j] * dv
            ept_new[j] = (ept[j] + delta_e_rad[j] - 0.5 * Pt[j] * dv) / den
            
            Pt_new[j] = (gm - 1.0) * ept_new[j] * rhot_new[j]
            wt_new[j] = 1.0 + (ept_new[j] + Pt_new[j]/rhot_new[j]) / R0_scale

        if flag_bh_formed:
            target_j = cell_start 
            neighbor_j = cell_start + 1
            rhot_new[target_j] = rhot_new[neighbor_j]
            ept_new[target_j]  = ept_new[neighbor_j]
            Pt_new[target_j]   = Pt_new[neighbor_j]
            wt_new[target_j]   = wt_new[neighbor_j]

        for i in range(loop_start, idx_end): Ut[i], Rt[i], Gammat[i] = Ut_new[i], Rt_new[i], Gammat_new[i]
        for j in range(idx_bh, cell_end): rhot[j], ept[j], Pt[j], wt[j] = rhot_new[j], ept_new[j], Pt_new[j], wt_new[j]

    # === D. General finalization ===
    rf_end = avg_c2f_non(rhot[:-1], A)
    for i in range(idx_bh, idx_end):
        qt[i] = qt_new[i]
        den_eA = Rt[i]**2 * rf_end[i]**2
        if den_eA > 0: eA[i] = qt[i] / (4*np.pi*den_eA)
        
    if idx_end == N_face:
        rhot[-1], Pt[-1], ept[-1], wt[-1] = rhot[-2], Pt[-2], ept[-2], wt[-2]

# --- 3. Simulation driver ---
@njit(cache=True)
def run_simulation_chunk_numba(
    Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
    A, dt, Cq_viscosity, R0_scale, sigma, fq, a, b, gm, ephib, w,
    chunk_steps, split_index, sub_steps,
    freeze_radius, flag_hse, flag_heat,
    flag_bh_formed, idx_bh_arr, eps_exc,
    flag_damp, damp_coeff,
    flag_smooth_cell,
    v_limit_factor
):
    for step in range(chunk_steps):
        for _ in range(sub_steps):
            evolve_kernel_unified(
                Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
                A, dt, Cq_viscosity, R0_scale, sigma, fq, a, b, gm, ephib, w,
                freeze_radius, flag_hse, flag_heat,
                flag_bh_formed, idx_bh_arr, eps_exc,
                flag_damp, damp_coeff,
                flag_smooth_cell,
                v_limit_factor
            )