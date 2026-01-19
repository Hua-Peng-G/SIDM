import numpy as np
from numba import njit

# --- 1. Helper functions ---
@njit(cache=True)
def avg_c2f(v):
    N = len(v) + 1
    out = np.empty(N, dtype=np.float64)
    out[1:-1] = 0.5 * (v[:-1] + v[1:])
    out[0] = v[0]; out[-1] = v[-1]
    return out

@njit(cache=True)
def avg_f2c(v):
    return 0.5 * (v[:-1] + v[1:])

@njit(cache=True)
def deriv_c2f(v_cell, coords_A):
    dV = v_cell[1:] - v_cell[:-1]
    dA = 0.5 * (coords_A[2:] - coords_A[:-2])
    N_face = len(v_cell) + 1
    grad = np.zeros(N_face, dtype=np.float64)
    grad[1:-1] = dV / dA
    grad[0] = grad[1]; grad[-1] = grad[-2]
    return grad

@njit(cache=True)
def deriv_f2c(v_face, coords_A):
    dA = coords_A[1:] - coords_A[:-1] 
    dV = v_face[1:] - v_face[:-1]      
    return dV / dA

@njit(cache=True)
def compute_current_P_fac(Ut, Rt, rhot, ept, R0_scale, a, sigma):
    N_cell = len(rhot) - 1
    P_fac_out = np.zeros(len(rhot), dtype=np.float64)
    for j in range(N_cell):
        u_val = np.abs(Ut[j+1])
        if u_val < 1e-99:
            P_fac = 1.0
        else:
            tsc = (Rt[j+1] * R0_scale) / (u_val / R0_scale)
            tr_denom = a * sigma * (rhot[j]/R0_scale**3) * np.sqrt(ept[j]/1.5/R0_scale)
            if tr_denom < 1e-99: P_fac = 0.0
            else:
                tr = 1.0 / tr_denom
                P_fac = 1.0 / (tr/tsc + 1.0)
        P_fac_out[j] = P_fac
    P_fac_out[-1] = 2*P_fac_out[-2] - P_fac_out[-3]
    return P_fac_out

# --- 2. Core evolution kernel ---
@njit(cache=True)
def evolve_kernel_unified(
    Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
    A, dt, Cq_viscosity, 
    R0_scale, sigma, fq, a, b, gm, ephib,
    freeze_radius,
    flag_hse, flag_P_fac, flag_heat,
    val_C 
):
    coeff_sigma = sigma / (R0_scale**1.5)
    N_face = len(Ut)
    N_cell = N_face - 1
    
    # === A. Determine calculation region ===
    limit_idx = N_face
    for i in range(N_face):
        if Rt[i] > freeze_radius:
            limit_idx = i
            break
    idx_end = limit_idx
    if idx_end < 5: idx_end = 5
    if idx_end > N_face: idx_end = N_face
    cell_end = min(N_cell, idx_end)

    # === B. Pre-calculate radiative heat flux ===
    rhot_cell = rhot[:-1]
    Pt_cell = Pt[:-1]
    ept_cell = ept[:-1] 
    
    qt_new = np.zeros(N_face, dtype=np.float64)
    delta_e_rad = np.zeros(N_cell, dtype=np.float64)
    
    if flag_heat:
        rhot_face_avg = avg_c2f(rhot_cell)
        ephit_cell_avg = avg_f2c(ephit)
        eeda = ept_cell * ephit_cell_avg
        deeda = deriv_c2f(eeda, A)
        deeda[0] = deeda[1]
        
        ep_face = avg_c2f(ept_cell)
        Pt_face = avg_c2f(Pt_cell)
        Vt_face = avg_c2f(1.0/rhot_cell)
        
        calc_end = min(N_face, idx_end + 1)
        for i in range(1, calc_end):
            denom = (1.0/val_C + a/b * sigma**2 / 4 / np.pi / R0_scale**4 * Pt_face[i])
            qt_new[i] = fq * np.sqrt(ep_face[i]) * Pt_face[i] / Vt_face[i] * Rt[i]**2 / ephit[i] * deeda[i] / denom
        
        Q_face = np.zeros(N_face, dtype=np.float64)
        for i in range(calc_end):
            Q_face[i] = 4 * np.pi * Rt[i]**2 * qt_new[i] * ephit[i]**2
        d4q_cell = deriv_f2c(Q_face, A)
        
        for j in range(cell_end):
            loss_rate = d4q_cell[j] / (R0_scale**3.5) * sigma / ephit_cell_avg[j]
            delta_e_rad[j] = -1.0 * loss_rate * dt

    # === C. Evolution branch selection ===
    if flag_hse:
        # --- HSE Branch ---
        Rt_iter = Rt.copy()
        Gammat_iter = Gammat.copy()
        ephit_iter = ephit.copy()
        rhot_iter = np.zeros(N_cell, dtype=np.float64)
        Pt_iter = np.zeros(N_cell, dtype=np.float64)
        wt_iter = np.zeros(N_cell, dtype=np.float64)
        ept_iter = ept_cell.copy()
        dphidA_arr = np.zeros(N_face, dtype=np.float64)
        phit_new_arr = np.zeros(N_face, dtype=np.float64)
        
        max_hse_iter = 100
        rel_tol = 1e-6
        pseudo_time_factor = 0.5 

        for k in range(max_hse_iter):
            for i in range(1, idx_end):
                val = 1.0 - 2.0 * mt[i] / (Rt_iter[i] * R0_scale)
                if val < 1e-9: val = 1e-9
                Gammat_iter[i] = np.sqrt(val)
            Gam_cell = avg_f2c(Gammat_iter)

            for j in range(cell_end):
                dVol = (4.0/3.0) * np.pi * (Rt_iter[j+1]**3 - Rt_iter[j]**3)
                if dVol < 1e-150: dVol = 1e-150
                dM = Gam_cell[j] * (A[j+1] - A[j])
                if dM < 1e-150: dM = 1e-150
                rhot_iter[j] = 1.0 / (dVol / dM)

            for j in range(cell_end):
                vol_old = 1.0 / rhot_cell[j]
                vol_new = 1.0 / rhot_iter[j]
                dV = vol_new - vol_old
                P_old = Pt_cell[j]
                
                denom_factor = 1.0 + 0.5 * (gm - 1.0) * rhot_iter[j] * dV
                numerator = ept_cell[j] + delta_e_rad[j] - 0.5 * P_old * dV
                
                if denom_factor < 1e-5: denom_factor = 1e-5
                e_val = numerator / denom_factor
                if e_val < 1e-10: e_val = 1e-10
                ept_iter[j] = e_val
                Pt_iter[j] = (gm - 1.0) * ept_iter[j] * rhot_iter[j]
                wt_iter[j] = 1.0 + (ept_iter[j] + Pt_iter[j]/rhot_iter[j]) / R0_scale

            Pt_face_iter = avg_c2f(Pt_iter)
            rhot_face_iter = avg_c2f(rhot_iter)
            wt_face_iter = avg_c2f(wt_iter)
            dPdA_iter = deriv_c2f(Pt_iter, A)
            
            eA_iter = np.zeros(N_face, dtype=np.float64)
            for i in range(1, idx_end):
                 denom = Rt_iter[i]**2 * rhot_face_iter[i]**2
                 if denom > 1e-150:
                     eA_iter[i] = qt_new[i] / (4 * np.pi * denom)
            
            dphidA_arr[:] = 0.0 
            for i in range(idx_end):
                if np.abs(wt_face_iter[i]) > 1e-99:
                    term_rad_iter = 0
                    bracket = dPdA_iter[i] / rhot_face_iter[i] + term_rad_iter
                    dphidA_arr[i] = (-1.0 / wt_face_iter[i]) * bracket
            
            phit_new_arr[-1] = np.log(ephib) * R0_scale
            for i in range(N_face - 2, -1, -1):
                 dA_val = A[i+1] - A[i]
                 val = phit_new_arr[i+1] - 0.5 * (dphidA_arr[i] + dphidA_arr[i+1]) * dA_val
                 phit_new_arr[i] = val
            for i in range(N_face):
                 ephit_iter[i] = np.exp(phit_new_arr[i] / R0_scale)

            max_rel_change = 0.0
            delta_R = np.zeros(N_face, dtype=np.float64)
            for i in range(1, idx_end): 
                dphidA_val = dphidA_arr[i]
                term1 = -1.0 * Gammat_iter[i]**2 * dphidA_val * 4 * np.pi * Rt_iter[i]**2 * rhot_face_iter[i] / ephit_iter[i]
                term2 = mt[i] / (Rt_iter[i]**2)
                term3 = 4 * np.pi * Pt_face_iter[i] * Rt_iter[i] / R0_scale
                
                acc_val = (-1.0 * ephit_iter[i] * (term1 + term2 + term3)) / R0_scale
                if mt[i] < 1e-150: t_dyn_sq = 1.0
                else: t_dyn_sq = Rt_iter[i]**3 / mt[i]
                
                step_val = pseudo_time_factor * t_dyn_sq * acc_val
                limit_frac = 0.1 
                max_step = limit_frac * Rt_iter[i]
                if step_val > max_step: step_val = max_step
                if step_val < -max_step: step_val = -max_step
                delta_R[i] = step_val
                
                curr_R = Rt_iter[i] if Rt_iter[i] > 1e-150 else 1e-150
                rel_change = np.abs(step_val) / curr_R
                if rel_change > max_rel_change: max_rel_change = rel_change
            
            for i in range(1, idx_end):
                Rt_iter[i] += delta_R[i]
                if Rt_iter[i] <= Rt_iter[i-1]:
                    Rt_iter[i] = Rt_iter[i-1] + (Rt_iter[i+1] - Rt_iter[i-1]) * 0.1 

            if max_rel_change < rel_tol:
                break
        
        for i in range(1, idx_end):
            Rt[i] = Rt_iter[i]
            Gammat[i] = Gammat_iter[i]
        for i in range(N_face):
            ephit[i] = ephit_iter[i] 
            Ut[i] = 0.0 
        for j in range(cell_end):
            rhot[j] = rhot_iter[j]
            ept[j] = ept_iter[j] 
            Pt[j] = Pt_iter[j]
            wt[j] = wt_iter[j]

    else:
        # --- Dynamic Branch ---
        rhot_face_avg = avg_c2f(rhot_cell)
        Pt_face_avg = avg_c2f(Pt_cell)
        wt_face_avg = avg_c2f(wt)
        dPdA = deriv_c2f(Pt_cell, A)
        dUdt = np.zeros(N_face, dtype=np.float64)
        loop_start_acc = max(1, 0) 

        for i in range(loop_start_acc, idx_end):
            denom_eA = Rt[i]**2 * rhot_face_avg[i]**2
            if denom_eA < 1e-150: denom_eA = 1e-150
            eA_term = qt_new[i] / (4 * np.pi * denom_eA) 
            term_rad = coeff_sigma / ephit[i] * (eA_term - eA[i]) / dt
            bracket = dPdA[i] / rhot_face_avg[i] + term_rad
            dphidA_val = (-1.0 / wt_face_avg[i]) * bracket
            term1 = -1.0 * Gammat[i]**2 * dphidA_val * 4 * np.pi * Rt[i]**2 * rhot_face_avg[i] / ephit[i]
            term2 = mt[i] / (Rt[i]**2)
            term3 = 4 * np.pi * Pt_face_avg[i] * Rt[i] / R0_scale
            dUdt[i] = (-1.0 * ephit[i] * (term1 + term2 + term3)) / R0_scale

        Ut_new = Ut.copy()
        Rt_new = Rt.copy()
        for i in range(loop_start_acc, idx_end):
            Ut_new[i] = Ut[i] + dUdt[i] * dt
            
        # Outer boundary treatment
        if idx_end == N_face:
            boundary_width = 20 
            start_smooth = N_face - boundary_width
            if start_smooth < loop_start_acc: start_smooth = loop_start_acc
            for _ in range(2): 
                temp_last = Ut_new[start_smooth-1]
                for i in range(start_smooth, N_face - 1):
                    val_smooth = 0.25 * temp_last + 0.5 * Ut_new[i] + 0.25 * Ut_new[i+1]
                    temp_last = Ut_new[i] 
                    Ut_new[i] = val_smooth
            extrap_val = 2.0 * Ut_new[-2] - Ut_new[-3]
            if extrap_val > Ut_new[-2]: Ut_new[-1] = Ut_new[-2]
            else: Ut_new[-1] = extrap_val
        
        for i in range(loop_start_acc, idx_end):
             Rt_new[i] = Rt[i] + ephit[i] * Ut_new[i] * dt / (R0_scale**2)
        
        Gammat_new = Gammat.copy()
        for i in range(1, idx_end):
            val = 1.0 + (Ut_new[i]/R0_scale)**2 - 2 * mt[i] / (Rt_new[i] * R0_scale)
            if val < 0: val = 0
            Gammat_new[i] = np.sqrt(val)
        Gam_cell = avg_f2c(Gammat_new)

        rhot_new = rhot.copy()
        ept_new = ept.copy()
        Pt_new = Pt.copy()
        wt_new = wt.copy()

        for j in range(cell_end):
            dVol = (4.0/3.0) * np.pi * (Rt_new[j+1]**3 - Rt_new[j]**3)
            dM = Gam_cell[j] * (A[j+1] - A[j])
            if dM < 1e-150: dM = 1e-150
            rhot_new[j] = 1.0 / (dVol / dM)
            
            vol_old = 1.0 / rhot[j]
            vol_new = 1.0 / rhot_new[j]
            dV = vol_new - vol_old
            P_old = Pt[j]
            denom_factor = 1.0 + 0.5 * (gm - 1.0) * rhot_new[j] * dV
            numerator = ept[j] + delta_e_rad[j] - 0.5 * P_old * dV
            if denom_factor < 1e-5: denom_factor = 1e-5
            e_val = numerator / denom_factor
            if e_val < 1e-10: e_val = 1e-10
            ept_new[j] = e_val
            
            P_fac = 1.0
            if flag_P_fac:
                u_val = np.abs(Ut_new[j+1]) 
                if u_val < 1e-99: P_fac = 1.0
                else:
                    tsc = (Rt_new[j+1] * R0_scale) / (u_val / R0_scale)
                    tr_denom = a * sigma * (rhot_new[j]/R0_scale**3) * np.sqrt(ept_new[j]/1.5/R0_scale)
                    if tr_denom < 1e-99: P_fac = 0.0 
                    else:
                        tr = 1.0 / tr_denom
                        P_fac = 1.0 / (tr/tsc + 1.0)
            
            Pt_new[j] = (gm - 1.0) * ept_new[j] * rhot_new[j] * P_fac
            wt_new[j] = 1.0 + (ept_new[j] + Pt_new[j]/rhot_new[j]) / R0_scale

        for i in range(idx_end):
            Ut[i] = Ut_new[i]
            Rt[i] = Rt_new[i]
            Gammat[i] = Gammat_new[i]
        for j in range(cell_end):
            rhot[j] = rhot_new[j]
            ept[j] = ept_new[j]
            Pt[j] = Pt_new[j]
            wt[j] = wt_new[j]
            
    # === D. Finalize ===
    for i in range(idx_end):
        qt[i] = qt_new[i]
        denom = Rt[i]**2 * avg_c2f(rhot)[i]**2
        if denom > 1e-150:
            eA[i] = qt[i] / (4 * np.pi * denom)

    if idx_end == N_face:
         rhot[-1] = rhot[-2]
         Pt[-1] = Pt[-2]
         ept[-1] = ept[-2]
         wt[-1] = wt[-2]

# --- 3. Driver ---
@njit(cache=True)
def run_simulation_chunk_numba(
    Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
    A, dt, Cq_viscosity, R0_scale, sigma, fq, a, b, gm, ephib,
    chunk_steps, sub_steps,
    freeze_radius, flag_hse, flag_P_fac, flag_heat, val_C
):
    for step in range(chunk_steps):
        for _ in range(sub_steps):
            evolve_kernel_unified(
                Ut, Rt, rhot, ept, Pt, wt, ephit, mt, Gammat, eA, qt,
                A, dt, Cq_viscosity, R0_scale, sigma, fq, a, b, gm, ephib,
                freeze_radius, flag_hse, flag_P_fac, flag_heat, val_C
            )