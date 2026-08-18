import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 16
})

# ------------------------------------------
# PARÁMETROS GLOBALES Y CONFIGURACIÓN
# ------------------------------------------
p = 0.41
bs = 0.07
bd = 0.43

n_puntos_x = 40 
n_puntos_y = 40 
dd_vals = np.linspace(0.0, 0.8, n_puntos_x)
ds_vals = np.linspace(0.0, 0.08, n_puntos_y)
DD, DS = np.meshgrid(dd_vals, ds_vals)

n_reps = 10
t_max_sim = 3000

N_vals = [100, 1000]

dD_off = 0.06  # Tasa de muerte natural (sin tratamiento) de Células Diferenciadas
dS_off = 0.004 # Tasa de muerte natural (sin tratamiento) de CSC

ciclos = [
    (1.0, 10.0),   
    (7.0, 7.0),   
]

# ------------------------------------------
# FUNCIONES GILLESPIE: PARÁMETROS FIJOS
# ------------------------------------------
@njit
def run_gillespie_total(N, S_init, D_init, R_init, p, bs, bd, ds, dd, t_max):
    S1 = S_init
    D1 = D_init
    R1 = R_init
    t = 0.0
    
    while t < t_max:
        if S1 == 0 and D1 == 0:
            break
            
        a1 = p * bs * ((S1 * R1) / N)
        a2 = (1 - p) * bs * ((S1 * R1) / N)
        a3 = ds * S1
        a4 = bd * ((D1 * R1) / N)
        a5 = dd * D1
        a0 = a1 + a2 + a3 + a4 + a5
        
        if a0 == 0:
            t = t_max
            break
            
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        tau = -np.log(r1) / a0
        u = r2 * a0
        
        if u < a1:
            S1 += 1; R1 -= 1
        elif u < a1 + a2:
            D1 += 1; R1 -= 1
        elif u < a1 + a2 + a3:
            S1 -= 1; R1 += 1
        elif u < a1 + a2 + a3 + a4:
            D1 += 1; R1 -= 1
        else:
            D1 -= 1; R1 += 1
            
        t += tau
        
    return t

@njit(parallel=True)
def calcular_tiempos_extincion_total(n_puntos_y, n_puntos_x, DD, DS, p, bs, bd, N, n_reps, t_max_sim):
    T_ext_media_total = np.zeros_like(DD)
    
    for i in prange(n_puntos_y):
        for j in range(n_puntos_x):
            dd_val = DD[i, j]
            ds_val = DS[i, j]
            
            denominador = (1 - p) * bs * ds_val + p * bs * dd_val - bd * ds_val
            
            if ds_val < p * bs and ds_val < (p * bs / bd) * dd_val:
                xS_star = ((p * bs * dd_val - bd * ds_val) * (p * bs - ds_val)) / (p * bs * denominador)
                xD_star = ((1 - p) * ds_val * (p * bs - ds_val)) / (p * denominador)
            elif ds_val >= (p * bs / bd) * dd_val and dd_val < bd:
                xS_star = 0.0
                xD_star = 1.0 - (dd_val / bd)
            else:
                xS_star = 0.0
                xD_star = 0.0
                
            xS_star = max(0.0, xS_star)
            xD_star = max(0.0, xD_star)
                
            S_init = int(xS_star * N)
            D_init = int(xD_star * N)
            R_init = N - S_init - D_init
            
            if S_init == 0 and D_init == 0:
                T_ext_media_total[i, j] = 0.0
            else:
                suma_tiempos_total = 0.0
                for rep in range(n_reps):
                    t_ext = run_gillespie_total(N, S_init, D_init, R_init, p, bs, bd, ds_val, dd_val, t_max_sim)
                    suma_tiempos_total += t_ext
                T_ext_media_total[i, j] = suma_tiempos_total / n_reps
                
    return T_ext_media_total

# ------------------------------------------
# FUNCIONES GILLESPIE: PARÁMETROS CÍCLICOS
# ------------------------------------------
@njit
def run_gillespie_ciclos_total(N, S_init, D_init, R_init, p, bs, bd, ds_on, dd_on, ds_off, dd_off, T_on, T_off, t_max):
    S1 = S_init
    D1 = D_init
    R1 = R_init
    t = 0.0
    T_ciclo = T_on + T_off
    
    while t < t_max:
        if S1 == 0 and D1 == 0:
            break
            
        t_in_cycle = t % T_ciclo
        if t_in_cycle < T_on:
            ds = ds_on
            dd = dd_on
            time_to_next_phase = T_on - t_in_cycle
        else:
            ds = ds_off
            dd = dd_off
            time_to_next_phase = T_ciclo - t_in_cycle
            
        a1 = p * bs * ((S1 * R1) / N)
        a2 = (1 - p) * bs * ((S1 * R1) / N)
        a3 = ds * S1
        a4 = bd * ((D1 * R1) / N)
        a5 = dd * D1
        a0 = a1 + a2 + a3 + a4 + a5
        
        if a0 <= 0:
            t = t_max
            break
            
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        tau = -np.log(r1) / a0
        
        if tau > time_to_next_phase:
            t += time_to_next_phase
            continue
            
        u = r2 * a0
        
        if u < a1:
            S1 += 1; R1 -= 1
        elif u < a1 + a2:
            D1 += 1; R1 -= 1
        elif u < a1 + a2 + a3:
            S1 -= 1; R1 += 1
        elif u < a1 + a2 + a3 + a4:
            D1 += 1; R1 -= 1
        else:
            D1 -= 1; R1 += 1
            
        t += tau
        
    return t

@njit(parallel=True)
def calcular_tiempos_extincion_ciclos_total(n_puntos_y, n_puntos_x, DD_mean, DS_mean, p, bs, bd, dD_off, dS_off, T_on, T_off, N, n_reps, t_max_sim):
    T_ext_media_total = np.zeros_like(DD_mean)
    
    T_ciclo = T_on + T_off
    
    for i in prange(n_puntos_y):
        for j in range(n_puntos_x):
            dd_mean_val = DD_mean[i, j]
            ds_mean_val = DS_mean[i, j]
            
            dd_on = (dd_mean_val * T_ciclo - T_off * dD_off) / T_on
            ds_on = (ds_mean_val * T_ciclo - T_off * dS_off) / T_on
            
            if dd_on < dD_off:
                dd_on = dD_off
            if ds_on < dS_off:
                ds_on = dS_off
                
            denominador = (1 - p) * bs * ds_mean_val + p * bs * dd_mean_val - bd * ds_mean_val
            
            if ds_mean_val < p * bs and ds_mean_val < (p * bs / bd) * dd_mean_val:
                xS_star = ((p * bs * dd_mean_val - bd * ds_mean_val) * (p * bs - ds_mean_val)) / (p * bs * denominador)
                xD_star = ((1 - p) * ds_mean_val * (p * bs - ds_mean_val)) / (p * denominador)
            elif ds_mean_val >= (p * bs / bd) * dd_mean_val and dd_mean_val < bd:
                xS_star = 0.0
                xD_star = 1.0 - (dd_mean_val / bd)
            else:
                xS_star = 0.0
                xD_star = 0.0
                
            xS_star = max(0.0, xS_star)
            xD_star = max(0.0, xD_star)
                
            S_init = int(xS_star * N)
            D_init = int(xD_star * N)
            R_init = N - S_init - D_init
            
            if S_init == 0 and D_init == 0:
                T_ext_media_total[i, j] = 0.0
            else:
                suma_tiempos_total = 0.0
                for rep in range(n_reps):
                    t_ext = run_gillespie_ciclos_total(N, S_init, D_init, R_init, p, bs, bd, ds_on, dd_on, dS_off, dD_off, T_on, T_off, t_max_sim)
                    suma_tiempos_total += t_ext
                T_ext_media_total[i, j] = suma_tiempos_total / n_reps
                
    return T_ext_media_total

# ------------------------------------------
# SIMULACIÓN Y GENERACIÓN DE GRÁFICAS
# ------------------------------------------
if __name__ == '__main__':
    resultados_fijos = {}
    for N in N_vals:
        print(f"Tiempos de extinción fijos para N={N}...")
        T_fijo_tot = calcular_tiempos_extincion_total(
            n_puntos_y, n_puntos_x, DD, DS, p, bs, bd, N, n_reps, t_max_sim
        )
        resultados_fijos[N] = {'total': T_fijo_tot}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True, constrained_layout=True)

    diff_results = {}
    global_max_diff = 1.0

    for row_idx, (T_on, T_off) in enumerate(ciclos):
        T_ciclo = T_on + T_off
        print(f"\nTiempos de extinción cíclicos para Ciclo={T_ciclo} (Ton={T_on}, Toff={T_off})...")
        
        for col_idx, N in enumerate(N_vals):
            T_ciclo_tot = calcular_tiempos_extincion_ciclos_total(
                n_puntos_y, n_puntos_x, DD, DS, p, bs, bd, dD_off, dS_off, T_on, T_off, N, n_reps, t_max_sim
            )
            
            diff_tot = T_ciclo_tot - resultados_fijos[N]['total']
            diff_results[(row_idx, col_idx)] = diff_tot
            global_max_diff = max(global_max_diff, np.max(np.abs(diff_tot)))
            
    for row_idx, (T_on, T_off) in enumerate(ciclos):
        T_ciclo = T_on + T_off
        
        for col_idx, N in enumerate(N_vals):
            ax = axes[row_idx, col_idx]
            diff_tot = diff_results[(row_idx, col_idx)]
            
            im = ax.pcolormesh(DD, DS, diff_tot, cmap='coolwarm', shading='auto', vmin=-global_max_diff, vmax=global_max_diff)
            
            ax.set_title(f'N={N}, ($T_{{on}}$={T_on}, $T_{{off}}$={T_off})')
            
            ax.plot([0, bd], [0, p*bs], 'k--', linewidth=2.0, label='Frontera teórica')
            ax.plot([bd, 1.0], [p*bs, p*bs], 'k:', linewidth=2.0)
            ax.plot([bd, bd], [p*bs, 0.1], 'k-.', linewidth=2.0)
            ax.set_xlim(0, 0.8) 
            ax.set_ylim(0, 0.08)
            ax.legend(loc='upper right', fontsize=10)

    fig.supxlabel(r'Tasa media de muerte diferenciadas ($\langle d_D \rangle$)')
    fig.supylabel(r'Tasa media de muerte CSC ($\langle d_S \rangle$)')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label(r'$\Delta T_{ext}$ ($T_{ciclo} - T_{fijo}$)')

    plt.savefig('Comparacion_tiempos.png', dpi=300, bbox_inches='tight')
