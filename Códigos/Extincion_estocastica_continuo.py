import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.titlesize': 16
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

cmap = plt.cm.plasma
cmap.set_over('darkgray')

#  -----FUNCIÓN GILLESPIE (EXTINCIÓN TOTAL)-----
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

#  -----FUNCIÓN GILLESPIE (EXTINCIÓN CSC)-----
@njit
def run_gillespie_csc(N, S_init, D_init, R_init, p, bs, bd, ds, dd, t_max):
    S1 = S_init
    D1 = D_init
    R1 = R_init
    t = 0.0
    
    while t < t_max:
        if S1 == 0:
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
def calcular_tiempos_extincion(n_puntos_y, n_puntos_x, DD, DS, p, bs, bd, N, n_reps, t_max_sim):
    T_ext_media_total = np.zeros_like(DD)
    T_ext_media_csc = np.zeros_like(DD)
    
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
                
            if S_init == 0:
                T_ext_media_csc[i, j] = 0.0
            else:
                suma_tiempos_csc = 0.0
                for rep in range(n_reps):
                    t_ext = run_gillespie_csc(N, S_init, D_init, R_init, p, bs, bd, ds_val, dd_val, t_max_sim)
                    suma_tiempos_csc += t_ext
                T_ext_media_csc[i, j] = suma_tiempos_csc / n_reps
            
    return T_ext_media_total, T_ext_media_csc

#  -----GRAFICACIÓN-----
fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True, constrained_layout=True)

for idx, N in enumerate(N_vals):
    print(f"Calculando tiempos de extinción (N={N})...")
    T_ext_media_total, T_ext_media_csc = calcular_tiempos_extincion(n_puntos_y, n_puntos_x, DD, DS, p, bs, bd, N, n_reps, t_max_sim)
    
    im1 = axes[idx, 0].pcolormesh(DD, DS, T_ext_media_total, cmap=cmap, shading='auto', vmin=0, vmax=t_max_sim-100)

    axes[idx, 0].plot([0, bd], [0, p*bs], 'w--', linewidth=2.5, label='Frontera teórica')
    axes[idx, 0].plot([bd, 1.0], [p*bs, p*bs], 'w:', linewidth=2.5)
    axes[idx, 0].plot([bd, bd], [p*bs, 0.1], 'w-.', linewidth=2.5)

    axes[idx, 0].set_xlim(0, 0.8) 
    axes[idx, 0].set_ylim(0, 0.08)
    axes[idx, 0].set_title(f'Extinción Total (N={N})')
    axes[idx, 0].legend(loc='upper right', fontsize=10)

    # Gráfica 2: Extinción CSC
    im2 = axes[idx, 1].pcolormesh(DD, DS, T_ext_media_csc, cmap=cmap, shading='auto', vmin=0, vmax=t_max_sim-100)

    axes[idx, 1].plot([0, bd], [0, p*bs], 'w--', linewidth=2.5, label='Frontera teórica')
    axes[idx, 1].plot([bd, 1.0], [p*bs, p*bs], 'w:', linewidth=2.5)
    axes[idx, 1].plot([bd, bd], [p*bs, 0.1], 'w-.', linewidth=2.5)

    axes[idx, 1].set_xlim(0, 0.8) 
    axes[idx, 1].set_ylim(0, 0.08)
    axes[idx, 1].set_title(f'Extinción CSC (N={N})')
    axes[idx, 1].legend(loc='upper right', fontsize=10)

fig.supxlabel('Tasa muerte dif. ($d_D$)')
fig.supylabel('Tasa muerte CSC ($d_S$)')

cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), extend='max')
cbar.set_label('Tiempo Medio Extinción $\\langle T_{ext} \\rangle$')

plt.savefig('figura_4_6.png', dpi=300, bbox_inches='tight')
plt.show()
