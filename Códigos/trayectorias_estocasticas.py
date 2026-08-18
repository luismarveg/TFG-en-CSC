import numpy as np
import matplotlib.pyplot as plt
from numba import njit

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 18
})

# ------------------------------------------
# PARÁMETROS GLOBALES Y CONFIGURACIÓN
# ------------------------------------------
p = 0.41
bs = 0.07
bd = 0.43
N = 1000

dD_off = 0.06  # Tasa muerte natural Diferenciadas
dS_off = 0.004 # Tasa muerte natural CSC

# Punto de evaluación antes de la frontera determinista 
dd_mean = 0.38
ds_mean = 0.03

# Tiempos del ciclo a evaluar
T_on = 1
T_off = 10
T_ciclo = T_on + T_off

# Tasas ON para mantener la media requerida
dd_on = (dd_mean * T_ciclo - T_off * dD_off) / T_on
ds_on = (ds_mean * T_ciclo - T_off * dS_off) / T_on
if dd_on < dD_off: dd_on = dD_off
if ds_on < dS_off: ds_on = dS_off

# Determinación de la población inicial (estacionario sin tratamiento)
denominador = (1 - p) * bs * dS_off + p * bs * dD_off - bd * dS_off
if dS_off < p * bs and dS_off < (p * bs / bd) * dD_off:
    xS_star = ((p * bs * dD_off - bd * dS_off) * (p * bs - dS_off)) / (p * bs * denominador)
    xD_star = ((1 - p) * dS_off * (p * bs - dS_off)) / (p * denominador)
elif dS_off >= (p * bs / bd) * dD_off and dD_off < bd:
    xS_star = 0.0
    xD_star = 1.0 - (dD_off / bd)
else:
    xS_star = 0.0
    xD_star = 0.0
    
xS_star = max(0.0, xS_star)
xD_star = max(0.0, xD_star)
S_init = int(xS_star * N)
D_init = int(xD_star * N)
R_init = N - S_init - D_init

print(f"Población inicial (estacionario sin tratamiento): CSC={S_init}, D={D_init}")
print(f"Tasas medias: <d_D>={dd_mean}, <d_S>={ds_mean}")
print(f"Tasas ON: dD_on={dd_on:.4f}, dS_on={ds_on:.4f}")

# ------------------------------------------
# FUNCIONES GILLESPIE PARA TRAYECTORIAS
# ------------------------------------------
@njit
def run_gillespie_traj_fijo(N, S_init, D_init, R_init, p, bs, bd, ds, dd, t_max):
    S1 = S_init
    D1 = D_init
    R1 = R_init
    t = 0.0
    
    max_steps = 1000000
    times = np.zeros(max_steps)
    S_arr = np.zeros(max_steps)
    D_arr = np.zeros(max_steps)
    
    times[0] = t
    S_arr[0] = S1
    D_arr[0] = D1
    step = 1
    
    while t < t_max and step < max_steps:
        if S1 == 0 and D1 == 0:
            times[step] = t_max
            S_arr[step] = S1
            D_arr[step] = D1
            step += 1
            break
            
        a1 = p * bs * ((S1 * R1) / N)
        a2 = (1 - p) * bs * ((S1 * R1) / N)
        a3 = ds * S1
        a4 = bd * ((D1 * R1) / N)
        a5 = dd * D1
        a0 = a1 + a2 + a3 + a4 + a5
        
        if a0 <= 0:
            times[step] = t_max
            S_arr[step] = S1
            D_arr[step] = D1
            step += 1
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
        times[step] = t
        S_arr[step] = S1
        D_arr[step] = D1
        step += 1
        
    return times[:step], S_arr[:step], D_arr[:step]

@njit
def run_gillespie_traj_ciclo(N, S_init, D_init, R_init, p, bs, bd, ds_on, dd_on, ds_off, dd_off, T_on, T_off, t_max):
    S1 = S_init
    D1 = D_init
    R1 = R_init
    t = 0.0
    T_ciclo = T_on + T_off
    
    max_steps = 1000000
    times = np.zeros(max_steps)
    S_arr = np.zeros(max_steps)
    D_arr = np.zeros(max_steps)
    
    times[0] = t
    S_arr[0] = S1
    D_arr[0] = D1
    step = 1
    
    while t < t_max and step < max_steps:
        if S1 == 0 and D1 == 0:
            times[step] = t_max
            S_arr[step] = S1
            D_arr[step] = D1
            step += 1
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
            times[step] = t_max
            S_arr[step] = S1
            D_arr[step] = D1
            step += 1
            break
            
        r1 = np.random.rand()
        r2 = np.random.rand()
        tau = -np.log(r1) / a0
        
        if tau > time_to_next_phase:
            t += time_to_next_phase
            times[step] = t
            S_arr[step] = S1
            D_arr[step] = D1
            step += 1
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
        times[step] = t
        S_arr[step] = S1
        D_arr[step] = D1
        step += 1
        
    return times[:step], S_arr[:step], D_arr[:step]

# ------------------------------------------
# SIMULACIÓN Y GENERACIÓN DE LA GRÁFICA
# ------------------------------------------
n_trayectorias = 5
t_max_sim = 1000

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey='row')

cmap = plt.get_cmap('tab10')

# Simulación y Gráficas: Tratamiento Continuo
for i in range(n_trayectorias):
    color = cmap(i % 10)
    t_fijo, S_fijo, D_fijo = run_gillespie_traj_fijo(N, S_init, D_init, R_init, p, bs, bd, ds_mean, dd_mean, t_max_sim)
    
    estado_S_fijo = "Extinguida" if S_fijo[-1] == 0 else "Sobrevive"
    estado_D_fijo = "Extinguida" if D_fijo[-1] == 0 else "Sobrevive"
    
    axes[0, 0].step(t_fijo, S_fijo, where='post', color=color, alpha=0.6, linewidth=1.5, label=f'Trayectoria {i+1} ({estado_S_fijo})')

    axes[1, 0].step(t_fijo, D_fijo, where='post', color=color, alpha=0.6, linewidth=1.5, label=f'Trayectoria {i+1} ({estado_D_fijo})')

# Simulación y Gráficas: Tratamiento Cíclico
for i in range(n_trayectorias):
    color = cmap(i % 10)
    t_ciclo, S_ciclo, D_ciclo = run_gillespie_traj_ciclo(N, S_init, D_init, R_init, p, bs, bd, ds_on, dd_on, dS_off, dD_off, T_on, T_off, t_max_sim)
    
    estado_S_ciclo = "Extinguida" if S_ciclo[-1] == 0 else "Sobrevive"
    estado_D_ciclo = "Extinguida" if D_ciclo[-1] == 0 else "Sobrevive"
    
    # Gráfica S Cíclico
    axes[0, 1].step(t_ciclo, S_ciclo, where='post', color=color, alpha=0.6, linewidth=1.5, label=f'Trayectoria {i+1} ({estado_S_ciclo})')
    # Gráfica D Cíclico
    axes[1, 1].step(t_ciclo, D_ciclo, where='post', color=color, alpha=0.6, linewidth=1.5, label=f'Trayectoria {i+1} ({estado_D_ciclo})')

axes[0, 0].set_title('CSC - Tratamiento Continuo')
axes[0, 0].set_ylabel('Población de CSC')
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)

axes[0, 1].set_title('CSC (S) - Tratamiento Cíclico (1 / 10)')
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)

axes[1, 0].set_title('Células Diferenciadas (D) - Tratamiento Continuo')
axes[1, 0].set_xlabel('Tiempo')
axes[1, 0].set_ylabel('Población de D')
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)

axes[1, 1].set_title('Células Diferenciadas (D) - Tratamiento Cíclico (1 / 10)')
axes[1, 1].set_xlabel('Tiempo')
axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)

for ax in [axes[0, 1], axes[1, 1]]:
    for i in range(int(t_max_sim / T_ciclo) + 1):
        ax.axvspan(i * T_ciclo, i * T_ciclo + T_on, color='gray', alpha=0.2)

for ax in axes.flat:
    ax.legend(loc='best')

plt.tight_layout()
plt.show()