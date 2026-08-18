import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

@njit
def simular_ssa(N, beta, mu, I0, t_max, dt):
    pasos = int(t_max / dt)
    historial_x = np.zeros(pasos)
    
    I = I0
    S = N - I
    t = 0.0
    idx = 0
    
    while t < t_max:
        a1 = beta * S * I / N
        a2 = mu * I
        a0 = a1 + a2
        
        if a0 == 0:
            while idx < pasos:
                historial_x[idx] = I / N
                idx += 1
            break
            
        r1 = np.random.rand()
        r2 = np.random.rand()
        if r1 < 1e-15:
            r1 = 1e-15
            
        tau = -np.log(r1) / a0
        
        while idx < pasos and idx * dt < t + tau:
            historial_x[idx] = I / N
            idx += 1
            
        if idx >= pasos:
            break
            
        if r2 * a0 < a1:
            I += 1
            S -= 1
        else:
            I -= 1
            S += 1
            
        t += tau
        
    return historial_x

@njit
def simular_langevin(N, beta, mu, x0, t_max, dt):
    pasos = int(t_max / dt)
    historial_x = np.zeros(pasos)
    x = x0
    
    sqrt_N = np.sqrt(N)
    sqrt_dt = np.sqrt(dt)
    
    for i in range(pasos):
        drift = beta * x * (1.0 - x) - mu * x
        variance = beta * x * (1.0 - x) + mu * x
        if variance < 0:
            variance = 0.0
            
        eta = np.random.randn()
        
        x = x + drift * dt + (1.0 / sqrt_N) * np.sqrt(variance) * sqrt_dt * eta
        
        if x < 1.0 / N:
            x = 0.0
        elif x > 1.0:
            x = 1.0
            
        historial_x[i] = x
        
        if x == 0.0:
            for j in range(i+1, pasos):
                historial_x[j] = 0.0
            break
            
    return historial_x

def simular_trayectoria_larga(metodo, N, beta, mu, I0, t_max, dt):

    intentos = 0
    while True:
        intentos += 1
        if metodo == 'ssa':
            hist = simular_ssa(N, beta, mu, I0, t_max, dt)
        else:
            hist = simular_langevin(N, beta, mu, I0/N, t_max, dt)
            
        fraccion_viva = np.mean(hist > 0)
        
        if fraccion_viva > 0.5:
            if intentos > 1:
                print(f"  [{metodo.upper()} N={N}] Requirió {intentos} intento(s) (trayectorias previas se extinguieron muy rápido).")
            
            transitorio = int(len(hist) * 0.2)
            datos_utiles = hist[transitorio:]
            return datos_utiles[datos_utiles > 0]
            
        if intentos > 500:
            return hist[hist > 0]

if __name__ == '__main__':
    beta = 1.5
    mu = 1.0
    t_max_bajo = 5000  
    t_max_alto = 500 
    dt = 0.01
    
    equilibrio = 1 - (mu / beta)
    print(f"Parámetros: beta={beta}, mu={mu}, equilibrio={equilibrio:.3f}\n")
    
    t0 = time.time()
    
    # ------------------
    # Caso 1: N Bajo
    # ------------------
    N_bajo = 50
    I0_bajo = 16
    print(f"Simulando Caso 1: N = {N_bajo} (t_max = {t_max_bajo})...")
    
    ssa_bajo = simular_trayectoria_larga('ssa', N_bajo, beta, mu, I0_bajo, t_max_bajo, dt)
    lang_bajo = simular_trayectoria_larga('langevin', N_bajo, beta, mu, I0_bajo, t_max_bajo, dt)
    
    # ------------------
    # Caso 2: N Alto
    # ------------------
    N_alto = 10000
    I0_alto = 3333
    print(f"Simulando Caso 2: N = {N_alto} (t_max = {t_max_alto})...")
    
    ssa_alto = simular_trayectoria_larga('ssa', N_alto, beta, mu, I0_alto, t_max_alto, dt)
    lang_alto = simular_trayectoria_larga('langevin', N_alto, beta, mu, I0_alto, t_max_alto, dt)
    
    # ------------------
    # Gráficas
    # ------------------
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16, 'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    min_val = min(np.min(ssa_bajo), np.min(lang_bajo))
    max_val = max(np.max(ssa_bajo), np.max(lang_bajo))
    min_bin = np.floor(min_val * N_bajo) / N_bajo
    max_bin = np.ceil(max_val * N_bajo) / N_bajo
    bins_bajo = np.arange(min_bin - 0.5/N_bajo, max_bin + 1.5/N_bajo, 1.0/N_bajo)
    axes[0].hist(ssa_bajo, density=True, bins=bins_bajo, alpha=0.6, 
                label=f'SSA', color='blue', edgecolor='black')
    axes[0].hist(lang_bajo, density=True, bins=bins_bajo, alpha=0.6, 
                label=f'Langevin', color='orange', edgecolor='black')
    axes[0].axvline(x=equilibrio, color='red', linestyle='--', linewidth=2,
                   label=f'Equilibrio={equilibrio:.3f}')
    axes[0].set_title(f"N = {N_bajo}")
    axes[0].set_xlabel("I / N (fracción infectados)")
    axes[0].set_ylabel("Densidad de probabilidad")
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(ssa_alto, density=True, bins=50, alpha=0.6, 
                label=f'SSA', color='blue', edgecolor='black')
    axes[1].hist(lang_alto, density=True, bins=50, alpha=0.6, 
                label=f'Langevin', color='orange', edgecolor='black')
    axes[1].axvline(x=equilibrio, color='red', linestyle='--', linewidth=2,
                   label=f'Equilibrio={equilibrio:.3f}')
    axes[1].set_title(f"N = {N_alto}")
    axes[1].set_xlabel("I / N (fracción infectados)")
    axes[1].set_ylabel("Densidad de probabilidad")
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('HistogramaV3.png', dpi=300)
    plt.show()
