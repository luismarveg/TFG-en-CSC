import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots(figsize=(8, 9))
plt.subplots_adjust(bottom=0.35) 

def actualizar_grafico(val):
    ax.clear()
    
    p = slider_p.val
    bS = slider_bS.val
    dS = slider_dS.val
    bD = slider_bD.val
    dD = slider_dD.val
    
    xS_vals = np.linspace(0, 1, 22)
    xD_vals = np.linspace(0, 1, 22)
    X_S, X_D = np.meshgrid(xS_vals, xD_vals)
    R = 1 - X_S - X_D
    
    qS = p * bS * X_S * R - dS * X_S
    qD = (1 - p) * bS * X_S * R + bD * X_D * R - dD * X_D
    
    norma = np.sqrt(qS**2 + qD**2)
    norma[norma == 0] = 1 
    qS_norm = qS / norma
    qD_norm = qD / norma

    mascara_imposible = (X_S + X_D) > 1
    qS_norm[mascara_imposible] = np.nan
    qD_norm[mascara_imposible] = np.nan

    ax.quiver(X_S, X_D, qS_norm, qD_norm, color='gray', alpha=0.6)
    ax.fill_between([0, 1], [1, 0], [1, 1], color='red', alpha=0.1, label='Región Imposible ($x_S+x_D>1$)')
    
    xS_line = np.linspace(0.001, 1, 100) 
    if p * bS != 0:
        xD_null_S = 1 - xS_line - (dS / (p * bS))
        ax.plot(xS_line, xD_null_S, color='blue', linewidth=2, label='Nulclina S ($q_S=0$)')
        
    X_S_dense, X_D_dense = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    R_dense = 1 - X_S_dense - X_D_dense
    qD_dense = (1 - p) * bS * X_S_dense * R_dense + bD * X_D_dense * R_dense - dD * X_D_dense
    ax.contour(X_S_dense, X_D_dense, qD_dense, levels=[0], colors='green', linewidths=2)
    
    denominador_comun = (1 - p) * bS * dS + p * bS * dD - bD * dS
    
    if p * bS != 0 and p != 0 and denominador_comun != 0:
        xS_star = ((p * bS * dD - bD * dS) * (p * bS - dS)) / (p * bS * denominador_comun)
        xD_star = ((1 - p) * dS * (p * bS - dS)) / (p * denominador_comun)
        
        if xS_star >= 0 and xD_star >= 0 and (xS_star + xD_star <= 1):
            etiqueta_punto = f'Punto Fijo ({xS_star:.4f}, {xD_star:.4f})'
            ax.plot(xS_star, xD_star, 'ro', markersize=10, markeredgecolor='black', label=etiqueta_punto, zorder=5)
        else:
            ax.set_title('Extinción o Régimen Inválido', fontsize=12, color='red', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Fracción de Células Madre ($x_S$)', fontsize=12)
    ax.set_ylabel('Fracción Diferenciadas ($x_D$)', fontsize=12)
    
    if not ax.get_title():
        ax.set_title(f'Espacio de Fases (p={p:.2f}, bS={bS:.2f}, dS={dS:.2f}, bD={bD:.2f}, dD={dD:.2f})', fontsize=14)
    
    ax.plot([], [], color='green', linewidth=2, label='Nulclina D ($q_D=0$)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.canvas.draw_idle()

ax_p  = plt.axes([0.15, 0.25, 0.75, 0.03])
ax_bS = plt.axes([0.15, 0.20, 0.75, 0.03])
ax_dS = plt.axes([0.15, 0.15, 0.75, 0.03])
ax_bD = plt.axes([0.15, 0.10, 0.75, 0.03])
ax_dD = plt.axes([0.15, 0.05, 0.75, 0.03])

slider_p  = Slider(ax_p,  'p',   0.01, 1.0, valinit=0.25, valstep=0.01)
slider_bS = Slider(ax_bS, 'b_S', 0.01,  5.0, valinit=0.15, valstep=0.01)
slider_dS = Slider(ax_dS, 'd_S', 0.0,  1.0, valinit=0.01, valstep=0.001)
slider_bD = Slider(ax_bD, 'b_D', 0.01,  5.0, valinit=0.30, valstep=0.01)
slider_dD = Slider(ax_dD, 'd_D', 0.0,  1.0, valinit=0.10, valstep=0.01)

slider_p.on_changed(actualizar_grafico)
slider_bS.on_changed(actualizar_grafico)
slider_dS.on_changed(actualizar_grafico)
slider_bD.on_changed(actualizar_grafico)
slider_dD.on_changed(actualizar_grafico)

actualizar_grafico(None)
plt.show()