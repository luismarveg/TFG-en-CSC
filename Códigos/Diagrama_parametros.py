import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 14
})

p = 0.41
bs = 0.07
bd = 0.43

dd_vals = np.linspace(0.0, 1.0, 600)
ds_vals = np.linspace(0.0, 0.1, 600)
DD, DS = np.meshgrid(dd_vals, ds_vals)

denominador_base = (1 - p) * bs * DS + p * bs * DD - bd * DS
with np.errstate(divide='ignore', invalid='ignore'):
    xS_coex = ((p * bs * DD - bd * DS) * (p * bs - DS)) / (p * bs * denominador_base)
    xD_coex = ((1 - p) * DS * (p * bs - DS)) / (p * denominador_base)

mask_coex = (DS < p * bs) & (DS < (p * bs / bd) * DD)
mask_D_only = (~mask_coex) & (DD < bd)
mask_total_ext = (~mask_coex) & (DD >= bd)

xS_full = np.zeros_like(DD)
xD_full = np.zeros_like(DD)

xS_full[mask_coex] = xS_coex[mask_coex]
xD_full[mask_coex] = xD_coex[mask_coex]

xS_full[mask_D_only] = 0.0
xD_full[mask_D_only] = 1.0 - (DD[mask_D_only] / bd)

xS_full[mask_total_ext] = 0.0
xD_full[mask_total_ext] = 0.0

x_tot_full = xS_full + xD_full

Z = np.zeros_like(DD)
Z[mask_coex] = 1
Z[mask_D_only] = 2
Z[mask_total_ext] = 3

# ==================== GRÁFICA ====================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, constrained_layout=True)
ax1, ax2, ax3 = axes

im1 = ax1.pcolormesh(DD, DS, xS_full, cmap='inferno', vmin=0, vmax=1, shading='auto')
ax1.set_title('Fracción de CSC ($x_S^*$)')

im2 = ax2.pcolormesh(DD, DS, xD_full, cmap='inferno', vmin=0, vmax=1, shading='auto')
ax2.set_title('Fracción Diferenciadas ($x_D^*$)')

im3 = ax3.pcolormesh(DD, DS, x_tot_full, cmap='inferno', vmin=0, vmax=1, shading='auto')
ax3.set_title('Tamaño Total ($x^*=x_S^*+x_D^*$)')

fig.supxlabel('Tasa muerte dif. ($d_D$)')
fig.supylabel('Tasa muerte CSC ($d_S$)')

cbar = fig.colorbar(im3, ax=axes.ravel().tolist())

fig2, ax_cat = plt.subplots(figsize=(7, 6))
cmap_zones = ListedColormap(['#d9534f', '#5bc0de', '#e6e6e6']) 
im4 = ax_cat.pcolormesh(DD, DS, Z, cmap=cmap_zones, shading='auto')

all_axes = [ax1, ax2, ax3, ax_cat]
for ax in all_axes:

    ax.plot([0, bd], [0, p*bs], 'w--', linewidth=2.5, label='Extinción CSC')

    ax.plot([bd, 1.0], [p*bs, p*bs], 'w:', linewidth=2.5, label='Límite Sup. CSC')
    
    ax.plot([bd, bd], [p*bs, 0.1], 'w-.', linewidth=2.5, label='Límite Sup. D')
    
    if ax == ax_cat:
        ax.set_xlabel('Tasa muerte dif. ($d_D$)')
        ax.set_ylabel('Tasa muerte CSC ($d_S$)')
        
    ax.set_xlim(0, 0.8) 
    ax.set_ylim(0, 0.08)


leyenda = ax_cat.legend(loc='upper right', fontsize=14)
for linea in leyenda.get_lines():
    linea.set_color('black')
ax_cat.text(0.45, 0.014, 'Coexistencia', ha='center', va='center', fontsize=16, color='black', fontweight='bold')
ax_cat.text(0.22, 0.05, '\nSolo Diferenciadas', ha='center', va='center', fontsize=16, color='black', fontweight='bold')
ax_cat.text(0.65, 0.05, 'Curación\n(Extinción Total)', ha='center', va='center', fontsize=16, color='black', fontweight='bold')

# ---Añadir las ecuaciones matemáticas de las fronteras ---

ax_cat.text(0.18, 0.016, r'$d_S = \frac{p \cdot b_S}{b_D} d_D$', color='black', fontsize=18, rotation=20, ha='center')

ax_cat.text(0.65, 0.032, r'$d_S = p \cdot b_S$', color='black', fontsize=18, ha='center')

ax_cat.text(0.44, 0.07, r'$d_D = b_D$', color='black', fontsize=18, rotation=90, va='top')

fig2.tight_layout()
plt.show()