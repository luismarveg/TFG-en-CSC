import numpy as np
import matplotlib.pyplot as plt

#Tasas de nacimiento/muerte
p = 0.5
bs = 0.07
bd = 0.43
ds = 0.01
dd = 0.30

N2=10000
S2=100
D2=0
R2=N2-S2-D2

xS0 = S2 / N2
xD0 = D2 / N2

t_max =10000
dt = 0.01

historial_t2 = [0]
historial_xS = [xS0]
historial_xD = [xD0]
historial_xR = [1.0-xS0-xD0]

h=int(t_max/dt)
rng=np.random.default_rng()

xS = xS0
xD = xD0
t2 = 0

for t in range(h):

    xR = 1.0 - (xS + xD)

    qS = (p * bs * xR - ds) * xS
    qD = (1 - p) * bs * xR * xS + (bd * xR - dd) * xD

    gS = np.sqrt(np.abs((p * bs * xR + ds) * xS))
    gD = np.sqrt(np.abs((1 - p) * bs * xR * xS + (bd * xR + dd) * xD))

    eta_S, eta_D = rng.normal(size=2)

    xS = xS + qS * dt + (1 / np.sqrt(N2)) * gS * np.sqrt(dt) * eta_S
    xD = xD + qD * dt + (1 / np.sqrt(N2)) * gD * np.sqrt(dt) * eta_D
    t2=t2+dt

#Barreras de extinción
    if xS < 1/N2:
        xS = 0.0
    if xD < 1/N2:
        xD = 0.0

#barrera reflectante
    if (xS + xD) > 1.0:
        factor = 1.0 / (xS + xD)
        xS = xS*factor
        xD = xD*factor

    historial_xS.append(xS)
    historial_xD.append(xD)
    historial_xR.append(xR)
    historial_t2.append(t2)
    
print(xS,xD,t2)

if p*bs<ds:
    print('Se incumple la condición de viabilidad')
elif p*bs*dd<bd*ds:
    print('Se incumple la condición de dominancia')
else: 
    # 1. Calcular la base del denominador común 
    denominador = (1 - p) * bs * ds + p * bs * dd - bd * ds

    # 2. Calcular los puntos fijos exactos (xS* y xD*)
    xS_star = ((p * bs * dd - bd * ds) * (p * bs - ds)) / (p * bs * denominador)
    xD_star = ((1 - p) * ds * (p * bs - ds)) / (p * denominador)

    print(f"Punto fijo CSC (xS*): {xS_star:.4g}")
    print(f"Punto fijo Diferenciadas (xD*): {xD_star:.4g}")

    # 3. Dibujar las gráficas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Gráfica para CSC (xS) ---
    axes[0].plot(historial_t2, historial_xS, label='Langevin')
    axes[0].axhline(y=xS_star, color='red', linestyle='--', label='Equilibrio') # Línea de punto fijo
    axes[0].set_xlabel("Tiempo")
    axes[0].set_ylabel("Fracción de CSC")
    axes[0].set_title("CSC (Langevin + Punto de equilibrio)")
    axes[0].legend() 

    # --- Gráfica para Células diferenciadas (xD) ---
    axes[1].plot(historial_t2, historial_xD, label='Langevin')
    axes[1].axhline(y=xD_star, color='red', linestyle='--', label='Equilibrio') # Línea de punto fijo
    axes[1].set_xlabel("Tiempo")
    axes[1].set_ylabel("Fracción de Células diferenciadas")
    axes[1].set_title("Células diferenciadas (Langevin + Punto de equilibrio)")
    axes[1].legend() 

    plt.tight_layout()
    plt.show()