import numpy as np
import matplotlib.pyplot as plt

#Tasas de nacimiento/muerte
p = 0.5
bs = 0.07
bd = 0.43
ds = 0.01
dd = 0.30

N1=10000
S1=100
D1=0
R1=N1-S1-D1

x=np.array([S1,D1,R1]) #Vector de estado
t=0
t_max=10000

#Vectores de cambio de estado 
v1=np.array([1, 0,-1]) # S+R-->S+S
v2=np.array([0, 1,-1]) # S+R-->S+D
v3=np.array([-1, 0,1]) # S-->R
v4=np.array([0, 1,-1]) # D+R-->D+D
v5=np.array([0, -1,1]) # D-->R

V = [v1, v2, v3, v4, v5]

historial_S1=[S1]
historial_D1=[D1]
historial_R1=[R1]
historial_t1=[t]

while t<t_max:

    a1=p*bs*((S1*R1)/N1)
    a2=(1-p)*bs*((S1*R1)/N1)
    a3=ds*S1
    a4=bd*((D1*R1)/N1)
    a5=dd*D1
    a0=a1+a2+a3+a4+a5

    if a0 == 0:
        break

    r1=np.random.rand()
    r2=np.random.rand()

    tau=-np.log(r1) / a0
    u=r2*a0

    if u < a1:
        j=0
    elif u < a1+a2:
        j=1    
    elif u < a1+a2+a3:
        j=2
    elif u < a1+a2+a3+a4:
        j=3
    else:
        j=4

    x = x + V[j]
    S1, D1, R1 = x
    t=t+tau

    historial_S1.append(S1)
    historial_D1.append(D1)
    historial_R1.append(R1)
    historial_t1.append(t)
print (x)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))  

axes[0].plot(historial_t1, historial_S1)
axes[0].set_xlabel("Tiempo")
axes[0].set_ylabel("CSC")
axes[0].set_title("CSC (Gillespie)")

axes[1].plot(historial_t1, historial_D1)
axes[1].set_xlabel("Tiempo")
axes[1].set_ylabel("Células diferenciadas")
axes[1].set_title("Células diferenciadas (Gillespie)")

axes[2].plot(historial_t1, historial_R1)
axes[2].set_xlabel("Tiempo")
axes[2].set_ylabel("Recursos libres")
axes[2].set_title("Recursos libres (Gillespie)")

plt.tight_layout()
plt.show()
