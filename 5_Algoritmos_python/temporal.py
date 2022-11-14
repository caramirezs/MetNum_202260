import numpy as np
dx = lambda t: 2-0.2*t
dt= 0.0001
t_ini, x_ini = 0.0, 5.0
x_fin = 5
n = x_fin/dt
print('pasos:', n)
lista_t =np.linspace(t_ini, 5, int(n)+1)
for t in lista_t:
    print(f't={t:.3f}, x={x_ini:.3f}')
    x = dx(t)*(dt) + x_ini
    x_ini = x
