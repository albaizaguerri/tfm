import numpy as np
import matplotlib.pyplot as plt

# 1. Leemos el fichero generado por la simulación (saltando la fila de cabecera)
datos = np.loadtxt("resultados_entropia.csv", delimiter=",", skiprows=1)

tiempos = datos[:, 0]
historia_vn = datos[:, 1]
historia_s2 = datos[:, 2]

# 2. Ploteamos los resultados
plt.figure(figsize=(8, 5))
plt.plot(tiempos, historia_vn, 'o-', label='Von Neumann ($S_{vN}$)')
plt.plot(tiempos, historia_s2, 's--', label='Rényi-2 ($S_2$)')

plt.xlabel('Tiempo / Capa ($t$)')
plt.ylabel('Entropía de entrelazamiento (bits)')
plt.title('Evolución del Entrelazamiento - Circuito Haar PBC')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()