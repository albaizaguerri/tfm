import numpy as np
import matplotlib.pyplot as plt

# 1. Leer el archivo generado por la simulación
datos = np.loadtxt("resultados_entropia_N.csv", delimiter=",", skiprows=1)

# Extraer cada columna
tiempos = datos[:, 0]
media_vn = datos[:, 1]
error_vn = datos[:, 2]
media_s2 = datos[:, 3]
error_s2 = datos[:, 4]

# 2. Dibujar la gráfica
plt.figure(figsize=(8, 5))

plt.errorbar(tiempos, media_vn, yerr=error_vn, fmt='o-', capsize=3, markersize=5, label='Von Neumann ($\\overline{S_{vN}}$)')
plt.errorbar(tiempos, media_s2, yerr=error_s2, fmt='s--', capsize=3, markersize=5, label='Rényi-2 ($\\overline{S_2}$)')

plt.xlabel('Tiempo / Capa ($t$)')
plt.ylabel('Entropía de entrelazamiento promedio')
plt.title('Evolución Promediada - Circuito Haar')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()