import qutip as qt
import numpy as np

# Definición de funciones auxiliares 
def random_gate():
    #Genera una matriz unitaria aleatoria de Haar 4x4 para 2 qubits.
    U = qt.rand_unitary(4)
    U.dims = [[2, 2], [2, 2]]
    return U

def construir_capa_impar(L):
    #Construye la capa de puertas para los tiempos impares (t = 1, 3, 5...).
    puertas = [random_gate() for _ in range(L // 2)]
    return qt.tensor(*puertas)


def construir_capa_par_periodica(L):
    #Capa t = 2, 4, 6... actúa sobre (1,2), (3,4)... y envuelve (L-1, 0).

    # Empezamos con el operador identidad global del sistema de L qubits
    capa = qt.tensor([qt.qeye(2)] * L)
    
    # Recorremos los índices impares: 1, 3, ..., L-1
    for i in range(1, L, 2):
        # El operador módulo (% L) hace que si i = L-1 (ej. 5), j sea 0
        j = (i + 1) % L  
        
        U = random_gate()
        # Expandimos la puerta 4x4 al espacio de L qubits apuntando a [i, j]
        U_expandida = qt.expand_operator(U, dims=[2] * L, targets=[i, j])

        # Multiplicamos para ir acumulando todas las puertas de la capa
        capa = U_expandida * capa
        
    return capa

# Función main para ejecutar el programa
def main():
    # Parámetros
    L = 6   # Número de qubits
    T = 3*L  # Número de capas de evolución
    N = 100 # Número de trayectorias (a promediar)

    # Creamos el estado inicial |\psi_0> = |000000>
    qubit_0 = qt.basis(2, 0) 
    psi = qt.tensor([qubit_0] * L)  #función incial psi_0

    # Definimos los qubits del subsistema A (la primera mitad de la cadena)
    subsistema_A = list(range(L // 2))

    # Listas para guardar la evolución temporal de las entropías (4 espacios)
    historia_vn = [0.0]
    historia_s2 = [0.0]

    # Evolucionamos el estado a través de varias capas de puertas
    for t in range(1, T + 1):
            if t % 2 != 0:
                capa = construir_capa_impar(L)
            else:
                capa = construir_capa_par_periodica(L)
            
            # Evolucionamos el estado actual aplicándole la nueva capa
            psi = capa * psi

            """
            #Comprobación: norma del estado después de aplicar la capa
            print(f"Capa t={t:2d} aplicada | Norma: {psi.norm():.4f}")
            """

            # 1. Matriz densidad global del sistema 
            rho_total = qt.ket2dm(psi)

            # 2. Matriz densidad reducida del subsistema A trazando B
            rho_A = rho_total.ptrace(subsistema_A)

            """
            # Comprobación: pureza Tr(rho_A^2)
            print(f"t={t:2d} | Pureza de A: {pureza_A:.4f}")
            """
            # Cálculo de entropías (en base 2)
            s_vn = qt.entropy_vn(rho_A, base=2)
            pureza_A = np.real((rho_A * rho_A).tr())
            s_2 = -np.log2(pureza_A)  # S_2 = -log2(Tr(rho_A^2))
            """
            # Comprobación: pureza Tr(rho_A^2)
            print(f"t={t:2d} | Pureza de A: {pureza_A:.4f}")
            """

            historia_vn.append(s_vn)
            historia_s2.append(s_2)
            #print(f"t={t:2d} | Von Neumann: {s_vn:.4f} | Rényi-2: {s_2:.4f}")
            
    tiempos = np.arange(0, T + 1)
    datos = np.column_stack((tiempos, historia_vn, historia_s2))
    np.savetxt("resultados_entropia.csv", datos, delimiter=",", header="t,S_vN,S_2", comments="")
    print("Datos guardados correctamente en resultados_entropia.csv")        
            

# Punto de entrada del programa 
if __name__ == "__main__":
    main()