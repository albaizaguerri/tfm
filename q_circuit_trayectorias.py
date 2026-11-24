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

def construir_capa_par_abierta(L):
    #Construye la capa de puertas para los tiempos pares (t = 2, 4, 6...) con bordes abiertos.
    id_1q = qt.qeye(2)
    puertas_centro = [random_gate() for _ in range((L // 2) - 1)]
    return qt.tensor(id_1q, *puertas_centro, id_1q)

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
    psi_inicial = qt.tensor([qubit_0] * L)  #función incial psi_0

    # Definimos los qubits del subsistema A (la primera mitad de la cadena)
    subsistema_A = list(range(L // 2))

    # Matrices para guardar todas las trayectorias: filas = trayectoria n, columnas = tiempo t
    # (La columna t=0 ya queda automáticamente en 0.0)
    todas_vn = np.zeros((N, T + 1))
    todas_s2 = np.zeros((N, T + 1))

    for n in range(N):
        # Reiniciamos el estado al inicial en cada nueva trayectoria
        psi = psi_inicial.copy()

        # Evolucionamos el estado a través de varias capas de puertas
        for t in range(1, T + 1):
                if t % 2 != 0:
                    capa = construir_capa_impar(L)
                else:
                    capa = construir_capa_par_periodica(L)
                
                # Evolucionamos el estado actual aplicándole la nueva capa
                psi = capa * psi

                # 1. Matriz densidad global del sistema 
                rho_total = qt.ket2dm(psi)

                # 2. Matriz densidad reducida del subsistema A trazando B
                rho_A = rho_total.ptrace(subsistema_A)

                # Cálculo de entropías (en base 2)
                s_vn = qt.entropy_vn(rho_A, base=2)
                pureza_A = np.real((rho_A * rho_A).tr())
                s_2 = -np.log2(pureza_A)  # S_2 = -log2(Tr(rho_A^2))

                # Guardamos en la fila n, columna t
                todas_vn[n, t] = s_vn
                todas_s2[n, t] = s_2



    #   Promedio de las N trayec y guardamos csv
    media_vn = np.mean(todas_vn, axis=0)
    error_vn = np.std(todas_vn, axis=0) / np.sqrt(N)

    media_s2 = np.mean(todas_s2, axis=0)
    error_s2 = np.std(todas_s2, axis=0) / np.sqrt(N)

    tiempos = np.arange(0, T + 1)
    datos = np.column_stack((tiempos, media_vn, error_vn, media_s2, error_s2))
    np.savetxt(
        "resultados_entropia_N.csv", 
        datos, 
        delimiter=",", 
        header="t,media_vn,error_vn,media_s2,error_s2", 
        comments=""
    )
    print("Datos guardados en resultados_entropia_N.csv")

                
                
        
                
        



            



    

# Punto de entrada del programa 
if __name__ == "__main__":
    main()