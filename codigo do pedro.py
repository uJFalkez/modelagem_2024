import numpy as np
import scipy.signal as signal
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Parâmetros do sistema (defina valores reais conforme necessário)
C1, C2, C3 = 1.0, 1.0, 1.0  # Capacitâncias térmicas
R12, R23, Rrad, R0, Rd = 1.0, 1.0, 1.0, 1.0, 1.0  # Resistências térmicas

# Sistema de equações diferenciais
def thermal_system(t, x):
    Tsol, T1, T2, T3 = x

    dTsol_dt = (T1 - Tsol) / R0 + (Tsol - T1) / Rd
    dT1_dt = ((Tsol - T1) / Rd + (T2 - T1) / R12) / C1
    dT2_dt = ((T1 - T2) / R12 + (T3 - T2) / R23) / C2
    dT3_dt = ((T2 - T3) / R23 + (Tsol - T3) / Rrad) / C3

    return [dTsol_dt, dT1_dt, dT2_dt, dT3_dt]

# Condições iniciais
x0 = [20.0, 20.0, 20.0, 20.0]  # Temperaturas iniciais

time_span = (0, 100)  # Tempo de simulação
time_eval = np.linspace(time_span[0], time_span[1], 1000)  # Pontos de tempo

# Resolver o sistema de equações diferenciais
sol = integrate.solve_ivp(thermal_system, time_span, x0, t_eval=time_eval)

# Gráficos
tl = ["Tsol", "T1", "T2", "T3"]
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(sol.t, sol.y[i], label=tl[i])
plt.xlabel("Tempo (s)")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.title("Evolução das Temperaturas no Sistema Térmico")
plt.grid()
plt.show()

# Matriz de espaço de estados
A = np.array([[-(1/R0 + 1/Rd), 1/R0, 0, 0],
              [1/(Rd * C1), -(1/(Rd * C1) + 1/(R12 * C1)), 1/(R12 * C1), 0],
              [0, 1/(R12 * C2), -(1/(R12 * C2) + 1/(R23 * C2)), 1/(R23 * C2)],
              [1/(Rrad * C3), 0, 1/(R23 * C3), -(1/(R23 * C3) + 1/(Rrad * C3))]])

B = np.array([[1/Rd], [0], [0], [0]])
C = np.eye(4)  # Saída é o estado completo
D = np.zeros((4, 1))

# Obter funções de transferência
num, den = signal.ss2tf(A, B, C, D)

# Imprimir funções de transferência
for i in range(4):
    num_str = " + ".join(f"{coef} s^{len(num[i])-j-1}" for j, coef in enumerate(num[i]) if coef != 0)
    den_str = " + ".join(f"{coef} s^{len(den)-j-1}" for j, coef in enumerate(den) if coef != 0)
    print(f"TF {i+1}: ({num_str}) / ({den_str})")

# Plotar resposta ao degrau
plt.figure(figsize=(10, 6))
for i in range(4):
    system_tf = signal.TransferFunction(num[i], den)  # Criar sistema individual
    t, y = signal.step(system_tf)
    plt.plot(t, y, label=f"TF {i+1}")

plt.xlabel("Tempo (s)")
plt.ylabel("Resposta ao Degrau")
plt.legend()
plt.title("Respostas ao Degrau das Funções de Transferência")
plt.grid()
plt.show()

# Plotar diagramas de Bode
plt.figure(figsize=(10, 6))
for i in range(4):
    system_tf = signal.TransferFunction(num[i], den)  # Criar sistema individual
    w, mag, phase = signal.bode(system_tf)

    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag, label=f"TF {i+1}")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase, label=f"TF {i+1}")
    plt.xlabel("Frequência (rad/s)")
    plt.ylabel("Fase (graus)")
    plt.grid(True)

plt.subplot(2, 1, 1)
plt.legend()
plt.title("Diagrama de Bode - Magnitude")
plt.subplot(2, 1, 2)
plt.legend()
plt.title("Diagrama de Bode - Fase")
plt.show()