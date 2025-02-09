import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import TransferFunction, bode, StateSpace
from parametros import *
from qrad import Q_rad

# Definição dos parâmetros do circuito
R12, R23, R31 = 1e3, 1e3, 1e3  # Resistências em ohms
C1, C2, C3 = 1e-6, 1e-6, 1e-6  # Capacitâncias em farads

TNH3 = -10 + 273

Tinf_0 = 10 + 273
Tinf = 10 + 273
Tsolo = Tinf

n = 1000000

def q_i_(T3, TNH3, m_nh3):
    C = (m_ar*Cp_ar)/(m_nh3*Cp_nh3)
    eps = (1-np.e**(-(1+C)*NUT))/(1+C)
    return m_ar*Cp_ar*eps*(T3-TNH3)*1000

# Definição do sistema de equações diferenciais
def circuit_dynamics(t, y):
    T1, T2, T3 = y  # Tensões nos capacitores
    
    Tinf = (10 + 273) + 3*np.sin(np.pi*t/86400) + 10*np.sin(2*np.pi*(t//86400)/365)
    
    q_solo = (T1 - Tsolo)/R_solo
    q_teto = (Tinf - T3)/R_teto(T3,Tinf)
    q_rad  = Q_rad(np.floor(t))
    q_i    = q_i_(T3,TNH3,0.02-abs(0.01*np.sin(np.pi*t/86400)))
    q_12   = (T1-T2)/R_e(T1,T2)
    q_23   = (T2-T3)/R_e(T2,T3)
    q_31   = (T3-T1)/R_31
    
    q_C1 = -q_solo-q_12+q_31-q_i
    q_C2 = -q_23+q_12
    q_C3 = q_rad+q_teto-q_31+q_23
    
    dT1_dt = (1/n) * q_C1 / C_1(T1,Tinf)
    dT2_dt = (1/n) * q_C2 / C_2(T2,Tinf)
    dT3_dt = (1/n) * q_C3 / C_3(T3,Tinf)
    
    return [dT1_dt, dT2_dt, dT3_dt]

# Condições iniciais
T1_0 = 14 + 273
T2_0 = 20 + 273
T3_0 = 26 + 273

# Intervalo de tempo e resolução
t_span = (0, 365*86400)
t_eval = np.linspace(*t_span, n)

# Solução pelo método de Runge-Kutta 4/5
sol = solve_ivp(circuit_dynamics, t_span, [T1_0, T2_0, T3_0], method='RK45', t_eval=t_eval)

print(min(sol.y[0]),max(sol.y[0]))

# Plotagem dos resultados
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='T1 (V)')
plt.plot(sol.t, sol.y[1], label='T2 (V)')
plt.plot(sol.t, sol.y[2], label='T3 (V)')
plt.xlabel('Tempo (s)')
plt.ylabel('Tensão (V)')
plt.legend()
plt.grid()
plt.show()

# Representação no espaço de estados
A = np.array([[-1/(R12*C1), 1/(R12*C1), 0],
              [-1/(R12*C2), (-1/R23 - 1/R12)/C2, 1/(R23*C2)],
              [0, -1/(R23*C3), (-1/R31 - 1/R23)/C3]])
B = np.array([[Tsolo/(R12*C1)], [0], [Tinf/(R31*C3)]])
C = np.eye(3)  # Observamos todas as variáveis de estado
D = np.zeros((3, 1))

sys = StateSpace(A, B, C, D)

# Função de Transferência
num, den = TransferFunction(*sys.to_tf()[0]).to_tf()
H_s = TransferFunction(num, den)

# Diagrama de Bode
w, mag, phase = bode(H_s)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title("Diagrama de Bode")
plt.ylabel("Magnitude (dB)")
plt.grid()

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.ylabel("Fase (graus)")
plt.xlabel("Frequência (rad/s)")
plt.grid()
plt.show()