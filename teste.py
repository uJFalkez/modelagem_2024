import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import TransferFunction, bode, StateSpace, ss2tf
from parametros import *
from qrad import Q_rad
    
TNH3_0  = -10 + 273
Tinf_0  =  10 + 273

n = 1000000

# Definição do sistema de equações diferenciais
def circuit_dynamics(t, y):
    T1, T2, T3, TNH3s = y
    
    TNH3 = TNH3_0 + 5*np.sin(np.pi*t/8640)
    
    Tinf = (10 + 273) + 3*np.sin(np.pi*t/86400) + 10*np.sin(2*np.pi*(t//86400)/365)
    
    q_solo = (T1 - Tinf)/R_solo
    q_teto = (Tinf - T3)/R_teto(T3,Tinf)
    q_rad  = Q_rad(np.floor(t))
    q_i    = m_ar*Cp_ar*eps*(T3-TNH3)*1000
    q_12   = (T1-T2)/R_e(T1,T2)
    q_23   = (T2-T3)/R_e(T2,T3)
    q_31   = (T3-T1)/R_31
    
    q_C1 = -q_solo-q_12+q_31-q_i
    q_C2 = -q_23+q_12
    q_C3 = q_rad+q_teto-q_31+q_23
    
    dT1_dt = (1/n) * q_C1 / C_1
    dT2_dt = (1/n) * q_C2 / C_2
    dT3_dt = (1/n) * q_C3 / C_3
    dTNH3s_dt = eps*(dT3_dt-dT1_dt)*(m_ar*Cp_ar)/(m_nh3*Cp_nh3)
    
    return [dT1_dt, dT2_dt, dT3_dt, dTNH3s_dt]

# Condições iniciais
T1_0   =  14 + 273
T2_0   =  20 + 273
T3_0   =  26 + 273

# Intervalo de tempo e resolução
t_span = (0, 365*86400)
t_eval = np.linspace(*t_span, n)

TNH3s_0 = TNH3_0+eps*(T3_0-T1_0)*(m_ar*Cp_ar)/(m_nh3*Cp_nh3)

X_0 = [T1_0, T2_0, T3_0, TNH3s_0]

# Solução pelo método de Runge-Kutta 4/5
sol = solve_ivp(circuit_dynamics, t_span, X_0, method='RK45', t_eval=t_eval)

# Plotagem dos resultados
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='T1 (K)')
plt.plot(sol.t, sol.y[1], label='T2 (K)')
plt.plot(sol.t, sol.y[2], label='T3 (K)')
plt.plot(sol.t, sol.y[3], label='TNH3s (K)')
plt.xlabel('Tempo (s)')
plt.ylabel('Temperatura (K)')
plt.legend()
plt.grid()
plt.show()

# Representação no espaço de estados
T_1_minmax = min(sol.y[0]), max(sol.y[0])
T_2_minmax = min(sol.y[1]), max(sol.y[1])
T_3_minmax = min(sol.y[2]), max(sol.y[2])

R_12_lin   = sum([R_e(T_1_minmax[0],T_2_minmax[0]), R_e(T_1_minmax[0],T_2_minmax[1]), R_e(T_1_minmax[1],T_2_minmax[0]), R_e(T_1_minmax[1],T_2_minmax[1])])/4
R_23_lin   = sum([R_e(T_2_minmax[0],T_3_minmax[0]), R_e(T_2_minmax[0],T_3_minmax[1]), R_e(T_2_minmax[1],T_3_minmax[0]), R_e(T_2_minmax[1],T_3_minmax[1])])/4
R_teto_lin = sum([R_teto(T_3_minmax[0], Tinf_0), R_teto(T_3_minmax[1], Tinf_0)])/2

A11 = -(1/C_1)*(1/R_solo+1/R_12_lin+1/R_31)
A12 = 1/(C_1*R_12_lin)
A13 = -(1/C_1)*(1/R_31+m_ar*Cp_ar*eps)
A14 = 0

A21 = 1/(C_2*R_12_lin)
A22 = -(C_2)*(1/R_12_lin+1/R_23_lin)
A23 = 1/(C_2*R_23_lin)
A24 = 0

A31 = 1/(C_3*R_31)
A32 = 1/(C_3*R_23_lin)
A33 = -(1/C_3)*(1/R_teto_lin+1/R_31+1/R_23_lin)
A34 = 0

A41 = -eps*C_mm
A42 = 0
A43 = eps*C_mm
A44 = 0

B11 = m_ar*Cp_ar*eps/C_1
B21 = 1/(C_1*R_solo)

B12 = 0
B22 = 0

B13 = 0
B23 = 1/(C_3*R_teto_lin)

B14 = 0
B24 = 0


A = np.array([[A11, A21, A31, A41],
              [A21, A22, A23, A24],
              [A31, A32, A33, A34],
              [A41, A42, A43, A44]])

B = np.array([[B11, B21],
              [B12, B22],
              [B13, B23],
              [B14, B24]])

C = np.eye(4)
D = np.zeros([4,2])

sys = StateSpace(A, B, C, D)

num, den = ss2tf(A, B, C, D)

# Exibir a função de transferência
print(f"  Denominador: {den}\n")
title_list = ["T1", "T2", "T3", "TNH3s"]
for i in range(4):
    print(f"Função de Transferência {i+1}:")
    print(f"  Numerador: {num[i]}")
    
    bode_ret = TransferFunction(num[i], den)

    w, mag, phase = bode(bode_ret)

    # Plotar o diagrama de Bode
    plt.figure(figsize=(10, 6))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(w, mag)
    plt.title(f"Diagrama de Bode para {title_list[i]}")
    plt.xlabel("Frequência (rad/s)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    # Fase
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel("Frequência (rad/s)")
    plt.ylabel("Fase (graus)")
    plt.grid(True)

    plt.show()