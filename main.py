import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import TransferFunction, bode, StateSpace, ss2tf, tf2zpk
from parametros import *
from qrad import Q_rad

n = 1000000

# Parâmetros de ajuste para diferentes situações    
TNH3_0  = -10 + 273
Tinf_0  =  15 + 273

A_TNH3_d = 2
A_TNH3_a = 6

A_Tinf_d = 1
A_Tinf_a = 1

#m_nh3 = 0.05 # Padrão é 0.1

# Perturbações
def Tinf_f(t):
    return Tinf_0 + A_Tinf_d*np.sin(np.pi*t/86400) + A_Tinf_a*np.sin(2*np.pi*t/(86400*365))

def TNH3_f(t):
    return TNH3_0 + A_TNH3_d*np.sin(np.pi*t/86400) + A_TNH3_a*np.sin(2*np.pi*t/(86400*365))

# Definição do sistema de equações diferenciais
def Sistema(t, y):
    T1, T2, T3, TNH3s = y
    
    TNH3 = Tinf_f(t)
    Tinf = TNH3_f(t)
    
    q_solo = (T1 - Tinf)/R_solo
    q_teto = (Tinf - T3)/R_teto(T3,Tinf)
    q_rad  = Q_rad(np.floor(t))
    q_i    = m_ar*Cp_ar*(T3-TNH3)*1000*eps*0.1
    q_12   = (T1-T2)/R_e(T1,T2)
    q_23   = (T2-T3)/R_e(T2,T3)
    q_31   = (T3-T1)/R_31
    
    q_C1 = -q_solo-q_12+q_31-q_i
    q_C2 = -q_23+q_12
    q_C3 = q_rad+q_teto-q_31+q_23
    
    dT1_dt = (1/n) * q_C1 / C_1
    dT2_dt = (1/n) * q_C2 / C_2
    dT3_dt = (1/n) * q_C3 / C_3
    dTNH3s_dt = (dT3_dt-dT1_dt)*C_mm
    
    return [dT1_dt, dT2_dt, dT3_dt, dTNH3s_dt]

# Condições iniciais
T1_0   =  20 + 273
T2_0   =  25 + 273
T3_0   =  30 + 273

# Intervalo de tempo e resolução
t_span = (0, 365*86400)
t_eval = np.linspace(*t_span, n)

TNH3s_0 = TNH3_0 + (T3_0-T1_0)*C_mm

# Vetor de estado
X_0 = [T1_0, T2_0, T3_0, TNH3s_0]

# Solução pelo método de Runge-Kutta 4/5
sol = solve_ivp(Sistema, t_span, X_0, method='RK45', t_eval=t_eval)

# Solução pelo método LSODA (Livermore Solver for Ordinary Differential Equations with Automatic method switching)
#sol = solve_ivp(Sistema, t_span, X_0, method='LSODA', t_eval=t_eval)

# Plotagem dos resultados
plt.figure(figsize=(10, 5))
plt.plot(sol.t/86400, Tinf_f(sol.t)-273, label='Tinf (ºC)', alpha=0.4, color='gray')
plt.plot(sol.t/86400, TNH3_f(sol.t)-273, label='TNH3e (ºC)', alpha=0.4, color='gray')
plt.plot(sol.t/86400, sol.y[0]-273, label='T1 (ºC)', color='blue')
plt.plot(sol.t/86400, sol.y[1]-273, label='T2 (ºC)', color='orange')
plt.plot(sol.t/86400, sol.y[2]-273, label='T3 (ºC)', color='red')
plt.plot(sol.t/86400, sol.y[3]-273, label='TNH3s (ºC)', color='purple')
plt.xlabel('Tempo (dias)')
plt.ylabel('Temperatura (ºC)')
plt.legend(loc='upper right')
plt.grid()
plt.show()

'''
# Representação no espaço de estados

# Versão linearizada das resistências
R_12_lin   = sum([R_e(T_1_minmax[0],T_2_minmax[0]), R_e(T_1_minmax[0],T_2_minmax[1]), R_e(T_1_minmax[1],T_2_minmax[0]), R_e(T_1_minmax[1],T_2_minmax[1])])/4
R_23_lin   = sum([R_e(T_2_minmax[0],T_3_minmax[0]), R_e(T_2_minmax[0],T_3_minmax[1]), R_e(T_2_minmax[1],T_3_minmax[0]), R_e(T_2_minmax[1],T_3_minmax[1])])/4
R_teto_lin = sum([R_teto(T_3_minmax[0], Tinf_0), R_teto(T_3_minmax[1], Tinf_0)])/2

# Coeficientes da matriz A
A11 = -(1/C_1)*(1/R_solo + 1/R_12_lin + 1/R_31)
A21 = 1/(C_1*R_12_lin)
A31 = (1/C_1)*(1/R_31 - m_ar*Cp_ar*eps)
A41 = 0

A12 = 1/(C_2*R_12_lin)
A22 = -1/(C_2)*(1/R_12_lin+1/R_23_lin)
A32 = 1/(C_2*R_23_lin)
A42 = 0

A13 = 1/(C_3*R_31)
A23 = 1/(C_3*R_23_lin)
A33 = -(1/C_3)*(1/R_teto_lin+1/R_31+1/R_23_lin)
A43 = 0

A14 = C_mm*(1/(C_3*R_31)+1/C_1*(1/R_solo+1/R_12_lin+1/R_31))
A24 = C_mm*(1/(C_3*R_23_lin)-1/(C_1*R_12_lin))
A34 = C_mm*(-1/C_3*(1/R_teto_lin+1/R_31+1/R_23_lin)-1/C_1*(1/R_31-m_ar*Cp_ar*eps))
A44 = 0

# Coeficientes da matriz B
B11 = m_ar*Cp_ar*eps/C_1
B12 = 1/(C_1*R_solo)

B21 = 0
B22 = 0

B31 = 0
B32 = 1/(C_3*R_teto_lin)

B41 = -C_mm*m_ar*Cp_ar*eps/C_1
B42 = C_mm*(1/(C_3*R_teto_lin)-1/(C_1*R_solo))


A = np.array([[A11, A12, A13, A14],
              [A21, A22, A23, A24],
              [A31, A32, A33, A34],
              [A41, A42, A43, A44]])

B = np.array([[B11, B12],
              [B21, B22],
              [B31, B32],
              [B41, B42]])

# Coeficientes da matriz C
C11 = 0
C12 = 1/(C_1+C_2)*1/R_12_lin
C13 = 1/(C_1+C_3)*1/R_31
C14 = 0

C21 = 1/(C_2+C_1)*1/R_12_lin
C22 = 0
C23 = 1/(C_2+C_3)*1/R_23_lin
C24 = 0

C31 = 1/(C_3+C_1)*1/R_31
C32 = 1/(C_3+C_2)*1/R_23_lin
C33 = 0
C34 = 0

C41 = eps*m_ar*Cp_ar/C_1
C42 = 0
C43 = eps*m_ar*Cp_ar/C_3
C44 = 0

# Coeficientes da matriz D
D11 = eps*m_ar*Cp_ar/C_1
D12 = 1/(C_1*R_solo)

D21 = eps*C_mm*(1/R_23_lin-1/R_12_lin)*1/C_2
D22 = (1/R_12_lin+1/R_solo+1/R_23_lin+1/R_teto_lin)*1/C_2

D31 = eps*m_ar*Cp_ar/C_3
D32 = 1/(C_3*R_teto_lin)

D41 = 1
D42 = eps*C_mm*(1/(C_1*R_solo)+1/(C_3*R_teto_lin))

C = np.array([[C11, C12, C13, C14],
              [C21, C22, C23, C24],
              [C31, C32, C33, C34],
              [C41, C42, C43, C44]])

D = np.array([[D11, D12],
              [D21, D22],
              [D31, D32],
              [D41, D42]])

sys = StateSpace(A, B, C, D)

# Resolve as equações
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
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    # Fase
    plt.subplot(2, 1, 2)
    plt.semilogx(w, phase)
    plt.xlabel("Frequência (rad/s)")
    plt.ylabel("Fase (graus)")
    plt.grid(True)

    plt.show()

print('Zeros, polos e ganhos das funções:')
for i in range(4):
    z, p, k = tf2zpk(num[i], den)
    print(f"T{i+1}:\nZeros: {z}\nPolos: {p}\nGanho: {k}\n")

'''