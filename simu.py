from qrad import Q_rad, Q_rad_preload
from parametros import *
import matplotlib.pyplot as plt
import numpy as np

def q_i(T3, TNH3, m_nh3):
    m_nh3 = 0.06
    C = (m_ar*Cp_ar)/(m_nh3*Cp_nh3)
    eps = (1-np.e**(-(1+C)*NUT))/(1+C)
    return m_ar*Cp_ar*eps*(T3-TNH3)

t_int = 1000
n = 100000

T_inf  = 273 + 10
T_solo = 273 + 10

#Qrad = Q_rad_preload()

t = np.linspace(0,t_int,n)

T1   = 273 + 14
T2   = 273 + 20
T3   = 273 + 26
TNH3 = 273 - 5

X = [[],[],[]]

for i in t:
    #j = int(i/t_int)
    
    q_solo = (t_int/n)*(T_solo-T1)/(R_solo)
    q_teto = T_inf/R_teto(T3,T_inf)
    q_12   = (t_int/n)*(T1-T2)*R_e(T1,T2)
    q_23   = (t_int/n)*(T2-T3)*R_e(T2,T3)
    q_31   = (t_int/n)*(T3-T1)*R_31
    q_t    = q_i(T3,TNH3,0)
    q_r    = 0#Q_rad(i)
    
    T1 += (-q_solo-q_12+q_31+q_t)/C_1
    T2 += (-q_23+q_12)/C_2
    T3 += (-q_r -q_teto-q_31-q_t+q_23)/C_3
    
    X[0].append(T1)
    X[1].append(T2)
    X[2].append(T3)

print(X[0][:1000:50])
print()
print(X[1][:1000:50])
print()
print(X[2][:1000:50])


plt.plot(t, X[0])
plt.plot(t, X[1])
plt.plot(t, X[2])
plt.show()