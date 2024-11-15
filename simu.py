import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Tau_t = 0.89
Tau_s = 0.83
offset_dia0 = 172
A_f = 50

U_t = 6.7             # Coef. global (condução + convecção) do vidro
U_s = 6.8             # Coef. global (condução + convecção) do PE Duplo
T_H = 273+26
T_0 = 273+14

T_inf = 273+10
T_solo = T_inf

cSB = 5.67e-8
eps_s = 0.85
eps_a = 0.883

h_c = 15


days = [x+20 for x in [-30,1,32,60,91,121,152,182,213,244,274,305,335,366]]

df_abc = pd.DataFrame([
[1233,0.142,0.057],
[1230,0.142,0.058],
[1214,0.144,0.060],
[1185,0.156,0.071],
[1135,0.180,0.097],
[1104,0.196,0.121],
[1088,0.205,0.134],
[1085,0.207,0.136],
[1107,0.201,0.122],
[1151,0.177,0.092],
[1192,0.160,0.073],
[1120,0.149,0.063],
[1233,0.142,0.057],
[1230,0.142,0.058]],
columns=['a','b','c'], index=days)

def interpolate(s,x,y):
    x0,x1 = x
    y0,y1 = y
    return ((y1-y0)*((s-x0)/(x1-x0)))+y0

def get_d(d):
    i = days.index([a for a in days if d<=a][0])
    d0 = days[i-1]
    d1 = days[i]
    return d0,d1

def get_abc(dia):
    dia_i = get_d(dia)
    a = interpolate(dia,dia_i,(df_abc['a'][dia_i[0]],df_abc['a'][dia_i[1]]))
    b = interpolate(dia,dia_i,(df_abc['b'][dia_i[0]],df_abc['b'][dia_i[1]]))
    c = interpolate(dia,dia_i,(df_abc['c'][dia_i[0]],df_abc['c'][dia_i[1]]))
    return a,b,c

def beta_sol(dia):
    return np.pi/2+np.sin(2*np.pi*(dia-offset_dia0)/(365))*0.4101524

def beta_dia(seg):
    seg = seg%86400
    if seg < 5*86400/24 or seg > 19*86400/24: return 0
    return np.pi*(seg/86400)

def beta(seg):
    return np.pi*(np.cos(beta_sol(seg//86400))*np.sin(beta_dia(seg)))

def I(seg):
    a,b,c = get_abc(seg//86400)
    beta_ = np.sin(beta(seg))
    return (c+beta_)*a*(0 if beta_ == 0 else np.e**(-b/beta_))

def Q_rad(seg):
    return Tau_t*I(seg)*A_f

#d = np.linspace(1,7*86400,10000)
#Q = []

#for d_ in d:
    Q.append(Q_rad(d_))
    
#plt.plot(d/86400,Q)
#plt.show()

Q_teto  = A_f*U_t*(T_H - T_inf)
Q_solo  = A_f*U_s*(T_0 - T_solo)
Q_rsolo = eps_s*Tau_s*A_f*cSB*(T_0**4-eps_a*T_solo**4)

Q_VC = Q_rad(86400/2)-Q_teto-Q_solo-Q_rsolo

T = Q_VC/(A_f*15)+20

print(T)