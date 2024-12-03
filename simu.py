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
T_2 = 273+20
T_0 = 273+14

T_inf = 273+10
T_solo = T_inf

cSB = 5.67e-8
eps_s = 0.85
eps_a = 0.883

h_c  = 15
h_12 = 3.4
h_23 = 3.4

rho_ar = 1.2        # kg/m³
cp_ar  = 1004       # J/kgK
m_ar   = 8.13e-2    # kg/s

cp_nh3 = 2230       # J/kgK
m_nh3  = 0.06       # kg/s

Q_12 = A_f*h_12*(T_H - T_2)
Q_21 = -Q_12
Q_23 = A_f*h_12*(T_2-T_0)
Q_32 = -Q_23

Q_teto  = A_f*U_t*(T_H - T_inf)
Q_solo  = A_f*U_s*(T_0 - T_solo)
Q_rsolo = eps_s*Tau_s*A_f*cSB*(T_0**4-eps_a*T_solo**4)

days = [x+20 for x in [-30,1,32,60,91,121,152,182,213,244,274,305,335,366]]

df_abc = pd.DataFrame([
[1085,0.207,0.136],
[1107,0.201,0.122],
[1151,0.177,0.092],
[1192,0.160,0.073],
[1120,0.149,0.063],
[1233,0.142,0.057],
[1230,0.142,0.058],
[1233,0.142,0.057],
[1230,0.142,0.058],
[1214,0.144,0.060],
[1185,0.156,0.071],
[1135,0.180,0.097],
[1104,0.196,0.121],
[1088,0.205,0.134]],
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
    return np.pi/2+np.cos(2*np.pi*(dia-offset_dia0)/365)*0.4101524/3

def beta_dia(seg):
    seg = seg%86400
    if seg < 5*86400/24 or seg > 19*86400/24: return 0
    return np.pi*(seg/86400)

def beta(seg):
    return np.pi*(np.cos(beta_sol(seg//86400))*np.sin(beta_dia(seg)))

def I(seg):
    a,b,c = get_abc(seg//86400)
    beta_ = abs(np.sin(beta(seg)))
    if beta_ < 10e-3: return 0
    return (c+beta_)*a*np.e**(-b/beta_)

def Q_rad(seg):
    return Tau_t*I(seg)*A_f

def m_nh3_corr(s0, corr_range):
    T_min = 285.8
    T_max = 302
    T_ar = 283
    
    m_nh3_aj = []
    for s in range(corr_range):
        T0_ar = T_ar
        T_ar = T0_ar + (T_max-T_min)*np.sin(np.pi*((s0+s)-5*60)/(14*60))
        
        m_nh3_aj.append(((Q_rad((s0+s)*60)-Q_teto-Q_solo-Q_rsolo)-m_ar*cp_ar*(T_ar-T0_ar))/cp_nh3)
        
    return m_nh3_aj

def Q_VC1_corr(s0, corr_range):
    T_min = 285.8
    T_max = 302
    T_ar = 283
    
    Q_aj = []
    for s in range(corr_range):
        T0_ar = T_ar
        T_ar = T0_ar + (T_max-T_min)*np.sin(np.pi*((s0+s)-5*60)/(14*60))
        
        Q_VC1 = Q_rad(s0+s)-Q_teto-Q_12 + m_ar*cp_ar*(T_ar-T0_ar)
        
        Q_aj.append(-Q_VC1)
        
    return Q_aj

def Q_VC2_corr(s0, corr_range):
    T_min = 285.8
    T_max = 302
    T_ar = 283
    
    Q_aj = []
    for s in range(corr_range):
        T0_ar = T_ar
        T_ar = T0_ar + (T_max-T_min)*np.sin(np.pi*((s0+s)-5*60)/(14*60))
        
        Q_VC2 = Q_21-Q_23 + m_ar*cp_ar*(T_ar-T0_ar)
        
        Q_aj.append(-Q_VC2)
        
    return Q_aj
    
def Q_VC3_corr(s0, corr_range):
    T_min = 285.8
    T_max = 302
    T_ar = 283
    
    Q_aj = []
    for s in range(corr_range):
        T0_ar = T_ar
        T_ar = T0_ar + (T_max-T_min)*np.sin(np.pi*((s0+s)-5*60)/(14*60))
        
        Q_VC3 = Q_32-Q_solo-Q_rsolo + m_ar*cp_ar*(T_ar-T0_ar) 
        
        Q_aj.append(-Q_VC3)
        
    return Q_aj    

def plot(type, di, df):
    di = di*1440
    df = df*1440
    days = df-di
    d = np.linspace(di,df,days)
    Q_VC  = []
    Q_VC1 = []
    Q_VC2 = Q_12+Q_32
    Q_VC3 = Q_23-Q_solo-Q_rsolo

    for d_ in d:
        Q_rad_ = Q_rad(d_)
        Q_VC.append(Q_rad_-Q_teto-Q_solo-Q_rsolo)
        Q_VC1.append(Q_rad_-Q_teto-Q_12)
    
    plt.figure(figsize=(16, 9))
    if type[1] == 'Q':
        plt.title("Calor trocado total pela estufa pelo tempo")
        plt.plot(d/1440, Q_VC, label="Q_VC")

    match type[0]:
        case '1':
            plt.title("Calor trocado total pelo VC1 estufa pelo tempo")
            if type[2] == 'Q': plt.plot(d/1440, Q_VC1, label="Q_VC1")
            if type[3] == 'C': plt.plot(d/1440, Q_VC1_corr(di, days), label="Q_VC1_corr")
        case '2':
            plt.title("Calor trocado total pelo VC2 estufa pelo tempo")
            if type[2] == 'Q': plt.plot(d/1440, days*[Q_VC2], label="Q_VC2")
            if type[3] == 'C': plt.plot(d/1440, Q_VC2_corr(di, days), label="Q_VC2_corr")
        case '3':
            plt.title("Calor trocado total pelo VC3 estufa pelo tempo")
            if type[2] == 'Q': plt.plot(d/1440, days*[Q_VC3], label="Q_VC3")
            if type[3] == 'C': plt.plot(d/1440, Q_VC3_corr(di, days), label="Q_VC3_corr")
        case 'm':
            plt.title("Correção da vazão pelo tempo")
            plt.plot(d/1440, m_nh3_corr(di, days), label="m_nh3_corr")
            plt.ylabel("Ajuste de vazão (kg/s)")
    
    if type[0] != 'm': plt.ylabel("Calor trocado pelo VC (W)")
    plt.xlabel("Tempo (dia do ano)")
    plt.rc('legend', fontsize=12)
    plt.legend(loc="upper left")
    plt.show()
    
plot('m___',0,365)

'''
d = np.linspace(0,90*86400,43200)
Q_  = []
Q_aj = []

T_min = 285.8
T_max = 302
T_ar = 283

for d_ in d:
    Q_.append((Q_rad(d_)-Q_teto-Q_solo-Q_rsolo))
    
    T0_ar = T_ar
    T_ar = T0_ar + (T_max-T_min)*np.sin(np.pi*(d_-5*3600)/(14*3600))
    
    Q_aj.append(-m_ar*cp_ar*(T_ar-T0_ar))

plt.figure(figsize=(16, 9))

plt.title("Calor trocado e correção total no VC (estufa completa) pelo tempo")
plt.plot(d/86400, Q_, label="Q_VC")
plt.plot(d/86400, Q_aj, label="Q_aj")
plt.ylabel("Calor trocado pelo VC (W)")
plt.xlabel("Tempo (dia do ano)")
plt.legend(loc="upper left")
plt.show()'''