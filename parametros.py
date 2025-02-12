import numpy as np

g = 9.81                # m/s²
A = 50                  # m²
P = 30                  # m
V = 10*5*5              # m³
V_1 = V/3               # m³
V_2 = V/3               # m³
V_3 = V/3               # m³

m_ar = 8.13e-2          # kg/s
Cp_ar = 1.006           # kJ/kgK
rho_ar = 1.2046         # kg/m³

m_nh3 = 0.06            # kg/s
Cp_nh3 = 2.13           # kJ/kgK

#A_T = 10               # m²
#U_T = 120              # W/m²K
NUT = (10*120)/(Cp_ar*m_ar)

C_mm = (m_ar*Cp_ar)/(m_nh3*Cp_nh3)
eps = (1-np.e**(-(1+C_mm)*NUT))/(1+C_mm)

# R_solo
# e_s = 0.1             # m
# h_s = 15              # w/m²K
# k_s = 0.48            # W/mK
R_solo = 1/(15*50) + 0.1/(0.45*50)

# R_teto
def R_teto(T3, Tinf):
    v = 1.568e-5        # m²/s
    k = 2.634e-2        # W/mK
    Pr = 0.7071         # -
    e_t = 0.05          # m
    L = A/P             # m
    beta = 1/Tinf
    
    Gr = (g*beta*abs(T3-Tinf)*(L**3))/v**2
    Ra = Gr*Pr
    Nu = 0.59*Ra**0.25
    h_t = (Nu*k)/L

    return 1/(h_t*A)+e_t/(A*k)

# R_12 ou R_23
def R_e(T1, T2):        # T1 embaixo, T2 em cima
    v = 1.667e-5        # m²/s
    Pr = 0.7080         # -
    e_1 = 0.1           # m
    e_2 = 0.1           # m
    k = 2.588e-2        # W/mK
    L = A/P             # m
    beta = 1/T1
    
    Gr = (g*beta*abs(T2-T1)*(e_1+e_2)**3)/v**2
    Nu = (0.42*(Gr*Pr)**0.25)*(Pr**0.012)*((e_1+e_2)/L)**0.3
    h_e = (Nu*k)/(e_1+e_2)
    
    return ((e_1+e_2)/(k*A))+1/(h_e*A)

# R_31
R_31 = m_ar*Cp_ar

C_1 = Cp_ar*rho_ar*V_1

C_2 = Cp_ar*rho_ar*V_2

C_3 = Cp_ar*rho_ar*V_3
