import numpy as np
import matplotlib.pyplot as plt

Tinf_0 = (10 + 273)

def Tinf(t):
    return Tinf_0 + 3*np.sin(np.pi*t/86400) + 10*np.sin(2*np.pi*(t//86400)/365)

t = np.linspace(0,2*365*86400,10000)

plt.plot(t,Tinf(t))
plt.show()