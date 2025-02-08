from qrad import Q_rad
import matplotlib.pyplot as plt

import numpy as np

t = np.linspace(0,86400*365,10000)

Q = []

for t_ in t:
    Q.append(Q_rad(t_))

with open("qrad.txt", "w") as file:
    file.write(str(Q))