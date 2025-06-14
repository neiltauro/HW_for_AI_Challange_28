# memristor_sim.py
# Simulating a memristor with pinched hysteresis (Biolek Model)
# by Neil Tauro

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Biolek memristor parameters
# ------------------------------------------------------------------------------
R_on = 100
R_off = 16000
D = 10e-9
mu_v = 10e-14
p = 10

# ------------------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------------------
timesteps = 1000
time = np.linspace(0, 1, timesteps)
V_max = 1
frequency = 5
voltage = V_max * np.sin(2 * np.pi * frequency * time)

# ------------------------------------------------------------------------------
# Initial state
# ------------------------------------------------------------------------------
w = D/2
current = np.ones(timesteps) * 0

# ------------------------------------------------------------------------------
# Biolek window function
# ------------------------------------------------------------------------------
def f(w, p, D):
    return 1 - (2*w/D - 1)**(2*p)

# ------------------------------------------------------------------------------
# Integrate state w
# ------------------------------------------------------------------------------
for i in range(1, timesteps):
    G = 1 / (R_on * (w/D) + R_off * (1 - w/D))
    current[i-1] = G * voltage[i-1]
    dw = mu_v * R_on/D * current[i-1] * f(w, p, D) * (time[i] - time[i-1]) 
    w += dw
    w = np.clip(w, 0, D)

# Final compute
G = 1 / (R_on * (w/D) + R_off * (1 - w/D))
current[-1] = G * voltage[-1]

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
plt.plot(voltage, current)
plt.title('Memristor I–V curve (pinched hysteresis)')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.grid()
plt.show()
plt.savefig("memristor_iv_curve.png")
print("Plot successfully saved to memristor_iv_curve.png")
