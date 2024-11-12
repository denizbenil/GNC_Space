import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# CRTBP modeline göre diferansiyel denklem sistemi ve STM
def CRTBP_STM(t, x, mu):
    rr = x[0:3]
    vv = x[3:6]
    PHI = x[6:].reshape((6, 6))
    
    rr1 = rr - np.array([-mu, 0, 0])
    r1 = np.linalg.norm(rr1)
    rr2 = rr - np.array([1 - mu, 0, 0])
    r2 = np.linalg.norm(rr2)
    
    rvdot = np.array([
        vv[0],
        vv[1],
        vv[2],
        2 * vv[1] + rr[0] - ((1 - mu) * (rr[0] + mu) / r1**3 + mu * (rr[0] + mu - 1) / r2**3),
        -2 * vv[0] + rr[1] - ((1 - mu) / r1**3 + mu / r2**3) * rr[1],
        -((1 - mu) / r1**3 + mu / r2**3) * rr[2]
    ])
    
    A = compute_A_matrix(rr, r1, r2, mu)
    PHI_dot = A @ PHI
    xdot = np.concatenate((rvdot, PHI_dot.flatten()))
    return xdot

# A matrisini hesaplayan fonksiyon
def compute_A_matrix(rr, r1, r2, mu):
    U_xx = 1 - (1 - mu) / r1**3 - mu / r2**3 + 3 * (1 - mu) * (rr[0] + mu)**2 / r1**5 + 3 * mu * (rr[0] + mu - 1)**2 / r2**5
    U_xy = 3 * (1 - mu) * (rr[0] + mu) * rr[1] / r1**5 + 3 * mu * (rr[0] + mu - 1) * rr[1] / r2**5
    U_yy = 1 - (1 - mu) / r1**3 - mu / r2**3 + 3 * (1 - mu) * rr[1]**2 / r1**5 + 3 * mu * rr[1]**2 / r2**5
    U_zz = -(1 - mu) / r1**3 - mu / r2**3 + 3 * (1 - mu) * rr[2]**2 / r1**5 + 3 * mu * rr[2]**2 / r2**5
    
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [U_xx, U_xy, 0, 0, 2, 0],
        [U_xy, U_yy, 0, -2, 0, 0],
        [0, 0, U_zz, 0, 0, 0]
    ])
    return A

# Olay fonksiyonu (y-eksenini geçişi tespit eder)
def ycross(t, x, mu):
    return x[1]  # y = 0'ı geçtiğinde durdurur

ycross.terminal = True
ycross.direction = 0

# ODE çözücü fonksiyonu
def propagate(x0, tspan, mu, stop=False):
    if stop:
        events = lambda t, x: ycross(t, x, mu)
        sol = solve_ivp(lambda t, x: CRTBP_STM(t, x, mu), [0, tspan], x0, method='RK45',
                        rtol=1e-12, atol=1e-12, events=events)
    else:
        sol = solve_ivp(lambda t, x: CRTBP_STM(t, x, mu), [0, tspan], x0, method='RK45',
                        rtol=1e-12, atol=1e-12)
    
    t = sol.t
    rv = sol.y[:6, :].T
    PHI_a = sol.y[6:, -1].reshape((6, 6)) if stop else None
    return t, rv, PHI_a

# Başlangıç koşulları ve parametreler
mu = 0.012150585609624  # Ay için
x0 = np.array([1.2, 0, 0, 0, 0.1, 0, *(np.identity(6).flatten())])  # Başlangıç durumu ve STM

tspan = 10  # Çözüm için zaman aralığı
t, rv, PHI_a = propagate(x0, tspan, mu, stop=True)

# Sonuçların çizdirilmesi
plt.figure()
plt.plot(rv[:, 0], rv[:, 1], label='Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Çift Cisim Problemi Yörüngesi')
plt.grid()
plt.legend()
plt.show()
