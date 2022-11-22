import numpy as np
import matplotlib as mpl
import math
import sympy as sy

def temp(t_guess, a1, b1, c1, a2, b2, c2, x_2, pt):   #root finding with Newton - Raphson method...
    t_sym = sy.symbols('t_sym')
    s_eval = ((1 - x_2) * (10 ** (a1 - ((b1) / (t_guess + c1))))) + ((x_2) * (10 ** (a2 - ((b2) / (t_guess + c2))))) - pt
    
    def fprime(t_sym):
        s_sym = ((1 - x_2) * (10 ** (a1 - ((b1) / (t_sym + c1))))) + ((x_2) * (10 ** (a2 - ((b2) / (t_sym + c2))))) - pt
        sp = sy.diff(s_sym, t_sym)
        return sp
    
    sp2 = fprime(t_sym).evalf(subs={t_sym: t_guess})
    t_guess2 = t_guess - (s_eval / sp2)

    while abs(t_guess2 - t_guess) > 0.01:
        t_guess = t_guess2
        sp2 = fprime(t_sym).evalf(subs={t_sym: t_guess})
        s_eval = ((1 - x_2) * (10 ** (a1 - (b1) / (t_guess + c1)))) + ((x_2) * (10 ** (a2 - (b2) / (t_guess + c2)))) - pt
        t_guess2 = t_guess - (s_eval / sp2)

    return t_guess2

def press(a, b, c, t):
    result = 10 ** (a - ((b)/(t + c)))
    return result
L0 = 100
A1 = 6.90565
B1 = 1211.033
C1 = 220.79

A2 = 6.95464
B2 = 1344.8
C2 = 219.482

Pt = 1.2 * 760
x2i = 0.4
x2f = 0.8
n = 400
h = (x2f - x2i) / n
x2 = np.arange(x2i, x2f + h, h)
x2 = list(x2)
L = [L0]
for g in x2:  #ODE solving with 4th order Runge - Kutta method...
    u = g
    l = L[-1]
    T = temp(70, A1, B1, C1, A2, B2, C2, u, Pt)
    p2 = press(A2, B2, C2, T)
    k2 = p2 / Pt
    k_1 = h * ((l) / (u * (k2 - 1)))
    u = g + (h / 2)
    l = L[-1] + (k_1 / 2)
    T = temp(70, A1, B1, C1, A2, B2, C2, u, Pt)
    p2 = press(A2, B2, C2, T)
    k2 = p2 / Pt
    k_2 = h * ((l) / (u * (k2 - 1)))
    l = L[-1] + (k_2 / 2)
    k_3 = h * ((l) / (u * (k2 - 1)))
    u = g + (h)
    l = L[-1] + (k_3)
    T = temp(70, A1, B1, C1, A2, B2, C2, u, Pt)
    p2 = press(A2, B2, C2, T)
    k2 = p2 / Pt
    k_4 = h * ((l) / (u * (k2 - 1)))
    f = L[-1] + 1 / 6 * (k_1 + (2 * k_2) + (2 * k_3) + k_4)
    L.append(f)
    print(L[-1])

print("\n*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*_*\n")
print("%s moles remain in batch when x2=0.8." % str(L[-1]))