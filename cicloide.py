import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Aceleração devido à gravidade (m/s²); posição final da partícula (m).
g = 9.81
x2, y2 = 1, 0.65

def cicloide(x2, y2, N=100):
    """Retorna o caminho da curva de braquistócrona de (0,0) a (x2, y2)."""

    # Primeiro, encontra theta2 a partir de (x2, y2) numericamente (pelo método de Newton-Raphson).
    def f(theta):
        return y2/x2 - (1-np.cos(theta))/(theta-np.sin(theta))
    theta2 = newton(f, np.pi/2)

    # O raio do círculo que gera a cicloide.
    R = y2 / (1 - np.cos(theta2))

    theta = np.linspace(0, theta2, N)
    x = R * (theta - np.sin(theta))
    y = R * (1 - np.cos(theta))

    # O tempo de viagem
    T = theta2 * np.sqrt(R / g)
    print('T(cicloide) = {:.3f}'.format(T))
    return x, y, T

def linear(x2, y2, N=100):
    """Retorna o caminho de uma linha reta de (0,0) a (x2, y2)."""

    m = y2 / x2
    x = np.linspace(0, x2, N)
    y = m*x

    # O tempo de viagem
    T = np.sqrt(2*(1+m**2)/g/m * x2)
    print('T(linear) = {:.3f}'.format(T))
    return x, y, T

def func(x, f, fp):
    """O integrando da integral de tempo a ser minimizado para um caminho f(x)."""

    return np.sqrt((1+fp(x)**2) / (2 * g * f(x)))

def círculo(x2, y2, N=100):
    """Retorna o caminho de um arco circular entre (0,0) e (x2, y2).

    O círculo usado é aquele com uma tangente vertical em (0,0).

    """

    # Raio do círculo
    r = (x2**2 + y2**2)/2/x2

    def f(x):
        return np.sqrt(2*r*x - x**2)
    def fp(x):
        return (r-x)/f(x)

    x = np.linspace(0, x2, N)
    y = f(x)

    # Calcula o tempo de viagem por integração numérica.
    T = quad(func, 0, x2, args=(f, fp))[0]
    print('T(círculo) = {:.3f}'.format(T))
    return x, y, T

def parábola(x2, y2, N=100):
    """Retorna o caminho de um arco parabólico entre (0,0) e (x2, y2).

    A parábola usada é aquela com uma tangente vertical em (0,0).

    """

    c = (y2/x2)**2

    def f(x):
        return np.sqrt(c*x)
    def fp(x):
        return c/2/f(x)

    x = np.linspace(0, x2, N)
    y = f(x)

    # Calcula o tempo de viagem por integração numérica.
    T = quad(func, 0, x2, args=(f, fp))[0]
    print('T(parábola) = {:.3f}'.format(T))
    return x, y, T

# Plota um gráfico comparando os quatro caminhos.
fig, ax = plt.subplots()

for curve in ('cicloide', 'círculo', 'parábola', 'linear'):
    x, y, T = globals()[curve](x2, y2)
    ax.plot(x, y, lw=4, alpha=0.5, label='{}: {:.3f} s'.format(curve, T))
ax.legend()

ax.set_xlabel('$x$')
ax.set_xlabel('$y$')
ax.set_xlim(0, 1)
ax.set_ylim(0.8, 0)
plt.savefig('braquistócrona.png')
plt.show()
