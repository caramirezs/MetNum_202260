import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def graficar(f, x_i, x_f, num=1000):
    plt.style.use('seaborn')
    """
    Gráfica de funciones algebraicas
    :param f: función, previamente definida
    :param x_i: límite inferior del intervalo
    :param x_f: límite superior del intervalo
    :param num: división del intervalo
    :return: gráfica de la función
    """
    x = np.linspace(x_i, x_f, num)
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(x, f(x))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.annotate("", xy=(xmax, 0), xytext=(xmin, 0),
                arrowprops=dict(color='gray', width=1.5, headwidth=8, headlength=10))
    ax.annotate("", xy=(0, ymax), xytext=(0, ymin),
                arrowprops=dict(color='gray', width=1.5, headwidth=8, headlength=10))
    plt.show()
    return None


#############################################
def met_biseccion(f, a, b, tol=1e-4, n=50, verbose=True, print_error=False):
    """
    Método de bisección
    :param f: Funcion a la que se le intenta encontrar una solucion
    para la ecuacion f(x)=0, previamente definida
    :param a: límite inferior
    :param b: límite superior
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :return: solución exacta o aproximada, si tiene.
    """
    if not f(a) * f(b) < 0:
        print(f'El intervalo no funciona: f({a})={f(a):.2f}, f({b})={f(b):.2f}')
        return None
    i = 1
    lista_errores = [abs(b - a)]
    while i <= n:
        p_i = (b + a) / 2  # punto medio
        if verbose:
            print(f'ite {i:<2}: a_{i - 1:<2} = {a:.4f}, b_{i - 1:<2} = {b:.4f}, p_{i:<2} = {p_i:.5f}')

        if f(p_i) == 0:
            if print_error:
                print(f'errores x iteración: {lista_errores}')
            print('solución exacta encontrada')
            return p_i

        if f(a) * f(p_i) < 0:
            b = p_i
        else:
            a = p_i

        e_abs = abs(b - a)
        lista_errores.append(e_abs)
        if e_abs < tol:
            if print_error:
                print(f'errores x iteración: {lista_errores}')
            print(f'>>> Solución encontrada después de {i} iteraciones: x->{p_i:.15f}')
            return p_i
        i += 1

    if print_error:
        print(f'errores x iteración: {lista_errores}')
    print('solución no encontrada, iteraciones agotadas')
    return None


#############################################
def met_regula_falsi(f, a, b, tol=1e-4, n=50, verbose=True, print_error=False):
    """
    Método de regula falsi
    :param f: Funcion a la que se le intenta encontrar una solucion
    para la ecuacion f(x)=0, previamente definida
    :param a: límite inferior
    :param b: límite superior
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :return: solución exacta o aproximada, si tiene.
    """
    if not f(a) * f(b) < 0:
        print(f'El intervalo no funciona: f({a})={f(a):.2f}, f({b})={f(b):.2f}')
        return None
    i = 1
    lista_errores = [abs(b - a)]
    while i <= n:
        p_i = a - (f(a) * (b - a)) / (f(b) - f(a))  # falsa posición
        if verbose:
            print(f'ite {i:<2}: a_{i - 1:<2} = {a:.4f}, b_{i - 1:<2} = {b:.4f}, p_{i:<2} = {p_i:.5f}')

        if f(p_i) == 0:
            if print_error:
                print(f'errores x iteración: {lista_errores}')
            print('solución exacta encontrada')
            return p_i
        if f(a) * f(p_i) < 0:
            b = p_i
        else:
            a = p_i
        e_abs = abs(b - a)
        lista_errores.append(e_abs)
        if e_abs < tol:
            if print_error:
                print(f'errores x iteración: {lista_errores}')
            print(f'>>> Solución encontrada después de {i} iteraciones: x->{p_i:.15f}')
            return p_i
        i += 1

    if print_error:
        print(f'errores x iteración: {lista_errores}')
    print('solución no encontrada, iteraciones agotadas')
    return None


#############################################
def met_newton_raphson(f, df, p_0, tol=1e-4, n=50, verbose=True, print_error=False):
    """
    Método de Newton-Rapphson
    :param f: Funcion a la que se le intenta encontrar una solucion
    para la ecuacion f(x)=0, previamente definida
    :param df: Derivada de la función
    :param p_0: semilla, valor inicial
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :return: solución exacta o aproximada, si tiene.
    """
    i = 1
    lista_errores = list()
    while i<=n:
        if df(p_0) == 0:
            print('Solución no encontrada (df(x)=0)')
            return None

        p_i = p_0 - f(p_0)/df(p_0)
        e_abs = abs(p_0 - p_i)
        lista_errores.append(e_abs)
        if verbose:
            print(f'ite {i:<2}: p_{i-1:<2} = {p_0:.4f}, p_{i:<2} = {p_i:.5f}')

        if f(p_i) == 0:
            if print_error:
                print(f'errores x iteración: {lista_errores}')
            print('solución exacta encontrada')
            return p_i

        if e_abs < tol:
            if print_error:
                print(f'errores x iteración: {lista_errores}')
            print('solución encontrada')
            return p_i

        p_0 = p_i
        i += 1
    if print_error:
        print(f'errores x iteración: {lista_errores}')
    print('solución no encontrada, iteraciones agotadas')
    return None
