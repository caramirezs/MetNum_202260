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
def met_biseccion(f, a, b, tol=1e-4, n=50, verbose=True, print_error=False,
                  deploy_error=False):
    """
    Método de bisección
    :param f: Funcion a la que se le intenta encontrar una solucion
    para la ecuacion f(x)=0, previamente definida
    :param a: límite inferior
    :param b: límite superior
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :param verbose:
    :param print_error:
    :param deploy_error:
    :return: solución exacta o aproximada, si tiene.
    """
    if not f(a) * f(b) < 0:
        print(f'El intervalo no funciona: f({a})={f(a):.2f}, f({b})={f(b):.2f}')
        return None
    i = 1
    lista_errores = [abs(b - a)]
    p_0 = a
    while i <= n:
        p_i = (b + a) / 2  # punto medio
        e_abs = abs(p_0 - p_i)
        if verbose:
            print(f'ite {i:<2}: a_{i - 1:<2} = {a:.4f}, b_{i - 1:<2} = {b:.4f}, p_{i:<2} = {p_i:.5f}'
                  f' | e_abs: {e_abs:e}')

        if f(p_i) == 0:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución exacta encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i

        if f(a) * f(p_i) < 0:
            b = p_i
        else:
            a = p_i
        p_0 = p_i
        lista_errores.append(e_abs)
        if e_abs < tol:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i
        i += 1

    if print_error:
        print(f'error x iteración: {lista_errores}')
    print('solución no encontrada, iteraciones agotadas')
    return None


#############################################
def met_regula_falsi(f, a, b, tol=1e-4, n=50, verbose=True, print_error=False,
                     deploy_error=False):
    """
    Método de regula falsi
    :param f: Funcion a la que se le intenta encontrar una solucion
    para la ecuacion f(x)=0, previamente definida
    :param a: límite inferior
    :param b: límite superior
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :param verbose:
    :param print_error:
    :param deploy_error:
    :return: solución exacta o aproximada, si tiene.
    """
    if not f(a) * f(b) < 0:
        print(f'El intervalo no funciona: f({a})={f(a):.2f}, f({b})={f(b):.2f}')
        return None
    i = 1
    lista_errores = [abs(b - a)]
    p_0 = a
    while i <= n:
        p_i = a - (f(a) * (b - a)) / (f(b) - f(a))  # falsa posición
        e_abs = abs(p_0 - p_i)
        if verbose:
            print(f'ite {i:<2}: a_{i - 1:<2} = {a:.4f}, b_{i - 1:<2} = {b:.4f}, p_{i:<2} = {p_i:.5f}'
                  f' | e_abs: {e_abs:e}')

        if f(p_i) == 0:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución exacta encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i
        if f(a) * f(p_i) < 0:
            b = p_i
        else:
            a = p_i
        p_0 = p_i
        lista_errores.append(e_abs)
        if e_abs < tol:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i
        i += 1

    if print_error:
        print(f'error x iteración: {lista_errores}')
    print('solución no encontrada, iteraciones agotadas')
    return None


#############################################
def met_newton_raphson(f, df, p_0, tol=1e-4, n=50, verbose=True, print_error=False,
                       deploy_error=False):
    """
    Método de Newton-Rapphson
    :param f: Funcion a la que se le intenta encontrar una solucion
    para la ecuacion f(x)=0, previamente definida
    :param df: Derivada de la función
    :param p_0: semilla, valor inicial
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :param verbose:
    :param print_error:
    :param deploy_error:
    :return: solución exacta o aproximada, si tiene.
    """
    i = 1
    lista_errores = list()
    while i <= n:
        if df(p_0) == 0:
            print('Solución no encontrada (df(x)=0)')
            return None

        p_i = p_0 - f(p_0) / df(p_0)
        e_abs = abs(p_0 - p_i)
        lista_errores.append(e_abs)
        if verbose:
            print(f'ite {i:<2}: p_{i - 1:<2} = {p_0:.4f}, p_{i:<2} = {p_i:.5f}'
                  f' | e_abs: {e_abs:e}')

        if f(p_i) == 0:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución exacta encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i

        if e_abs < tol:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i

        p_0 = p_i
        i += 1
    if print_error:
        print(f'error x iteración: {lista_errores}')
    print('solución no encontrada, iteraciones agotadas')
    return None


#############################################
def met_secante(f, p_m, p_n, tol=1e-4, n=50, verbose=True, print_error=False,
                deploy_error=False):
    """
    Método de la Secante
    :param f: Funcion a la que se le intenta encontrar una solucion
    para la ecuacion f(x)=0, previamente definida
    :param p_m: semilla 1
    :param p_n: semilla 2
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :param verbose:
    :param print_error:
    :param deploy_error:
    :return: solución exacta o aproximada, si tiene.
    """
    i = 1
    lista_errores = list()
    while i <= n:
        if f(p_m) == f(p_n):
            print('Solución no encontrada (recta secante horizontal)')
            return None

        p_i = p_n - (f(p_n) * (p_m - p_n)) / (f(p_m) - f(p_n))

        e_abs = abs(p_n - p_i)
        lista_errores.append(e_abs)
        if verbose:
            print(f'ite {i:<2}: [p_{i - 2:<2}={p_m:.5f}, p_{i - 1:<2}={p_n:.5f}] p_{i:<2}={p_i:.7f}'
              f' | e_abs: {e_abs:e}')
        if f(p_i) == 0:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución exacta encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i

        if e_abs < tol:
            if print_error:
                print(f'error x iteración: {lista_errores}')
            print(f'>>> Solución encontrada después de {i} iteraciones: x->{p_i:.15f}')
            if deploy_error:
                return lista_errores, p_i
            else:
                return p_i

        p_m = p_n
        p_n = p_i

        i += 1
    if print_error:
        print(f'error x iteración: {lista_errores}')
    print('solución no encontrada, iteraciones agotadas')
    return None


#############################################
def met_punto_fijo(g, p_0, tol=1e-4, n=200, verbose=True, print_error=False,
                   deploy_error=False):
    """
    Método del punto fijo
    :param g: Funcion transformada -> punto fijo
    :param p_0: semilla
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :param verbose:
    :param print_error:
    :param deploy_error:
    :return: solución exacta o aproximada, si tiene.
    """
    i = 1
    lista_errores = list()
    while i <= n:
        try:
            p_i = g(p_0)
            e_abs = abs(p_0 - p_i)
            lista_errores.append(e_abs)
            if verbose:
                print(f'ite {i:<2}: p_{i - 1:<2}={p_0:.5f} p_{i:<2}={p_i:.7f}'
                  f' | e_abs: {e_abs:e}')
            if g(p_i) == p_i:
                if print_error:
                    print(f'error x iteración: {lista_errores}')
                print(f'>>> Solución exacta encontrada después de {i} iteraciones: x->{p_i:.15f}')
                if deploy_error:
                    return lista_errores, p_i
                else:
                    return p_i

            if e_abs < tol:
                if print_error:
                    print(f'error x iteración: {lista_errores}')
                print(f'>>> Solución encontrada después de {i} iteraciones: x->{p_i:.15f}')
                if deploy_error:
                    return lista_errores, p_i
                else:
                    return p_i

        except OverflowError:
            print('solución no encontrada, OverflowError')
            return None
        p_0 = p_i
        i += 1
    print('solución no encontrada, iteraciones agotadas')
    return None


######################################
def estudio_error(listas_errores, orden=1):
    lista_consolidado = list()
    list_len = list()
    first_row = list()
    for lista in listas_errores:
        first_row.append(lista[0])
        first_row.append('----')
        list_len.append(len(lista))
    max_len = max(list_len)
    lista_consolidado.append(first_row)

    for i in range(1, max_len):
        new_row = list()
        for lista in listas_errores:
            try:
                new_row.append(lista[i])
                new_row.append(round(lista[i] / lista[i-1]**orden, 4))
            except IndexError:
                new_row.append('----')
                new_row.append('----')
        lista_consolidado.append(new_row)
    df = pd.DataFrame(lista_consolidado)
    return df

