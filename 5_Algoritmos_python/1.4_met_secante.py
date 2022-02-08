############################################################
## Método de la secante

def met_secante(f, p_n, p_m, tol=10**-4, n=50):
    """
    Método de la secante
    :param f: Funcion a la que se le intenta encontrar una solucion para la ecuacion f(x)=0, previamente definida
    :param p_0: semilla (punto inicial 1)
    :param p_1: semilla (punto inicial 2)
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :return: solución exacta o aproximada, si tiene.
    """
    i = 1
    print('semilla : p_{:<2}={:.7f}'.format(-1,p_n))
    print('semilla : p_{:<2}={:.7f}'.format(0,p_m))
    while i <= n:
        if f(p_n) - f(p_m) == 0:  # división por cero
            print('Solución no encontrada (error en puntos iniciales)')
            return None
        p_1 = p_n - f(p_n)*(p_m - p_n)/(f(p_m) - f(p_n))  
        e_abs = abs(p_1 - p_n)  # error absoluto
        print('ite {:<2}: p_{:<2}={:.7f} , e_abs={:.7f}'.format(i,i,p_1,e_abs))
        if e_abs < tol:  # criterio de parada
            print('Solución aproximada encontrada')
            return p_1
        p_n = p_m
        p_m = p_1
        i += 1
    print ('Solución no encontrada, iteraciones agotadas: {}'.format(i-1))
    return None         