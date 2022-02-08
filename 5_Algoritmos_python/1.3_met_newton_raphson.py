############################################################
## Método de Newton-Raphson
def met_newton_raphson(f, df, p_0, tol=10**-4, n=50):
    """
    Método Newton Raphson
    :param f: Funcion a la que se le intenta encontrar una solucion para la ecuacion f(x)=0, previamente definida
    :param df: derivada de la función
    :param p_0: semilla (punto inicial)
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :return: solución exacta o aproximada, si tiene.
    """ 
    print('semilla: p_0={:.2f}'.format(p_0))
    i = 1
    while i <= n:
        if df(p_0) == 0:
            print('Solución no encontrada (df(x)=0)')
            return None
        p_1 = p_0 - f(p_0)/df(p_0)
        e_abs = abs(p_1 - p_0)
        print('ite {:<2}: p_{:<2}={:.7f} , e_abs={:.10}'.format(i,i,p_1,e_abs))
        if e_abs < tol:
            print('Solución aproximada encontrada')
            return p_1
        p_0 = p_1
        i += 1
    print('Solución no encontrada, iteraciones agotadas')
    return None    