############################################################
## Método de punto fijo

def met_puntofijo(g, p_n, tol=10**-4, n=50):
    """
    Método de punto fijo
    :param g: Función 'especial' x=g(x)
    :param p_0: semilla (punto inicial)
    :param tol: toleracia, criterio de parada
    :param n: número máximo de iteraciones, criterio de parada
    :return: solución exacta o aproximada, si tiene.
    """
    i = 1
    print('semilla : p_{:<2} = {:.7f}'.format(0,p_n))
    while i <= n:
        try:
            p_1 = g(p_n)
            e_abs = abs(p_1 - p_n)  # error absoluto
            print('ite {:<2}: p_{:<2}={:.7f} , e_abs={:.7f}'.format(i,i,p_1,e_abs))
            if e_abs < tol:  # criterio de parada
                print('Solución aproximada encontrada')
                return p_1
            p_n = p_1
            i += 1
        except OverflowError:
            print('La función g no sirve')
            return None
    print ('Solución no encontrada, iteraciones agotadas: {}'.format(i-1))
    return None 