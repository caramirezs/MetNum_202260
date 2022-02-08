def met_biseccion(f, a, b, tol=10**-4, n=50):
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
    if not f(a)*f(b)<0:
        print('El intervalo no funciona: f({})={:.2f}, f({})={:.2f}'.format(a, f(a), b, f(b)))
        return None
    i = 1
    while i<=n:
        p_i = (b + a)/2  # punto medio
        print('ite {:<2}: a_{:<2} = {:.4f}, b_{:<2} = {:.4f}, p_{:<2} = {:.5f}'.format(i, i-1, a, i-1, b, i, p_i))
        if f(p_i) == 0:
            print('solución exacta encontrada: {}'.format(p_i))
            return p_i
        if f(a)*f(p_i) < 0:
            b = p_i
        else:
            a = p_i
        e_abs = abs(b - a)
        if e_abs < tol:
            print('solución encontrada: {:.5f}'.format(p_i))
            return p_i
        i += 1
    print('solución no encontrada, iteraciones agotadas')
    return None