def met_regula_falsi(f, a, b, tol=10**-4, n=50):
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
    if not f(a)*f(b)<0:
        print('El intervalo no funciona: f({})={:.2f}, f({})={:.2f}'.format(a, f(a), b, f(b)))
        return None
    p_n = a - ((f(a)*(b-a))/(f(b)-f(a)))
    print('ite 1 : a_0 ={:.5f}, b_0 ={:.5f}, p_1 ={:.5f}, e_abs=---------'.format(a, b, p_n))
    i = 2
    while i <= n:
        if f(p_n) == 0:
            print('Solución exacta encontrada')
            return p_n        
        p_a = p_n        
        if f(a)*f(p_n) < 0:
            b = p_n
        else:
            a = p_n            
        p_n = a - ((f(a)*(b-a))/(f(b)-f(a)))         
        e_abs = abs(p_n - p_a)  # Error absoluto         
        print('ite {:<2}: a_{:<2}={:.5f}, b_{:<2}={:.5f}, p_{:<2}={:.5f}, e_abs={:.7f}'.format(i, i-1, a, i-1, b, i, p_n, e_abs))
        if e_abs < tol:
            print('Solución aproximada encontrada')
            return p_n        
        i += 1
    print('Solución no encontrada, iteraciones agotadas')
    return None