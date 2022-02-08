#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
# Compendio de programas.
# Matemáticas para Ingeniería. Métodos numéricos con Python.
# Copyright (C) 2020 Los autores del texto.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
# ---------------------------------------------------------------------

# Implementación del método LU y algunos casos de salida.

from math import *
from pprint import pprint
from sympy import *


def lu(A):
    """
    Implementación del método LU
    Entradas:
    A -- matriz cuadrada

    Salidas:
    L, U -- matrices de la descomposición
    None -- en caso de no ser posible la descomposición
    """
    n = len(A)
    # crear matrices nulas
    L = [[0 for x in range(n)] for x in range(n)]
    U = [[0 for x in range(n)] for x in range(n)]

    # Doolittle
    L[0][0] = 1
    U[0][0] = A[0][0]

    if abs(L[0][0]*U[0][0]) <= 1e-15:
        print("Imposible descomponer")
        return None

    for j in range(1, n):
        U[0][j] = A[0][j]/L[0][0]
        L[j][0] = A[j][0]/U[0][0]

    for i in range(1, n-1):
        L[i][i] = 1
        s = sum([L[i][k]*U[k][i] for k in range(i)])
        U[i][i] = A[i][i] - s

        if abs(L[i][i]*U[i][i]) <= 1e-15:
            print("Imposible descomponer")
            return None

        for j in range(i+1, n):
            s1 = sum([L[i][k]*U[k][j] for k in range(i)])
            s2 = sum([L[j][k]*U[k][i] for k in range(i)])
            U[i][j] = A[i][j] - s1
            L[j][i] = (A[j][i] - s2)/U[i][i]

    L[n-1][n-1] = 1.0
    s3 = sum([L[n-1][k]*U[k][n-1] for k in range(n)])
    U[n-1][n-1] = A[n-1][n-1] - s3

    if abs(L[n-1][n-1]*U[n-1][n-1]) <= 1e-15:
        print("Imposible descomponer")
        return None

    print("Matriz L:\n")
    pprint(Matrix(L))
    print("Matriz U:\n")
    pprint(Matrix(U))
    return L, U


#A = [[4, 3], [6, 3]]
#print("Matriz A:")
#pprint(A)
#lu(A)

#A = [[0, 1], [1, 1]]
#print("Matriz A:")
#pprint(A)
#lu(A)

#A = [[3, 1, 6], [-6, 0, -16], [0, 8, -17]]
#print("Matriz A:")
#pprint(A)
#lu(A)

#A = [[1, 2, 3], [2, 4, 5], [1, 3, 4]]
#print("Matriz A:")
#pprint(A)
#lu(A)
