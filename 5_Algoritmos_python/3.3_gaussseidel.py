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

# Implementación del método de Gauss-Seidel y algunos casos de salida.

from math import *
from pprint import pprint


def distinf(x, y):
    """Implementación distancia dada por la norma infinito"""
    return max([abs(x[i]-y[i]) for i in range(len(x))])


def GaussSeidel(A, b, x0, TOL, MAX):
    """
    Implementación del método de Gauss-Seidel
    Entradas:
    A -- matriz cuadrada
    b -- vector
    x0 -- aproximación inicial
    TOL -- tolerancia
    MAX -- número máximo de iteraciones

    Salida:
    x -- aproximación a solución del sistema Ax = b
    None -- en caso de agotar las iteraciones o presentar errores
    """
    n = len(A)
    x = [0.0 for x in range(n)]
    k = 1
    while k <= MAX:
        for i in range(n):
            if abs(A[i][i]) <= 1e-15:
                print("Imposible iterar")
                return None
            s1 = sum([A[i][j]*x[j] for j in range(i)])
            s2 = sum([A[i][j]*x0[j] for j in range(i+1, n)])
            x[i] = (b[i] - s1 - s2)/A[i][i]
        pprint(x)
        if distinf(x, x0) < TOL:
            print(r"Solución encontrada")
            return x
        k += 1
        for i in range(n):
            x0[i] = x[i]
    print("Iteraciones agotadas")
    return None


#A = [[2, 1], [5, 7]]
#b = [11, 13]
#x0 = [1, 1]
#print("Matriz A:")
#pprint(A)
#print("Vector b:")
#pprint(b)
#print("Semilla x0:")
#pprint(x0)
#print("Iteración de Gauss-Seidel")
# $TOL = 10^{-5}$, $MAX=50$
#GaussSeidel(A, b, x0, 1e-5, 50)


#A = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
#b = [6, 25, -11]
#x0 = [0, 0, 0]
#print("Matriz A:")
#pprint(A)
#print("Vector b:")
#pprint(b)
#print("Semilla x0:")
#pprint(x0)
#print("Iteración de Gauss-Seidel")
# $TOL = 10^{-10}$, $MAX=50$
#GaussSeidel(A, b, x0, 1e-10, 50)
