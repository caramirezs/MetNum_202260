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

# Implementación del interpolador de Lagrange y algunos casos de salida.

from math import *


def LagrangePol(datos):
    """
    Implementación del interpolador de Lagrange
    Entradas:
    datos -- lista de puntos (x, y) en el plano

    Salida:
    P -- función de interpolación
    """

    def L(k, x):
        """Implementación funciones L_k(x)"""
        # pol $L_k(x)=\prod\limits_{i \neq k}\frac{x-x_i}{x_k-x_i}$
        out = 1
        for i, p in enumerate(datos):
            if i != k:
                out *= (x - p[0])/(datos[k][0] - p[0])
        return out

    def P(x):
        """Implementación polinomio P(x)"""
        # polinomio $P(x)=\sum\limits_{k}f(x_k)L_k(x)$
        lag = 0
        for k, p in enumerate(datos):
            lag += p[1]*L(k, x)
        return lag

    return P


# datos para $f(x)=\frac{1}{x}$ con $x_0=2$, $x_1=2.75$ y $x_2=4$
#datosf = [(2, 1/2), (11/4, 4/11), (4, 1/4)]
#Pf = LagrangePol(datosf)
#print("Polinomio de Lagrange en x = 3:")
#print("{0:.12f}".format(Pf(3)))

# datos $g(x)=\sin(3x)$, $x_0=1$, $x_1=1.3$, $x_2=1.6$, $x_3=1.9$ y $x_4=2.2$
#datosg = [(1, 0.1411), (1.3, -0.6878), (1.6, -0.9962),(1.9, -0.5507), (2.2, 0.3115)]
#Pg = LagrangePol(datosg)
#print("Polinomio de Lagrange en x = 1.5:")
#print("{0:.12f}".format(Pg(1.5)))
