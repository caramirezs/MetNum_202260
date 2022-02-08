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

# Implementación de la recta de mínimos cuadrados.

from math import *


def RectaMinSq(datos):
    """
    Implementación recta de mínimos cuadrados
    Entradas:
    datos -- lista de puntos (x, y) en el plano

    Salida:
    P -- recta de mínimos cuadrados
    """
    X = sum([p[0] for p in datos])
    Y = sum([p[1] for p in datos])
    XX = sum([(p[0])**2 for p in datos])
    XY = sum([p[0]*p[1] for p in datos])
    m = len(datos)

    def P(x):
        """Recta de mínimos cuadrados"""
        a0 = (Y*XX - X*XY)/(m*XX - X**2)
        a1 = (m*XY - X*Y)/(m*XX - X**2)
        return a0 + a1*x

    return P


def ErrorSq(f, datos):
    """Calcular error cuadrático"""
    E = sum([(p[1] - f(p[0]))**2 for p in datos])
    return E


# datos de prueba
#datos = [(-1, 2), (0, -1), (1, 1), (2, -2)]
#f = RectaMinSq(datos)
#print("Recta de ajuste. Evaluar en x = 1:")
#print("{0:.10f}".format(f(1.0)))

# datos de prueba
#datos = [(1.0, 1.3), (2.0, 3.5), (3.0, 4.2), (4.0, 5.0), (5.0, 7.0),
#         (6.0, 8.8), (7.0, 10.1), (8.0, 12.5), (9.0, 13.0)]
#f = RectaMinSq(datos)
#print("Recta de ajuste. Evaluar en x = 1:")
#print("{0:.10f}".format(f(1.0)))
