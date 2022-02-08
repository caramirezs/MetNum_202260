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

# Implementación del método de Verlet y algunos casos de salida.

from math import *


def test1(x):  # $y''=x$
    """Función de prueba"""
    return x


def test2(x):  # $y''=-x$
    """Función de prueba"""
    return -x


def Verlet(a, b, x0, v0, f, N):
    """
    Implementación método de Verlet
    Entradas:
    a -- inicio intervalo
    b -- fin intervalo
    x0 -- aproximación inicial
    v0 -- aproximación inicial
    f -- función
    N -- pasos

    Salida:
    p1 -- aproximación final
    """
    h = (b - a)/N
    p0 = x0
    p1 = p0 + v0*h + 0.5*f(p0)*h**2
    print("a0  = {0:0.2f}, p0  = {1:0.12f}".format(a, p0))
    for i in range(1, N+1):
        p = 2*p1 - p0 + f(p1)*h**2
        s = a + i*h
        print("a{0:<2} = {1:0.2f}, p{0:<2} = {2:.12f}".format(i, s, p1))
        p0 = p1
        p1 = p
    return p1


# $\frac{d^2x}{dt^2}=x$, $a=0$, $b=1$, $x_0=1$, $v_0=1$, $N=20$
print("Método de Verlet:")
Verlet(0, 1, 1, 1, test1, 20)

# $\frac{d^2x}{dt^2}=-x$, $a=0$, $b=1$, $x_0=1$, $v_0=0$, $N=20$
print("Método de Verlet:")
Verlet(0, 1, 1, 0, test2, 20)
