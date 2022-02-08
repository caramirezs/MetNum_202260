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

# Implementación del método RK4 y algunos casos de salida.

from math import *


def test1(t, y):  # $y'=y-t^2+1$
    """Función de prueba"""
    return y - t**2 + 1


def test2(t, y):  # $y'=2-e^{-4t}-2y$
    """Función de prueba"""
    return 2 - exp(-4*t) - 2*y


def RK4(a, b, y0, f, N):
    """
    Implementación método RK4
    Entradas:
    a -- inicio intervalo
    b -- fin intervalo
    y0 -- aproximación inicial
    f -- función
    N -- pasos

    Salida:
    w -- aproximación final
    """
    h = (b - a)/N
    t = a
    w = y0
    print("t0  = {0:.2f}, w0  = {1:.12f}".format(t, w))

    for i in range(1, N+1):
        k1 = f(t, w)
        k2 = f(t + h/2, w + h*k1/2)
        k3 = f(t + h/2, w + h*k2/2)
        k4 = f(t + h, w + h*k3)
        w = w + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = a + i*h
        print("t{0:<2} = {1:.2f}, w{0:<2} = {2:.12f}".format(i, t, w))
    return w


# $\frac{dy}{dt}=y-t^2+1$, $a=0$, $b=2$, $y_0=0.5$, $N=10$
print("Método RK4:")
RK4(0, 2, 0.5, test1, 10)

# $\frac{dy}{dt}=2-e^{-4t}-2y$, $a=0$, $b=1$, $y_0=1$, $N=20$
print("Método RK4:")
RK4(0, 1, 1, test2, 20)
