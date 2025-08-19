#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:28:36 2024

@author: bouteillerp
"""
from math import log, exp
from numpy import linspace
#Comparaison équation d'état hyper-élastique.
kappa = 10
def p1(J):
    return -(J-1)

def p2(J):
    return -log(J)/J

def p3(J):
    return (p1(J)+p2(J))/2

def p5(J):
    return -log(J)

def p7(J):
    return -1./2 * (exp(J-1) - 1/J)

def p8(J):
    return -1./2 * (log(J) - 1/J + 1)

J_list = linspace(1./3, 3, 100)
p1_list = []
p2_list = []
p3_list = []
p5_list = []
p7_list = []
p8_list = []
for J in J_list:
    p1_list.append(p1(J))
    p2_list.append(p2(J))
    p3_list.append(p3(J))
    p5_list.append(p5(J))
    p7_list.append(p7(J))
    p8_list.append(p8(J))

import matplotlib.pyplot as plt
plt.plot(J_list, p1_list, linestyle = "--", label = "p1")
plt.plot(J_list, p2_list, linestyle = "--", label = "p2")
plt.plot(J_list, p3_list, linestyle = "--", label = "p3")
plt.plot(J_list, p5_list, linestyle = "--", label = "p5")
plt.plot(J_list, p7_list, linestyle = "--", label = "p7")
plt.plot(J_list, p8_list, linestyle = "--", label = "p8")
plt.xlim(1./3,3)
plt.legend()
plt.xlabel(r"Dilatation volumique J", size = 18)
plt.ylabel(r"Pression normalisée $p/\kappa$ ", size = 18)
plt.axhline(linewidth=1, color='grey')
plt.axvline(x = 1, linewidth=1, color='grey')
plt.savefig("../U_EOS.pdf")
plt.show()
    