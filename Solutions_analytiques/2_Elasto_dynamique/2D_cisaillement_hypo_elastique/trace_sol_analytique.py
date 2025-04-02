#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 11:57:55 2024

@author: bouteillerp
"""
from math import cos, sin, log
from ufl import ln
from numpy import linspace
import matplotlib.pyplot as plt
U_f = 0.6

def s_11_norm(S, U):
    return 1 - cos(S) + (1 + ln(1 + U))*((1 - cos(S/(1+U))) * cos(S) - sin(S/(1+U)) * sin(S))

def s_12_norm(S, U):
    return -sin(S) + (1 + ln(1 + U)) * (sin(S/(1+U)) * cos(S) + (1 - cos(S/(1+U))) * sin(S))

U_array = linspace(0, U_f, 100)
s_11_list_03 = [s_11_norm(0.3, U) for U in U_array]
s_12_list_03 = [s_12_norm(0.3, U) for U in U_array]
s_11_list_06 = [s_11_norm(0.6, U) for U in U_array]
s_12_list_06 = [s_12_norm(0.6, U) for U in U_array]

plt.xlim(0, U_f)
plt.plot(U_array, s_11_list_03, linestyle = "-", label = r"$s_{11}/\mu,\overline{S}=0.3$", color = "green")
plt.plot(U_array, s_12_list_03, linestyle = "-", label = r"$s_{12}/\mu,\overline{S}=0.3$", color = "red")
plt.plot(U_array, s_11_list_06, linestyle = "--", label = r"$s_{11}/\mu,\overline{S}=0.6$", color = "green")
plt.plot(U_array, s_12_list_06, linestyle = "--", label = r"$s_{12}/\mu,\overline{S}=0.6$", color = "red")



print("s_xx", s_11_norm(0.6, 0.6))
print("s_xy", s_12_norm(0.6, 0.6))


Uf_charon = [0.05 * i for i in range(1,13)]
solution_charon_0_6_s_xx = [-0.008164105713174741, -0.015086340910192089, 
                            -0.02098825337370215, -0.02600089974044599, 
                            -0.03023846953505393, -0.03380013721054802, 
                            -0.03677190584442844, -0.03942243938355922, 
                            -0.041234050268864356, -0.04300759753324445, 
                            -0.044110494528197396, -0.04520148366714268]
solution_charon_0_6_s_xy = [-0.002413281210979769, -0.005899820762272771, 
                            -0.010193740321351208, -0.015089984014181422, 
                            -0.020429573753140598, -0.02608871363687067, 
                            -0.03197066567151618, -0.03798820796115054, 
                            -0.044116164198933897, -0.05024771288558408, 
                            -0.05643514646401167, -0.06252878568680502]

# plt.scatter(Uf_charon, solution_charon_0_6_s_xx, marker = "x", color = "black")
# plt.scatter(Uf_charon, solution_charon_0_6_s_xy, marker = "x", color = "black", label = "CHARON")


Uf_MaHyCo = [0.1 * i for i in range(1,7)]
solution_MaHyCo_0_6_s_xx = [-0.0152946657635867, -0.0262038070649965,
                            -0.0339996633969432, -0.0394262210465291,
                            -0.0430429293133982, -0.0452712530819504]
solution_MaHyCo_0_6_s_xy = [-0.00590634366482316, -0.015102668790344,
                            -0.0261075135455321, -0.0380246900404858,
                            -0.050305077312655, -0.0626105842247727]


solution_MaHyCo_0_3_s_xx = [-0.00391828213032344, -0.00673128971851388,
                            -0.00875726566652038, -0.0101823698490219,
                            -0.0111467995145736, -0.0117564161032867]
solution_MaHyCo_0_3_s_xy = [-0.00170325770975017, -0.00521340827848115,
                            -0.0097929224589606, -0.014982836916199,
                            -0.020492231188816, -0.0261344794035456]


plt.scatter(Uf_MaHyCo, solution_MaHyCo_0_3_s_xx, marker = "x", color = "black")
plt.scatter(Uf_MaHyCo, solution_MaHyCo_0_3_s_xy, marker = "x", color = "black")

plt.scatter(Uf_MaHyCo, solution_MaHyCo_0_6_s_xx, marker = "x", color = "black")
plt.scatter(Uf_MaHyCo, solution_MaHyCo_0_6_s_xy, marker = "x", color = "black", label = "FEM")



#%%


# import math
# X= [0 + 0.01*x for x in range (61) ]
# # fig = plt.figure(figsize = (16, 7))
# Sf=0.6
# h=1.
# Sb=Sf/h
# Y1 = [ (1-math.cos(Sb) + (1+math.log(1.+x/h))*((1-math.cos((Sb)/(1.+x/h)))*math.cos(Sb) - math.sin((Sb)/(1.+x/h))*math.sin(Sb))) for x in X ]
# plt.plot(X, Y1, 'b', label ='T11-S=0.6')
# Y2 = [ -math.sin(Sb) + (1.+math.log(1.+ x/h))*(math.sin(Sb/(1.+x/h))*math.cos(Sb)+(1-math.cos(Sb/(1.+x/h)))*math.sin(Sb)) for x in X ]
# plt.plot(X, Y2, 'g', label ='T12-S=0.6')
# Sf=0.3
# Sb=Sf/h
# Y1 = [ (1-math.cos(Sb) + (1+math.log(1.+x/h))*((1-math.cos((Sb)/(1.+x/h)))*math.cos(Sb) - math.sin((Sb)/(1.+x/h))*math.sin(Sb))) for x in X ]
# plt.plot(X, Y1, 'r', label ='T11-S=0.3')
# Y2 = [ -math.sin(Sb) + (1.+math.log(1.+ x/h))*(math.sin(Sb/(1.+x/h))*math.cos(Sb)+(1-math.cos(Sb/(1.+x/h)))*math.sin(Sb)) for x in X ]
# plt.plot(X, Y2, 'k', label ='T12-S=0.3')


plt.xlabel(r"Vertical Displacement $\overline{U}$", size = 18)
# plt.xlabel(r"Déplacement vertical $\overline{U}$", size = 18)
plt.ylabel(r"Normalized Shear", size = 18)
# plt.ylabel(r"Cisaillement normalisé", size = 18)
plt.legend()
plt.savefig("s_res.pdf", bbox_inches = 'tight')
plt.show()