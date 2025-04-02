#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:18:29 2024

@author: bouteillerp
"""
from Cylindre_compression import *

for model in [CylindricalUD, Axisymetric]:
    run_test(model)
    
    
df_1 = read_csv("Cylindre_axi-results/Displacement0.csv")
resultat_axi = [df_1[colonne].to_numpy() for colonne in df_1.columns]
ur_axi = [resultat_axi[2 * i + 4] for i in range((len(resultat_axi)-4)//2)]
length = len(ur_axi)
print("longueur axi", len(ur_axi))


df_2 = read_csv("Cylindre-results/Displacement0.csv")
resultat_1D = [df_2[colonne].to_numpy() for colonne in df_2.columns]
ur_1D = [resultat_1D[i + 2] for i in range((len(resultat_1D)-2))]
print("longueur cyl", len(ur_1D))


for j in range(length):
    plt.plot(resultat_axi[0], ur_axi[j], linestyle = "--")
    plt.scatter(resultat_1D[0], ur_1D[j], marker = "x")