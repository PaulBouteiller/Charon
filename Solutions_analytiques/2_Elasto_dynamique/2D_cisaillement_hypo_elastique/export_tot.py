"""
Created on Thu Sep 26 17:34:25 2024
@author: bouteillerp
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def extract_columns(filename, n_col):
    # Lire le fichier ligne par ligne
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Initialiser des listes pour chaque colonne
    columns = [[] for _ in range(n_col)]
    
    for line in lines:
        if line.strip().startswith('Temps'):
            # Diviser la ligne en valeurs, en ignorant 'Temps' au début
            values = line.split()[1:]
            # Ajouter chaque valeur à la colonne correspondante
            for i, value in enumerate(values):
                columns[i].append(float(value))
    
    # Convertir les listes en arrays numpy
    columns = [np.array(col) for col in columns]
    return columns

# Utilisation de la fonction
filename = 'MaHyCo_time-history-06-06_def.txt'
result = extract_columns(filename, 16)
temps_Mahyco = result[0]
s_xx_MaHyCo = -result[10]
s_xy_MaHyCo = -result[11]
s_yy_MaHyCo = -result[12]

print("Premiere colonne", result[10])

def csv_to_numpy(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = next(reader)  # Lire une seule ligne
    
    # Convertir les chaînes en nombres flottants
    numeric_data = [float(num) for num in data]
    
    return np.array(numeric_data)

# Lire le fichier CSV et convertir en tableau NumPy
t_list = csv_to_numpy('t.csv')

def name_csv(coord, essai, old):
    if old:
        return "csv/"+coord+str(essai)+"old.csv"
    else:
        return "csv/"+coord+str(essai)+".csv"

amplitude = 0.6
s_xx = csv_to_numpy(name_csv("s_xx", amplitude, False))
s_yy = csv_to_numpy(name_csv("s_yy", amplitude, False))
s_xy = csv_to_numpy(name_csv("s_xy", amplitude, False))

s_xx_old = csv_to_numpy(name_csv("s_xx", amplitude, True))
s_yy_old = csv_to_numpy(name_csv("s_yy", amplitude, True))
s_xy_old = csv_to_numpy(name_csv("s_xy", amplitude, True))
if amplitude == 0.006:
    yinf = -0.006
    ysup = 0.018

elif amplitude == 0.06:
    yinf = -0.05
    ysup = 0.16
    
elif amplitude == 0.6:
    yinf = -0.4
    ysup = 1.3


# Créer la figure principale
fig, ax1 = plt.subplots(figsize=(12, 8))

# Tracer les courbes sur le graphique principal
ax1.plot(t_list, s_xx, linestyle="--", label=r"$\overline{s}^{NH}_{xx}$", color = "b")
ax1.plot(t_list, s_yy, linestyle="--", label="$\overline{s}^{NH}_{yy}$", color = "r")
ax1.plot(t_list, s_xy, linestyle="--", label="$\overline{s}^{NH}_{xy}$", color = "g")
ax1.plot(t_list, s_xx_old, linestyle="-", label="$\overline{s}_{xx}$", color = "b")
ax1.plot(t_list, s_yy_old, linestyle="-", label="$\overline{s}_{yy}$", color = "r")
ax1.plot(t_list, s_xy_old, linestyle="-", label="$\overline{s}_{xy}$", color = "g")

ax1.plot(temps_Mahyco, s_xx_MaHyCo, linestyle="--", label=r"$\overline{s}^{M}_{xx}$", color = "black")
ax1.plot(temps_Mahyco, s_yy_MaHyCo, linestyle="--", label="$\overline{s}^{M}_{yy}$", color = "brown")
ax1.plot(temps_Mahyco, s_xx_MaHyCo, linestyle="--", label="$\overline{s}^{M}_{xy}$", color = "yellow")

# Ajouter la ligne en pointillé gris à y=0
ax1.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

ax1.set_xlim(0, 4.2)
ax1.set_ylim(yinf, ysup)
ax1.set_xlabel(r"Temps (s)", size=20)
ax1.set_ylabel(r"Déviateurs normalisés $s/ \mu$", size=20)
ax1.legend(fontsize = 18)

# Créer le sous-graphique intégré
axins = inset_axes(ax1, width="40%", height="30%", loc='lower left', 
                   bbox_to_anchor=(0.1, 0.65, 1, 1), bbox_transform=ax1.transAxes)

# Définir la plage pour le zoom (les 10% finaux des données)
zoom_start = int(0.99 * len(t_list))
print(zoom_start)

# Tracer les courbes dans le sous-graphique
axins.plot(t_list[zoom_start:], s_xx[zoom_start:], linestyle="--", label=r"s_xx/$\mu$", color = "b")
axins.plot(t_list[zoom_start:], s_yy[zoom_start:], linestyle="--", label="s_yy", color = "r")
axins.plot(t_list[zoom_start:], s_xy[zoom_start:], linestyle="--", label="s_xy", color = "g")
axins.plot(t_list[zoom_start:], s_xx_old[zoom_start:], linestyle="-", label="s_xx_old", color = "b")
axins.plot(t_list[zoom_start:], s_yy_old[zoom_start:], linestyle="-", label="s_yy_old", color = "r")
axins.plot(t_list[zoom_start:], s_xy_old[zoom_start:], linestyle="-", label="s_xy_old", color = "g")

# Ajouter la ligne en pointillé gris à y=0 dans le sous-graphique
axins.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)

axins.set_title("Déviateurs résiduels normalisés", fontsize=12)
axins.tick_params(axis='both', which='major', labelsize=8)

# Ajuster la mise en page
plt.tight_layout()
plt.savefig("s_tot"+str(amplitude)+".pdf", bbox_inches = 'tight')
# Afficher les valeurs finales
sxx_f = s_xx[-1]
print("Le deviateur normalisé s_{xx} final est", sxx_f)
syy_f = s_yy[-1]
print("Le deviateur normalisé s_{yy} final est", syy_f)
sxy_f = s_xy[-1]
print("Le deviateur normalisé s_{xy} final est", sxy_f)

plt.show()