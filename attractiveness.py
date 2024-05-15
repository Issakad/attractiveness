#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image

#image = Image.open('logo.PNG')
#st.sidebar.image(image, width=200)
from pathlib import Path

# Assurez-vous que le fichier est référencé correctement
#st.sidebar.image("logo.PNG")

#import seaborn as sns
import os
#print(os.listdir("C:\\Users\\Utilisateur\\Desktop\\Collaboration\\Recherche académique\\Projets d'articles\\Fiscalité locale\\Dashboard\\"))
#print(os.listdir("C:\\Users\\Utilisateur\\Desktop\\Data science\\Openclassrooms\\Projet 7\\data\\"))

# Chargement des données de démonstration
data = pd.read_excel("C:\\Users\\Utilisateur\\Desktop\\Collaboration\\Recherche académique\\Projets d'articles\\Fiscalité locale\\Dashboard\\data2019.xlsx") # Assurez-vous de remplacer "data.csv" par le chemin de votre fichier de données
# Ajout de l'index à la DataFrame data


# Titre du dashboard
# Sélectionner aléatoirement 100 lignes de votre DataFrame
random_rows = data.sample(n=100, random_state=42)

# Titre du dashboard
st.title("Dashboard d'attractivité des communes d'Outre-Mer français")
st.subheader("Travail réalisé par les chercheurs du CREDDI")
# Sélection de l'année
annee_selectionnee = st.sidebar.selectbox("Sélectionner l'année", sorted(data['Annee'].unique()))
# Afficher les 100 lignes sélectionnées aléatoirement
st.subheader("Données utilisées")
#st.write("100 lignes sélectionnées aléatoirement du tableau :")
#st.dataframe(random_rows)



# Ajouter une boîte de sélection dans la barre latérale
show_data = st.sidebar.selectbox('Afficher les données utilisées', ('Non', 'Oui'))

# Afficher les données si l'utilisateur sélectionne 'Oui'
if show_data == 'Oui':
    st.write("100 lignes sélectionnées aléatoirement du tableau :")
    st.dataframe(random_rows)


# Sélection du nom de la commune
commune_selectionnee = st.sidebar.selectbox("Sélectionner une commune", sorted(data['NomCommune'].unique()))

# Filtrer les données pour ne conserver que celles de la commune sélectionnée
donnees_commune = data[data['NomCommune'] == commune_selectionnee]

# Afficher les données de la commune sélectionnée
st.subheader("Les données de la commune sélectionnée")
st.write(f"Données pour la commune sélectionnée ({commune_selectionnee}) :")
st.write(donnees_commune)

# Sélectionner les données de la commune sélectionnée
commune_selected_data = data[data['NomCommune'] == commune_selectionnee].iloc[0]
#commune_selected_data = st.selectbox('Sélectionnez votre commune', data['CodeCommune'].unique())
# Extraire les valeurs des différentes variables pour la commune sélectionnée
# Afficher la liste des colonnes de commune_selected_data
#st.write(data.columns)

commune_selected_data=data.copy()


# Extraire les valeurs des différentes variables pour la commune sélectionnée
dynamisme_economique = commune_selected_data['Economie']
dependance_vieillesse = commune_selected_data['Demographie']
pression_fiscale = commune_selected_data['Fiscalite']
chomage = commune_selected_data['Chomage']
gouvernance = commune_selected_data['Gouvernance']
biens_publics = commune_selected_data['Goods']

# Calcul du score d'attractivité selon la formule donnée
score_attractivite = (3 ** 0.5) / 4 * (dynamisme_economique * dependance_vieillesse +
                                       dependance_vieillesse * pression_fiscale +
                                       pression_fiscale * chomage +
                                       chomage * gouvernance +
                                       gouvernance * biens_publics +
                                       biens_publics * dynamisme_economique)

# Création d'une nouvelle Series avec les noms des communes comme index
scores_communes = pd.Series(score_attractivite.values, index=data['NomCommune'])

# Extraction du score de la commune sélectionnée
score_commune_selectionnee = scores_communes.get(commune_selectionnee)
st.subheader("Le score d'attractivité de la commune sélectionnée et son classement")
# Affichage du score d'attractivité de la commune sélectionnée avec 3 chiffres après la virgule
if score_commune_selectionnee is not None:
    st.write(f"Le score d'attractivité de la commune sélectionnée ({commune_selectionnee}) est: {score_commune_selectionnee:.3f}")
else:
    st.write(f"Aucun score d'attractivité trouvé pour la commune sélectionnée ({commune_selectionnee})")
# Trier les scores des communes par ordre décroissant
scores_communes_sorted = scores_communes.sort_values(ascending=False)

# Trouver le rang de la commune sélectionnée
rang_commune_selectionnee = scores_communes_sorted.index.get_loc(commune_selectionnee) + 1

# Afficher le rang de la commune sélectionnée
st.write(f"Le rang de la commune sélectionnée ({commune_selectionnee}) selon son score est: {rang_commune_selectionnee}")
# Définir les seuils pour chaque catégorie
seuil_faible = 0.333
seuil_moyen = 0.67

# Déterminer la catégorie de la commune sélectionnée
if score_commune_selectionnee <= seuil_faible:
    categorie_commune = "faiblement attractive 🏢  ⭐⭐"
elif score_commune_selectionnee <= seuil_moyen:
    categorie_commune = " moyennement attractive 🏢  ⭐⭐⭐"
else:
    categorie_commune = "fortement attractive 🏢  ⭐⭐⭐⭐⭐"

# Afficher la catégorie de la commune sélectionnée
st.write(f"La commune sélectionnée ({commune_selectionnee}) est {categorie_commune}.")

st.subheader("Les facteurs explicatifs du score d'attractivité de la commune sélectionnée")
#Dimensions de l'indice
dimensions = ['Dynamisme économique', 'Dépendance vieillesse', 'Pression fiscale', 'Chômage', 'Gouvernance', 'Biens publics']
# Calculer les angles pour chaque dimension
num_dimensions = len(dimensions)
angles = np.linspace(0, 2 * np.pi, num_dimensions, endpoint=False).tolist()
angles += angles[:1]  # Répéter le premier angle pour fermer le graphique

# Convertir les noms de commune en minuscules dans la DataFrame
data['NomCommune'] = data['NomCommune'].str.lower()

# Convertir le nom de la commune sélectionnée en minuscules
commune_selectionnee = commune_selectionnee.lower()

# Sélectionner les données de la commune sélectionnée
commune_selected_data = data.set_index('NomCommune').loc[commune_selectionnee]

# Récupérer les valeurs pour la commune sélectionnée
values = [
    commune_selected_data['Economie'],
    commune_selected_data['Demographie'],
    commune_selected_data['Fiscalite'],
    commune_selected_data['Chomage'],
    commune_selected_data['Gouvernance'],
    commune_selected_data['Goods']
]




values += values[:1]  # Répéter la première valeur pour fermer le graphique
#st.write(values )
# Convertir les valeurs des dimensions en float
#values = [dynamisme_economique.values[0], dependance_vieillesse.values[0], pression_fiscale.values[0], chomage.values[0], gouvernance.values[0], biens_publics.values[0]]

# Répéter la première valeur pour fermer le graphique
#values += values[:1]

# Créer le graphique en radar
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.fill(angles, values, color='blue', alpha=0.2)  # Remplir la zone du graphique
ax.plot(angles, values, color='blue', linewidth=1)  # Tracer les lignes du graphique
ax.set_yticklabels([])  # Masquer les étiquettes sur l'axe y
ax.set_xticks(angles[:-1])  # Positionner les étiquettes des axes

# Ajouter les noms des axes
ax.set_xticklabels(dimensions, fontsize=8)

# Afficher les valeurs des étiquettes
for angle, value, dimension in zip(angles[:-1], values[:-1], dimensions):
    ax.text(angle, value, f'{value:.2f}', ha='center', va='bottom')

# Afficher le graphique
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

st.subheader("Classement de toutes les communes")
# Sélectionner les colonnes nécessaires pour le tableau
table_data = data[['NomCommune', 'Nom2022Département']]
table_data['Score'] = score_attractivite
# Ajouter une colonne "Classement" en fonction des scores d'attractivité
table_data['Classement'] = table_data['Score'].rank(ascending=False, method='dense').astype(int)
# Ajouter une colonne "Catégorie" en fonction des scores d'attractivité
table_data['Catégorie'] = pd.cut(table_data['Score'], bins=[-np.inf, 0.333, 0.67, np.inf], labels=['Attractivité Faible', 'Attractivité Moyenne', 'Attractivité Forte'], right=False)
# Définir une fonction pour mapper les catégories aux récompenses
def map_reward(category):
    if category == 'Attractivité Faible':
        return '⭐⭐'
    elif category == 'Attractivité Moyenne':
        return '⭐⭐⭐'
    elif category == 'Attractivité Forte':
        return '⭐⭐⭐⭐⭐'

# Ajouter une colonne "Reward" en utilisant la fonction de mappage
table_data['Reward'] = table_data['Catégorie'].apply(lambda x: map_reward(x))



# Afficher le tableau
st.write("Tableau des informations des communes:")
st.write(table_data)













st.subheader("Repères")

# Sélectionner le score de la commune sélectionnée
data['Score'] = score_attractivite
score_commune_selected = data[data['NomCommune'] == commune_selectionnee]['Score'].values[0]

# Score minimal, maximal et médian de l'échantillon
score_min = data['Score'].min()
score_max = data['Score'].max()
score_median = data['Score'].median()

# Créer un sous-graphique pour chaque catégorie
plt.figure(figsize=(10, 6))

# Histogramme du score de la commune sélectionnée
plt.subplot(2, 2, 1)
plt.hist(score_commune_selected, bins=10, color='skyblue', edgecolor='black')
plt.title('Score de la commune sélectionnée')
plt.xlabel('Score')
plt.ylabel('Fréquence')

# Histogramme du score minimal de l'échantillon
plt.subplot(2, 2, 2)
plt.hist(score_min, bins=10, color='salmon', edgecolor='black')
plt.title('Score minimal de l\'échantillon')
plt.xlabel('Score')
plt.ylabel('Fréquence')

# Histogramme du score maximal de l'échantillon
plt.subplot(2, 2, 3)
plt.hist(score_max, bins=10, color='lightgreen', edgecolor='black')
plt.title('Score maximal de l\'échantillon')
plt.xlabel('Score')
plt.ylabel('Fréquence')

# Histogramme du score médian de l'échantillon
plt.subplot(2, 2, 4)
plt.hist(score_median, bins=10, color='orange', edgecolor='black')
plt.title('Score médian de l\'échantillon')
plt.xlabel('Score')
plt.ylabel('Fréquence')

# Ajuster le placement des sous-graphiques pour éviter les chevauchements
plt.tight_layout()

# Afficher le graphique sur Streamlit
st.pyplot()




# Créer un graphique pour afficher les histogrammes
plt.figure(figsize=(10, 6))

# Définir les positions des barres
positions = [1, 2, 3, 4]

# Largeur des barres
width = 0.3
# Trouver le nom de la commune ayant le score minimal
commune_min_score = data[data['Score'] == data['Score'].min()]['NomCommune'].values[0]
commune_max_score = data[data['Score'] == data['Score'].max()]['NomCommune'].values[0]

# Diagramme en barres du score de la commune sélectionnée
plt.bar(positions[0], score_commune_selected, width, color='skyblue', label='Commune sélectionnée')

# Diagramme en barres du score minimal de l'échantillon
plt.bar(positions[1], score_min, width, color='salmon', label='Score minimal')

# Diagramme en barres du score maximal de l'échantillon
plt.bar(positions[2], score_max, width, color='lightgreen', label='Score maximal')

# Diagramme en barres du score médian de l'échantillon
plt.bar(positions[3], score_median, width, color='orange', label='Score médian')

# Définir les étiquettes des positions
plt.xticks(positions, ['Score de la cmmune sélectionnée', 'Score minimal', 'Score maximal', 'Score médian'])

# Ajouter une légende
plt.legend()

# Ajouter des titres et des étiquettes
plt.title('Distribution des scores des communes')
plt.xlabel('Catégories')
plt.ylabel('Score')

# Afficher le graphique
st.pyplot()


# Créer un graphique pour afficher les histogrammes
plt.figure(figsize=(10, 6))

# Définir les positions des barres
positions = [1, 2, 3, 4]

# Largeur des barres
width = 0.3

# Diagramme en barres du score de la commune sélectionnée
plt.bar(positions[0], score_commune_selected, width, color='skyblue', label=f'{commune_selectionnee}')

# Diagramme en barres du score minimal de l'échantillon
plt.bar(positions[1], score_min, width, color='salmon', label=f'{commune_min_score}')

# Diagramme en barres du score maximal de l'échantillon
plt.bar(positions[2], score_max, width, color='lightgreen', label=f'{commune_max_score}')

# Diagramme en barres du score médian de l'échantillon
plt.bar(positions[3], score_median, width, color='orange', label='Score médian')

# Définir les étiquettes des positions
plt.xticks(positions, [' Score de la commune sélectionnée', 'Score minimal', 'Score maximal', 'Score médian'])

# Ajouter une légende
plt.legend()

# Ajouter des titres et des étiquettes
plt.title('Position de la commune sélectionnée')
plt.xlabel('Catégories')
plt.ylabel('Score')
st.pyplot()


# In[ ]:





# In[ ]:




