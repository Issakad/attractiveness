#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("C:\\Users\\Utilisateur\\Desktop\\Collaboration\\Recherche académique\\Projets d'articles\\Fiscalité locale\\Dashboard\\"))
#print(os.listdir("C:\\Users\\Utilisateur\\Desktop\\Data science\\Openclassrooms\\Projet 7\\data\\"))

# Chargement des données de démonstration
data = pd.read_excel("C:\\Users\\Utilisateur\\Desktop\\Collaboration\\Recherche académique\\Projets d'articles\\Fiscalité locale\\Dashboard\\data2019.xlsx") # Assurez-vous de remplacer "data.csv" par le chemin de votre fichier de données
# Ajout de l'index à la DataFrame data
data = data.reset_index(drop=True)

# Titre du dashboard
st.title("Dashboard d'attractivité des communes")
# Afficher les 100 premières lignes du tableau
st.write("Les 100 premières lignes du tableau :")
st.dataframe(data.head(100))
# Sélection de l'année
annee_selectionnee = st.sidebar.selectbox("Sélectionner l'année", sorted(data['Annee'].unique()))

# Sélection du nom de la commune
commune_selectionnee = st.sidebar.selectbox("Sélectionner la commune", sorted(data['NomCommune'].unique()))

# Filtrer les données pour ne conserver que celles de la commune sélectionnée
donnees_commune = data[data['NomCommune'] == commune_selectionnee]

# Afficher les données de la commune sélectionnée
st.write(f"Données pour la commune sélectionnée ({commune_selectionnee}) :")
st.write(donnees_commune)

# Sélection des dimensions de l'indice d'attractivité
dimensions = st.sidebar.multiselect("Sélectionner les dimensions de l'indice",
                                    ['Dynamisme économique', 'Dépendance vieillesse', 'Pression fiscale', 'Chômage', 'Gouvernance', 'Biens publics'])


# Obtenir les valeurs des variables pour la commune sélectionnée
commune_selected_row = data[data['NomCommune'] == commune_selectionnee].iloc[0]
# Supprimer les espaces en début et en fin de chaîne et mettre en minuscule le nom de la colonne
commune_selected_row = commune_selected_row.index.str.strip().str.lower()

# Accès à la valeur dans la colonne 'dynamisme économique'
# Accès à la valeur dans la colonne 'Dynamisme économique'
dynamisme_economique = commune_selected_row['Dynamisme économique']


dependance_vieillesse = commune_selected_row['Dépendance vieillesse']
pression_fiscale = commune_selected_row['Pression fiscale']
chomage = commune_selected_row['Chômage']
gouvernance = commune_selected_row['Gouvernance']
biens_publics = commune_selected_row['Biens publics']

# Calcul du score d'attractivité selon la formule donnée
score_attractivite = (3**0.5)/4 * (dynamisme_economique * dependance_vieillesse +
                                   dependance_vieillesse * pression_fiscale +
                                   pression_fiscale * chomage +
                                   chomage * gouvernance +
                                   gouvernance * biens_publics +
                                   biens_publics * dynamisme_economique)

# Affichage du score d'attractivité de la commune sélectionnée
st.write(f"Score d'attractivité de la commune sélectionnée ({commune_selectionnee}): {score_attractivite}")
st.write(score_attractivite)

# Calcul des statistiques sur le score
score_min = data['Score'].min()
score_max = data['Score'].max()
score_median = data['Score'].median()
#score_commune_selectionnee = data_filtree.loc[data_filtree['NomCommune'] == commune_selectionnee, 'Score'].iloc[0]

# Affichage des scores
st.write(f"Score de la commune sélectionnée ({commune_selectionnee}): {score_commune_selectionnee}")
st.write(f"Score minimum de l'échantillon: {score_min}")
st.write(f"Score maximum de l'échantillon: {score_max}")
st.write(f"Score médian de l'échantillon: {score_median}")



# Affichage de l'histogramme du score
st.write("### Histogramme du score d'attractivité")
plt.figure(figsize=(8, 6))
sns.histplot(data['Score'], kde=True)
st.pyplot()

# Affichage du graphique en radar pour les dimensions de l'indice
st.write("### Graphique en radar des dimensions de l'indice")
data_commune = data[data['Commune'] == commune_selectionnee].iloc[0]
values = data_commune[dimensions].tolist()
categories = dimensions
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

values += values[:1]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories, color='grey', size=12)
ax.plot(angles, values)
ax.fill(angles, values, 'blue', alpha=0.1)
st.pyplot()


# In[9]:


import os
print(os.listdir("C:\\Users\\Utilisateur\\Desktop\\Collaboration\\Recherche académique\\Projets d'articles\\Fiscalité locale\\Dashboard\\"))                


# In[ ]:





# In[ ]:




