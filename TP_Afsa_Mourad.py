#!/bin/env/python3

import pandas as pd
import numpy as np
import numpy.lib.scimath as sm
import math

def Exo1():
    a=0
    global pivot, P
    """
    1) Matrice de corrélation
    """
    s = pd.read_csv('EU electricity price_v2.csv')
    data = s.dropna()
    lst=list(set(data['Country']))
    pivot = data.pivot(index='Date', columns='Country', values='price')
    matrix_corr = pivot.corr()
    #print("matrix corrélation:\n",matrix_corr)

    """
    #2) Trouver les matrices P orthogonale et D diagonale:
    """
    valeurs, P = np.linalg.eig(matrix_corr)
    global D
    D = np.diag(valeurs)
    #print("Matrice diagonale:\n",D)
    #print("\nMatrice Orthogonale:\n",P)

    """
    vérifions si P.D.Pt redonne la matrice de corrélation
    """
    Pt = np.transpose(P)
    M = np.dot(P, np.dot(D, Pt))
    #print("\n\n\n\nmatrice corrélation  et la matrice obtenue sont elles similaires?: ", M == matrix_corr)
    """
    Nous constatons que c'est bien le cas
    """

    """
    3) Produit matrice de corrélation et premier vecteur de P
        Soit vec1 ce vecteur et Vecteur_prod le résultat du calcul 
    """
    vec1 = P[:,0]
    Vecteur_prod = np.dot(matrix_corr, vec1)

    #print('\nVec pro is:\n',Vecteur_prod)
    #print("first is:", vec1)
    """
    on constate que le produit est proportionnel au premier vecteur
    lorsqu'on fait 15.5810262 * vecteur_prod = vec1 correspondant
    """

    """
    4) Explication de la valeur négative dans le Word
        Remplaçons par 1e-8
    """
    D[D < 0] = 1e-8
    D = np.array(D)

    """
    5) Montrons: voir TP.ipynb
    """

    global var_exp, var_cum, ACP, A, Q
    A = np.sqrt(D)
    Q = np.dot(P, A)
    x = pivot.to_numpy().transpose()
    comp_princip = np.dot(pivot, P[:, :3])
    ACP = np.dot(np.linalg.inv(Q), x)
    #print(ACP)
    var_exp = list(val / sum(np.diag(D)) for val in np.diag(D))
    var_cum = np.cumsum(var_exp)
    num = sum(1 for i in var_cum if float(i) <= 0.93) + 1
    #print(var_cum)
    
    """
    D'après le graphe on constate qu'environ 3 facteurs suffisent à expliquer 93%
    De même dans la liste de variance cumulée contenue dans la variable
    var_cum, on constate bien qu'uniquement 3 variables sont inférieures ou égales à 0.93
    """
    

    #6) (voir TP.ipynb)
         
    
    """
    7) L'anlyste a tort car dans la variable var_cum, il faut au moins 4 facteurs pour
     capturer environ 95% le prix de l’électricité dans la zone européenne.
     [0.82005401 0.89652598 0.93582733 0.95566739 0.97031384 0.97910324
     0.98449251 0.9884694  0.99148155 0.99374635 0.99560688 0.99691353
     0.99802332 0.99884079 0.99915728 0.99955934 0.99995541 1.
     1.        ]
    """
    
    """
    8) La formule est donnée par l'expression:
      cos(a) = (PC1 . pays-vecteur) / (PC1_norme * pays-vecteur_norme) 
    """
    #On récupère le vecteur des certains pays
    ACP1 = pd.DataFrame(data=comp_princip, columns=['PC1', 'PC2', 'PC3'], index=pivot.index)
    country1_vec = pivot['France'].values
    country2_vec = pivot['Poland'].values
    country3_vec = pivot['Romania'].values

    #On récupere ensuite les composantes principales
    PC1 = ACP1['PC1'].values
    PC2 = ACP1['PC2'].values
    PC3 = ACP1['PC3'].values


    #On fait le produit scalaire
    Prod = np.dot(PC1, country1_vec)
    Prod1 = np.dot(PC2, country2_vec)
    Prod2 = np.dot(PC3, country3_vec)


    #On calcule les normes des vecteurs
    PC_dist = np.linalg.norm(PC1)
    PC_dist1 = np.linalg.norm(PC2)
    PC_dist2 = np.linalg.norm(PC3)

    Country_dist = np.linalg.norm(country1_vec)
    Country_dist1 = np.linalg.norm(country2_vec)
    Country_dist2 = np.linalg.norm(country3_vec)


    cos_angle = (Prod / (PC_dist * Country_dist))
    cos_angle1 = (Prod1 / (PC_dist1 * Country_dist1))
    cos_angle2 = (Prod2 / (PC_dist2 * Country_dist2))

    angle = np.degrees(np.arccos(cos_angle))
    angle1 = np.degrees(np.arccos(cos_angle1))
    angle2 = np.degrees(np.arccos(cos_angle2))


    print(f"Vecteur du premier pays (France):\n", country1_vec)
    print(f"\n\ncos is {cos_angle} and angle is: {angle}")

    print(f"Vecteur du second pays (Poland):\n", country2_vec)
    print(f"\n\ncos is {cos_angle1} and angle is: {angle1}")

    print(f"Vecteur du tertio pays (Romania):\n", country3_vec)
    print(f"\n\ncos is {cos_angle2} and angle is: {angle2}")    



def Exo2():
    #1) Démonstration dans le word
    #2) SImulation de loi normale
    X_mu = 0
    X_sigma = 3
    Z_mu = 0
    Z_sigma = 1
    N = 1000

    X = np.random.normal(X_mu, X_sigma, N)

    #voir dans le fichier TP.ipynb pour la suite

def main():
    Exo1()
    Exo2()

main()