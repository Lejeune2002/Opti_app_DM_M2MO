# Opti_app_DM_M2MO

Ce repo est crée pour le cours d'optimisation pour l'apprentissage du Master 2 MO dans le cadre d'un DM portant sur la comparaison des convergences des méthodes de descente de gradient stochastique et spectrale.

Pour 21/10 : Pour l'instant, le travail effectué vise simplement à me familiariser avec Pytorch et l'implémentation des algorithmes de descente. On peut donc trouver à ce jour dans le repo différents fichiers :
- les fichiers toy_GD : des exemples basiques qui mettent en place des descentes de gradient "triviales" visant à comprendre l'objectif
- Le fichier toy2.py qui suit le tutoriel d'utilisation de Pytorch : https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html Ajoute cependant deux optimizers qui sont Muon et SGD 
- data : Utilisé dans le cadre du fichier toy2.py
- pytorche_examples.ipynb : exemple du cours fournit dans le cadre du projet.
- pp.py : Implémentation d'une descente de gradient classique pour trouver les poids d'un réseau de neurone à 5 couches en superposant Relu = max(0,x) et couche linéaire
- comp_SGD/Comp_SGG : comparaison descente de gradient stochastique et descente de gradient spectral pour toy2.py durant l'entraînement. On remarque pour l'instant que dans cet exemple, la descente de gradient stochastique fait moins bien vis-à-vis de la métrique choisie. Ce résultat ne vaut cependant rien à l'heure actuelle car Muon entraîne les poids des matrices là où la descente de gradient stochastique n'entraîne que le terme de biais.


Problèmes rencontrés jusq'à présent :
 Gestion du biais dans les régressions matricielles Convention Pytorch 
 Quelle métrique utiliser ? 
 Comparer les modèles sur des lr différents ?
 Choix de l'exemple à traiter ? Spectral GD et SGD ne sont pas créées dans le même but
 Au début du mal à implémenter des algos de descente de gradient à des problèmes d'apprentissage Lien optimisation / Apprentissage. Solution : on applique le backward() sur la métrique pas sur autre chose
