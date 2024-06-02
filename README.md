PROJETS Apprentissage automatique supervisé
Ventes Walmart

Description de l'entreprise 📇
Walmart Inc. est une société multinationale américaine de vente au détail qui exploite une chaîne d'hypermarchés, de grands magasins discount et d'épiceries des États-Unis, dont le siège est à Bentonville, Arkansas. L'entreprise a été fondée par Sam Walton en 1962.

Projet 🚧
Le service marketing de Walmart vous a demandé de construire un modèle de machine learning capable d'estimer les ventes hebdomadaires dans leurs magasins, avec la meilleure précision possible sur les prédictions faites. Un tel modèle les aiderait à mieux comprendre comment les ventes sont influencées par les indicateurs économiques et pourrait être utilisé pour planifier de futures campagnes marketing.

Objectifs 🎯
Le projet peut être divisé en trois étapes :

Partie 1 : réaliser une EDA et tous les prétraitements nécessaires pour préparer les données pour le machine learning
Partie 2 : entraîner un modèle de régression linéaire (baseline)
Partie 3 : éviter le surajustement en entraînant un modèle de régression régularisé
Portée de ce projet 🖼️
Pour ce projet, vous travaillerez avec un ensemble de données contenant des informations sur les ventes hebdomadaires réalisées par différents magasins Walmart, ainsi que d'autres variables telles que le taux de chômage ou le prix du carburant, qui pourraient être utiles pour prédire le montant des ventes. L'ensemble de données est issu d'un concours Kaggle, mais nous avons apporté quelques modifications par rapport aux données d'origine. Veuillez vous assurer que vous utilisez notre ensemble de données personnalisé (disponible sur JULIE). 🤓

Livrable 📬
Pour mener à bien ce projet, votre équipe doit :

Créer des visualisations
Former au moins un modèle de régression linéaire sur l'ensemble de données, qui prédit le montant des ventes hebdomadaires en fonction des autres variables
Évaluer les performances du modèle en utilisant une métrique pertinente pour les problèmes de régression
Interpréter les coefficients du modèle pour identifier les caractéristiques importantes pour la prédiction
Entraînez au moins un modèle avec régularisation (Lasso ou Ridge) pour réduire le surajustement
Aides 🦮
Pour vous aider à réaliser ce projet, voici quelques conseils qui devraient vous aider :

Partie 1 : EDA et prétraitement des données
Démarrez votre projet en explorant votre jeu de données : créez des chiffres, calculez des statistiques etc...

Ensuite, vous devrez effectuer un prétraitement sur l'ensemble de données. Vous pouvez suivre les directives du modèle de prétraitement . Il y aura également quelques transformations spécifiques à prévoir sur ce jeu de données, par exemple sur la colonne Date qui ne peut pas être incluse telle quelle dans le modèle. Voici quelques conseils qui pourraient vous aider 🤓

Prétraitement à prévoir avec les pandas
Supprimez les lignes où les valeurs cibles sont manquantes :

Ici, la variable cible (Y) correspond à la colonne Weekly_Sales . On peut voir ci-dessus qu'il manque des valeurs dans cette colonne.
Nous n'utilisons jamais de techniques d'imputation sur la cible : cela pourrait créer des biais dans les prédictions !
Ensuite, nous supprimerons simplement les lignes de l'ensemble de données pour lesquelles la valeur dans Weekly_Sales est manquante.
Créer des fonctionnalités utilisables à partir de la colonne Date : La colonne Date ne peut pas être incluse telle quelle dans le modèle. Soit vous pouvez supprimer cette colonne, soit vous créerez de nouvelles colonnes contenant les caractéristiques numériques suivantes :

année
mois
jour
jour de la semaine
Déposez les lignes contenant des valeurs invalides ou des valeurs aberrantes : Dans ce projet, seront considérées comme des valeurs aberrantes toutes les caractéristiques numériques qui n'entrent pas dans la plage :
[
𝑋
ˉ
−
3
𝜎
,
𝑋
ˉ
+
3
𝜎
]
[ 
X
ˉ
 −3 σ , 
X
ˉ
 +3 ]​. Cela concerne les colonnes : Temperature , Fuel_price , CPI et Unemployment

Variable cible/cible (Y) que l'on va essayer de prédire, pour se séparer des autres : Weekly_Sales

------------

Prétraitements à planifier avec scikit-learn
Variables explicatives (X) Nous devons identifier quelles colonnes contiennent des variables catégorielles et quelles colonnes contiennent des variables numériques, car elles seront traitées différemment.

Variables catégorielles : Store, Holiday_Flag
Variables numériques : Température, Fuel_Price, CPI, Chômage, Année, Mois, Jour, DayOfWeek
Partie 2 : Modèle de base (régression linéaire)
Une fois que vous avez entraîné un premier modèle, n'oubliez pas d'évaluer ses performances sur le train et sur les trains de test. Etes-vous satisfait des résultats ? En outre, il serait intéressant d'analyser les valeurs des coefficients du modèle pour savoir quelles caractéristiques sont importantes pour la prédiction. Pour ce faire, l' .coef_attribut de la classe LinearRegression de scikit-learn pourrait être utile. Veuillez vous référer au lien suivant pour plus d'informations 😉 https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

Partie 3 : Combattre le surapprentissage
Dans cette dernière partie, vous devrez entraîner un modèle de régression linéaire régularisé . Vous trouverez ci-dessous quelques classes utiles dans la documentation de scikit-learn :

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso
Question bonus

Dans les modèles de régression régularisés, il existe un hyperparamètre appelé force de régularisation qui peut être affiné pour obtenir les meilleures prédictions généralisées sur un ensemble de données donné. Ce réglage fin peut être effectué grâce à la classe GridSearchCV de scikit-learn : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

Vous trouverez également ici quelques exemples d'utilisation de GridSearchCV avec les modèles Ridge ou Lasso : https://alfurka.github.io/2018-11-18-grid-search/
