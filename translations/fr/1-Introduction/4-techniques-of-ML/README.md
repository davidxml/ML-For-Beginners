# Techniques d'apprentissage automatique

Le processus de création, d'utilisation et de maintenance des modèles d'apprentissage automatique et des données qu'ils utilisent est très différent de nombreux autres flux de travail de développement. Dans cette leçon, nous allons démystifier le processus et exposer les principales techniques que vous devez connaître. Vous allez :

- Comprendre les processus sous-jacents à l'apprentissage automatique à un niveau élevé.
- Explorer les concepts de base tels que « modèles », « prédictions » et « données d'entraînement ».

## [Quiz avant la leçon](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant cette leçon.

## Introduction

À un niveau élevé, l'art de créer des processus d'apprentissage automatique (ML) se compose de plusieurs étapes :

1. **Décider de la question**. La plupart des processus de ML commencent par poser une question qui ne peut pas être répondue par un programme conditionnel simple ou un moteur basé sur des règles. Ces questions tournent souvent autour des prédictions basées sur un ensemble de données.
2. **Collecter et préparer les données**. Pour pouvoir répondre à votre question, vous avez besoin de données. La qualité et, parfois, la quantité de vos données détermineront à quel point vous pouvez répondre à votre question initiale. Visualiser les données est un aspect important de cette phase. Cette phase inclut également la division des données en groupes d'entraînement et de test pour construire un modèle.
3. **Choisir une méthode d'entraînement**. Selon votre question et la nature de vos données, vous devez choisir comment vous souhaitez entraîner un modèle afin de mieux refléter vos données et faire des prédictions précises sur celles-ci. C'est la partie de votre processus ML qui nécessite une expertise spécifique et, souvent, une quantité considérable d'expérimentation.
4. **Entraîner le modèle**. En utilisant vos données d'entraînement, vous utiliserez divers algorithmes pour entraîner un modèle à reconnaître les motifs dans les données. Le modèle pourrait utiliser des poids internes que l'on peut ajuster pour privilégier certaines parties des données par rapport à d'autres afin de construire un meilleur modèle.
5. **Évaluer le modèle**. Vous utilisez des données jamais vues auparavant (vos données de test) issues de votre ensemble collecté pour voir comment le modèle fonctionne.
6. **Réglage des paramètres**. En fonction des performances de votre modèle, vous pouvez refaire le processus en utilisant différents paramètres, ou variables, qui contrôlent le comportement des algorithmes utilisés pour entraîner le modèle.
7. **Prédire**. Utilisez de nouvelles entrées pour tester la précision de votre modèle.

## Quelle question poser

Les ordinateurs sont particulièrement compétents pour découvrir des motifs cachés dans les données. Cette utilité est très utile pour les chercheurs qui ont des questions sur un domaine donné qui ne peuvent pas être facilement répondues en créant un moteur de règles conditionnelles. Par exemple, pour une tâche actuarielle, un data scientist pourrait être en mesure de construire des règles manuelles autour de la mortalité des fumeurs par rapport aux non-fumeurs.

Cependant, lorsque de nombreuses autres variables sont prises en compte, un modèle ML pourrait s'avérer plus efficace pour prédire les taux de mortalité futurs basés sur l'historique de santé passé. Un exemple plus joyeux pourrait être la réalisation de prévisions météorologiques pour le mois d'avril dans un lieu donné en se basant sur des données comprenant la latitude, la longitude, le changement climatique, la proximité de l'océan, les schémas du jet stream, et plus encore.

✅ Cette [présentation](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sur les modèles météorologiques offre une perspective historique sur l'utilisation du ML dans l'analyse météorologique.

## Tâches préalables à la construction

Avant de commencer à construire votre modèle, il y a plusieurs tâches que vous devez accomplir. Pour tester votre question et formuler une hypothèse basée sur les prédictions d'un modèle, vous devez identifier et configurer plusieurs éléments.

### Données

Pour pouvoir répondre à votre question avec une certaine certitude, vous avez besoin d'une bonne quantité de données du bon type. Il y a deux choses que vous devez faire à ce stade :

- **Collecter des données**. En gardant à l'esprit la leçon précédente sur l'équité dans l'analyse des données, collectez vos données avec soin. Soyez conscient des sources de ces données, de leurs biais intrinsèques éventuels, et documentez leur origine.
- **Préparer les données**. Il y a plusieurs étapes dans le processus de préparation des données. Vous pourriez avoir besoin de rassembler des données et de les normaliser si elles proviennent de sources diverses. Vous pouvez améliorer la qualité et la quantité des données par diverses méthodes telles que la conversion de chaînes en nombres (comme nous le faisons dans [Clustering](../../5-Clustering/1-Visualize/README.md)). Vous pouvez aussi générer de nouvelles données, basées sur l'originale (comme dans [Classification](../../4-Classification/1-Introduction/README.md)). Vous pouvez nettoyer et éditer les données (comme nous le ferons avant la leçon [Web App](../../3-Web-App/README.md)). Enfin, vous pourriez aussi avoir besoin de les randomiser et de les mélanger, en fonction de vos techniques d'entraînement.

✅ Après avoir collecté et traité vos données, prenez un moment pour vérifier si leur forme vous permettra d'aborder la question prévue. Il se peut que les données ne fonctionnent pas bien pour la tâche donnée, comme nous le découvrons dans nos leçons [Clustering](../../5-Clustering/1-Visualize/README.md) !

### Caractéristiques et cible

Une [caractéristique](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) est une propriété mesurable de vos données. Dans de nombreux ensembles de données, elle est exprimée comme un en-tête de colonne tel que « date », « taille » ou « couleur ». Votre variable caractéristique, généralement représentée par `X` dans le code, représente la variable d'entrée qui sera utilisée pour entraîner un modèle.

Une cible est ce que vous essayez de prédire. La cible, généralement représentée par `y` dans le code, représente la réponse à la question que vous posez à vos données : en décembre, quelle **couleur** de citrouilles sera la moins chère ? à San Francisco, quels quartiers auront le meilleur **prix** immobilier ? Parfois, la cible est aussi appelée attribut d'étiquette.

### Sélectionner votre variable caractéristique

🎓 **Sélection et extraction des caractéristiques** Comment savoir quelle variable choisir lors de la construction d'un modèle ? Vous passerez probablement par un processus de sélection ou d'extraction des caractéristiques pour choisir les bonnes variables pour le modèle le plus performant. Ce ne sont cependant pas la même chose : « L'extraction des caractéristiques crée de nouvelles caractéristiques à partir de fonctions des caractéristiques originales, tandis que la sélection des caractéristiques retourne un sous-ensemble des caractéristiques. » ([source](https://wikipedia.org/wiki/Feature_selection))

### Visualisez vos données

Un aspect important de la boîte à outils du data scientist est la capacité à visualiser les données en utilisant plusieurs excellentes bibliothèques telles que Seaborn ou MatPlotLib. Représenter vos données visuellement peut vous permettre de découvrir des corrélations cachées que vous pouvez exploiter. Vos visualisations peuvent aussi vous aider à détecter des biais ou des données déséquilibrées (comme nous le découvrons dans [Classification](../../4-Classification/2-Classifiers-1/README.md)).

### Divisez votre ensemble de données

Avant l'entraînement, vous devez diviser votre ensemble de données en deux parties ou plus de tailles inégales qui représentent toujours bien les données.

- **Entraînement**. Cette partie de l'ensemble de données sert à ajuster votre modèle pour l'entraîner. Cet ensemble constitue la majorité des données originales.
- **Test**. Un ensemble de test est un groupe indépendant de données, souvent tiré des données originales, que vous utilisez pour confirmer la performance du modèle construit.
- **Validation**. Un ensemble de validation est un groupe indépendant plus petit d'exemples que vous utilisez pour régler les hyperparamètres ou l'architecture du modèle afin d'améliorer celui-ci. Selon la taille de vos données et la question posée, vous pourriez ne pas avoir besoin de construire ce troisième ensemble (comme nous le notons dans [Prévision des séries temporelles](../../7-TimeSeries/1-Introduction/README.md)).

## Construire un modèle

En utilisant vos données d'entraînement, votre objectif est de construire un modèle, ou une représentation statistique de vos données, en utilisant divers algorithmes pour **l'entraîner**. L'entraînement d'un modèle l'expose aux données et lui permet de faire des hypothèses sur les motifs perçus qu'il découvre, valide, et accepte ou rejette.

### Choisir une méthode d'entraînement

Selon votre question et la nature de vos données, vous choisirez une méthode pour l'entraîner. En parcourant la [documentation de Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que nous utilisons dans ce cours - vous pouvez explorer plusieurs façons d'entraîner un modèle. Selon votre expérience, vous devrez peut-être essayer plusieurs méthodes différentes pour construire le meilleur modèle. Vous êtes susceptible de passer par un processus où les data scientists évaluent la performance d'un modèle en lui fournissant des données non vues, vérifient la précision, le biais, et d'autres problèmes dégradant la qualité, puis sélectionnent la méthode d'entraînement la plus appropriée pour la tâche.

### Entraîner un modèle

Armé de vos données d'entraînement, vous êtes prêt à le « fitter » pour créer un modèle. Vous remarquerez que dans beaucoup de bibliothèques ML, on trouve le code `model.fit` - c'est à ce moment que vous envoyez votre variable caractéristique sous forme de tableau de valeurs (généralement `X`) et une variable cible (généralement `y`).

### Évaluer le modèle

Une fois le processus d'entraînement terminé (cela peut prendre plusieurs itérations, ou « époques », pour entraîner un grand modèle), vous pourrez évaluer la qualité du modèle en utilisant des données de test pour mesurer ses performances. Ces données sont un sous-ensemble des données originales que le modèle n'a pas encore analysées. Vous pouvez afficher un tableau des métriques concernant la qualité de votre modèle.

🎓 **Ajustement du modèle**

Dans le contexte de l'apprentissage automatique, l'ajustement du modèle fait référence à la précision de la fonction sous-jacente du modèle alors qu'il tente d'analyser des données qu'il ne connaît pas.

🎓 L'**underfitting** (sous-apprentissage) et l'**overfitting** (sur-apprentissage) sont des problèmes courants qui dégradent la qualité du modèle, car le modèle est soit mal ajusté, soit trop ajusté. Cela fait que le modèle fait des prédictions soit trop proches, soit trop éloignées de ses données d'entraînement. Un modèle surajusté prédit trop bien les données d'entraînement parce qu'il a trop appris les détails et le bruit des données. Un modèle sous-ajusté n'est pas précis car il ne peut ni analyser correctement ses données d'entraînement ni les données qu'il n'a pas encore « vues ».

![overfitting model](../../../../translated_images/fr/overfitting.1c132d92bfd93cb6.webp)
> Infographie par [Jen Looper](https://twitter.com/jenlooper)

## Réglage des paramètres

Une fois votre entraînement initial terminé, observez la qualité du modèle et envisagez de l'améliorer en ajustant ses « hyperparamètres ». Pour en savoir plus sur ce processus, consultez la [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prédiction

C'est le moment où vous pouvez utiliser des données complètement nouvelles pour tester la précision de votre modèle. Dans un contexte de ML « appliqué », où vous construisez des actifs web pour utiliser le modèle en production, ce processus peut impliquer de recueillir une entrée utilisateur (par exemple, l'appui d'un bouton) pour définir une variable et l'envoyer au modèle pour inférence, ou évaluation.

Dans ces leçons, vous découvrirez comment utiliser ces étapes pour préparer, construire, tester, évaluer et prédire – tous les gestes d'un data scientist et plus encore, à mesure que vous progressez dans votre parcours pour devenir un ingénieur ML « full stack ».

---

## 🚀Défi

Dessinez un organigramme reflétant les étapes d'un praticien ML. Où vous situez-vous actuellement dans ce processus ? Où prévoyez-vous rencontrer des difficultés ? Qu'est-ce qui vous semble facile ?

## [Quiz après la leçon](https://ff-quizzes.netlify.app/en/ml/)

## Révision & Auto-apprentissage

Cherchez en ligne des interviews de data scientists qui parlent de leur travail quotidien. En voici [une](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Devoir

[Interviewer un data scientist](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Avis de non-responsabilité** :  
Ce document a été traduit à l’aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d’assurer l’exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source faisant autorité. Pour des informations critiques, une traduction professionnelle humaine est recommandée. Nous déclinons toute responsabilité en cas de malentendus ou de mauvaises interprétations résultant de l’utilisation de cette traduction.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->