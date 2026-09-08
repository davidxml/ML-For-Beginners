# Introduction à l'apprentissage automatique

## [Quiz avant le cours](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introduction to Machine Learning for Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant cette leçon.

Bienvenue dans ce cours sur l'apprentissage automatique classique pour débutants ! Que vous soyez complètement novice dans ce sujet, ou un praticien expérimenté de l'IA souhaitant réviser un domaine, nous sommes heureux de vous accueillir ! Nous voulons créer un point de départ convivial pour votre étude de l'IA et serions ravis d'évaluer, répondre et intégrer vos [retours](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction to ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : John Guttag du MIT présente l'apprentissage automatique

---
## Commencer avec l'apprentissage automatique

Avant de commencer ce programme, vous devez configurer votre ordinateur et être prêt à exécuter des notebooks localement.

- **Configurez votre machine avec ces vidéos**. Utilisez les liens suivants pour apprendre [comment installer Python](https://youtu.be/CXZYvNRIAKM) sur votre système et [configurer un éditeur de texte](https://youtu.be/EU8eayHWoZg) pour le développement.
- **Apprenez Python**. Il est également recommandé d’avoir une compréhension de base de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un langage de programmation utile pour les data scientists que nous utilisons dans ce cours.
- **Apprenez Node.js et JavaScript**. Nous utilisons également JavaScript plusieurs fois dans ce cours lors de la création d'applications web, vous aurez donc besoin d’avoir [node](https://nodejs.org) et [npm](https://www.npmjs.com/) installés, ainsi que [Visual Studio Code](https://code.visualstudio.com/) disponible pour le développement Python et JavaScript.
- **Créez un compte GitHub**. Puisque vous nous avez trouvés ici sur [GitHub](https://github.com), vous avez peut-être déjà un compte, sinon créez-en un puis fork ce programme pour l’utiliser vous-même. (N'hésitez pas à nous donner une étoile aussi 😊)
- **Explorez Scikit-learn**. Familiarisez-vous avec [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un ensemble de bibliothèques d’IA que nous référencions dans ces leçons.

---
## Qu'est-ce que l'apprentissage automatique ?

Le terme « apprentissage automatique » est l'un des plus populaires et fréquemment utilisés aujourd’hui. Il est fort probable que vous ayez entendu ce terme au moins une fois si vous avez un minimum de familiarité avec la technologie, quel que soit votre domaine. Cependant, les mécanismes de l'apprentissage automatique sont un mystère pour la plupart des gens. Pour un débutant en apprentissage automatique, le sujet peut parfois sembler écrasant. Il est donc important de comprendre ce qu'est réellement l'apprentissage automatique, et d'en apprendre pas à pas, à travers des exemples pratiques.

---
## La courbe de popularité

![ml hype curve](../../../../translated_images/fr/hype.07183d711a17aafe.webp)

> Google Trends montre la récente « courbe de popularité » du terme « machine learning »

---
## Un univers mystérieux

Nous vivons dans un univers rempli de mystères fascinants. De grands scientifiques comme Stephen Hawking, Albert Einstein et bien d'autres ont consacré leur vie à rechercher des informations significatives qui dévoilent les mystères du monde qui nous entoure. C’est la condition humaine de l’apprentissage : un enfant humain découvre de nouvelles choses et dévoile la structure de son monde année après année en grandissant jusqu’à l’âge adulte.

---
## Le cerveau de l’enfant

Le cerveau et les sens d’un enfant perçoivent les faits de leur environnement et apprennent progressivement les motifs cachés de la vie qui aident l’enfant à élaborer des règles logiques pour identifier les motifs appris. Le processus d’apprentissage du cerveau humain fait de l’homme la créature la plus sophistiquée de ce monde. Apprendre continuellement en découvrant des motifs cachés puis innover sur ces motifs nous permet de nous améliorer tout au long de notre vie. Cette capacité d’apprentissage et cette faculté d’évolution sont liées à un concept appelé [plasticité cérébrale](https://www.simplypsychology.org/brain-plasticity.html). Superficiellement, on peut tracer quelques similitudes motivantes entre le processus d'apprentissage du cerveau humain et les concepts d'apprentissage automatique.

---
## Le cerveau humain

Le [cerveau humain](https://www.livescience.com/29365-human-brain.html) perçoit les choses du monde réel, traite l’information perçue, prend des décisions rationnelles et effectue certaines actions selon les circonstances. C’est ce que l’on appelle se comporter intelligemment. Lorsque nous programmons une copie de ce processus comportemental intelligent sur une machine, cela s’appelle l’intelligence artificielle (IA).

---
## Quelques terminologies

Bien que les termes puissent être confondus, l'apprentissage automatique (ML) est un sous-ensemble important de l'intelligence artificielle. **Le ML consiste à utiliser des algorithmes spécialisés pour découvrir des informations significatives et trouver des motifs cachés dans les données perçues afin de corroborer le processus de prise de décision rationnelle**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../translated_images/fr/ai-ml-ds.537ea441b124ebf6.webp)

> Un schéma montrant les relations entre IA, ML, apprentissage profond et science des données. Infographie par [Jen Looper](https://twitter.com/jenlooper) inspirée par [ce graphique](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepts à couvrir

Dans ce programme, nous allons couvrir uniquement les concepts de base de l'apprentissage automatique qu’un débutant doit connaître. Nous couvrons ce que nous appelons « l'apprentissage automatique classique », principalement à l'aide de Scikit-learn, une excellente bibliothèque que beaucoup d’étudiants utilisent pour apprendre les bases. Pour comprendre des concepts plus larges d’intelligence artificielle ou d’apprentissage profond, une solide connaissance fondamentale de l'apprentissage automatique est indispensable, et nous souhaitons vous la proposer ici.

---
## Ce que vous apprendrez dans ce cours :

- concepts fondamentaux de l'apprentissage automatique
- l'histoire du ML
- ML et équité
- techniques de régression ML
- techniques de classification ML
- techniques de clustering ML
- techniques de traitement du langage naturel ML
- techniques de prévision de séries temporelles ML
- apprentissage par renforcement
- applications réelles de ML

---
## Ce que nous ne couvrirons pas

- apprentissage profond
- réseaux de neurones
- IA

Pour une meilleure expérience d’apprentissage, nous éviterons les complexités des réseaux de neurones, de l’apprentissage profond — construction de modèles multi-couches avec réseaux de neurones — et de l’IA, que nous aborderons dans un autre programme. Nous proposerons également un futur programme de science des données pour nous concentrer sur cet aspect de ce domaine plus large.

---
## Pourquoi étudier l'apprentissage automatique ?

L'apprentissage automatique, d’un point de vue système, est défini comme la création de systèmes automatisés capables d’apprendre des motifs cachés dans les données pour aider à prendre des décisions intelligentes.

Cette motivation est vaguement inspirée de la façon dont le cerveau humain apprend certaines choses à partir des données qu’il perçoit de l’extérieur.

✅ Réfléchissez un instant pourquoi une entreprise voudrait tenter d’utiliser des stratégies d’apprentissage automatique plutôt que de créer un moteur basé sur des règles codées en dur.

---
## Pourquoi la qualité des données est importante

Des données de haute qualité améliorent la performance du modèle. Des données pauvres ou bruitées peuvent conduire à des prédictions inexactes, même en utilisant des algorithmes avancés d'apprentissage automatique.

---
## Applications de l'apprentissage automatique

Les applications de l'apprentissage automatique sont désormais quasiment partout, aussi ubiquistes que les données qui circulent dans nos sociétés, générées par nos smartphones, nos appareils connectés et autres systèmes. Compte tenu du potentiel immense des algorithmes d'apprentissage automatique à la pointe, les chercheurs explorent leur capacité à résoudre des problèmes réels multidimensionnels et multidisciplinaires avec d'excellents résultats positifs.

---
## Exemples d’IA appliquée

**Vous pouvez utiliser l'apprentissage automatique de nombreuses façons** :

- Pour prédire la probabilité d’une maladie à partir de l’historique médical ou des rapports d’un patient.
- Pour exploiter les données météo afin de prévoir des événements climatiques.
- Pour comprendre le sentiment d’un texte.
- Pour détecter les fausses informations afin d’arrêter la propagation de la propagande.

La finance, l’économie, les sciences de la terre, l’exploration spatiale, le génie biomédical, les sciences cognitives, et même les domaines des sciences humaines ont adapté l'apprentissage automatique pour résoudre les problèmes lourds en traitement de données de leur domaine.

---
## Conclusion

L'apprentissage automatique automatise le processus de découverte de motifs en trouvant des informations significatives à partir de données réelles ou générées. Il s’est avéré extrêmement précieux dans les affaires, la santé, les applications financières, entre autres.

Dans un futur proche, comprendre les bases de l'apprentissage automatique sera indispensable pour les personnes de tous les domaines en raison de son adoption généralisée.

---
# 🚀 Défi

Esquissez, sur papier ou en utilisant une application en ligne comme [Excalidraw](https://excalidraw.com/), votre compréhension des différences entre IA, ML, apprentissage profond et science des données. Ajoutez quelques idées de problèmes que chacune de ces techniques est bonne à résoudre.

# [Quiz après le cours](https://ff-quizzes.netlify.app/en/ml/)

---
# Révision & Auto-apprentissage

Pour en savoir plus sur la manière de travailler avec des algorithmes ML dans le cloud, suivez ce [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Suivez un [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sur les bases du ML.

---
# Devoir

[Prenez en main et lancez-vous](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Avertissement** :
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforçions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source faisant autorité. Pour les informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous ne saurions être tenus responsables des malentendus ou erreurs d'interprétation découlant de l'utilisation de cette traduction.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->