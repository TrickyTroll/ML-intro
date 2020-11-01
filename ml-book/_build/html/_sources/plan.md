# Plan de rédaction du travail

**Comment allons-nous démontrer notre thèse en 3 ou 4 grandes étapes. Comment
ces étapes sont-elles utiles**

*Quel est le fonctionnent de l'intelligence artificielle et comment devrait-elle
être utilisée afin de bénéficier l’être humain?*

## Choix de la thèse

L'intelligence artificielle est au coeur de nos vies présentement et a évolué
à une vitesse fulgurante au courant de la dernière décennie. Les multiples 
nouvelles avancées du domaine augmentent aussi le nombre
[d'articles](https://www.nytimes.com/topic/subject/artificial-intelligence)
écrits sur le sujet ainsi que
[l'intérêt](https://trends.google.com/trends/explore?hl=en-GB&tz=240&date=all&q=%2Fm%2F0h1fn8h,machine+learning&sni=3)
de la population par rapport au domaine.

En effet, en faisant une recherche rapide sur le site du
[New York Times](https://www.nytimes.com/), il est possible de déceler que
seulement quelques 113 articles en lien avec l'IA ont été écrits entre le 13
aout 1984 et 12 novembre 2006, soit un peu plus de 5 par année. En comparaison,
975 articles ont été écrits sur le sujet par des auteurs affiliés au journal
entre novembre 2006 et le 8 octobre 2020, une augmentation d'environ 1400%
articles par année.

<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2213_RC01/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"machine learning","geo":"US","time":"2004-01-01 2020-10-26"},{"keyword":"artificial intelligence","geo":"US","time":"2004-01-01 2020-10-26"}],"category":0,"property":""}, {"exploreQuery":"date=all&geo=US&q=machine%20learning,artificial%20intelligence","guestPath":"https://trends.google.com:443/trends/embed/"}); </script>

Les tendances de recherche à l'aide du moteur [Google](https://www.google.com/) 
pointent aussi dans la même direction. L'intérêt dans le temps pour les termes
«Deep learning» et «Machine learning» a beaucoup augmenté récemment 
{cite}`trendsai` . Par contre, la popularité du terme «Artificial Intelligence» 
ne semble pas varier dans la même direction {cite}`trendsai` . Il faut alors prendre en 
considération que l'intelligence artificielle est un domaine qui date, et que
le moteur de recherche n'existe que depuis le début des années 2000.

En se basant sur les statistiques mentionnées plus haut, il est possible
d'affirmer que l'IA est un domaine d'actualité avec lequel la population se
fait de plus en plus familière. De multiples articles traitent déjà du sujet, 
une grande partie d'entre eux mentionnant comment son utilisation pourrait
bénéficier l'être humain.

Le principal problème avec ces articles est qu'ils ne permettent pas au lecteur
de réellement saisir comment un programme peut être entraîné à réaliser des 
tâches qui témoignent d'une intelligence humaine. Une connaissance de ces
concepts est nécessaire au développement d'un bon jugement critique face
aux bonnes et mauvaises utilisations de la technologie.

C'est pourquoi notre thèse inclus non seulement la manière dont l'IA pourrait
être utilisée pour bénéficier l'être humain, mais aussi quel est le 
fonctionnement de la technologie.

Nous allons d'abord introduire le sujet en mentionnant comment l'intelligence
artificielle a évolué depuis sa création. Nous allons ensuite décrire en détail
les composantes d'un _machine learing pipeline_ ainsi que les mathématiques qui
y sont associées. À l'aide de ces composantes, nous allons créer un modèle
permettant la reconnaissance optique de caractères. Par la suite, nous 
analyserons les capacités de notre modèle, mettant beaucoup d'emphase sur les
limites de celui-ci. Pour conclure, nous utiliserons les connaissances
nouvellement acquises afin de porter un regard critique sur les utilisations
potentielles de l'IA.



## Un bref historique

Afin de bien introduire la technologie, nous comptons tenir une ligne du temps
commentée des évènements marquants dans le développement de l'intelligence
artificielle. Voici les éléments que cette ligne du temps devrait pour l'instant
contenir.

### Les débuts de l'IA

#### Alan Turing

Au courant de la Deuxième Guerre mondiale, les forces de l'Axe encryptent leurs
communications à l'aide de machines 
[Enigma](https://en.wikipedia.org/wiki/Enigma_machine). Ces machines rendent le
décryptage par force brute pratiquement impossible à réaliser par des humains.
Les forces alliées ne peuvent alors décrypter les messages que lorsqu'ils
trouvent des indices quant à de l'information contenue dans les messages.

Afin de déchiffrer plus efficacement les communications cryptées, Alan Turing
améliore une machine polonaise et crée une version britannique de la
[Bombe](https://en.wikipedia.org/wiki/Bombe). Cette version rudimentaire de
l'ordinateur sera discutée plus en détail.

Alan Turing publie aussi
[Computing Machinery and Intelligence](https://en.wikipedia.org/wiki/Computing_Machinery_and_Intelligence)
quelques années après la guerre {cite}`ibmai`. C'est à ce papier que l'on doit
le [Turing Test](https://en.wikipedia.org/wiki/Turing_test), une procédure
dont nous comptons aussi discuter.

#### Eliza

Eliza est le tout premier _chatbot_ à être créé. Développée à MIT au courant des
années 60, Eliza permet à un utilisateur d'avoir une conversation similaire à
celles que l'on pourrait entretenir avec un humain {cite}`harvardeliza`.

#### Défaite de Garry Kasparov

En 1987, l'ordinateur Deep Blue de la compagnie IBM défait le champion mondial
aux échecs {cite}`wiredai`.

```{figure} ./img/garry.jpg
---
name: garry-deepblue
---
Garry Kasparov jouant une partie d'échecs contre Deep Blue.

*source: https://www.forbes.com/sites/davidewalt/2011/05/03/kasparov-vs-deep-blue/#1134dd3b30f8*
```

### Les grandes avancées

Ensuite, nous comptons discuter des principales avancées qui ont permis les
avancées fulgurantes des deux dernières décennies.

#### Les quantités massives de données

Nous comptons discuter de l'impact qu'ont eu Google et Facebook dans la collecte
et l'utilisation de l'information de leurs utilisateurs pour le développement de
l'intelligence artificielle.

#### GPUS

Les cartes graphiques ont aussi eu un rôle très important à jouer dans les récentes
avancées du domaine. Nous verrons plus en détail pourquoi les cartes graphiques
sont aussi performantes pour faire de l'intelligence artificielle.

### Les prédictions pour le futur

Nous comptons aussi conclure l'introduction avec une hypothèse quant aux bonnes
et mauvaises utilisations de l'intelligence artificielle. Nous discuterons de
comment cette technologie pourra révolutionner le monde sous la majorité de ses
aspects.

## Notions de base des procédés

La section **Notions de base des procédés** a comme fonction de définir et expliquer certains termes de base, et d'expliciter la structure d'un réseau neuronal. D'abord, le processus de reconnaissance optique de caractères sera expliqué sommairement, et servira d'exemple auquel pourront être rapportés les concepts qui suivront. Ensuite, les différentes composantes d'un réseau neuronal seront expliquées en ordre croissant de complexité, en débutant avec la plus simple unité, le neurone.

* Définition du mot OCR
   * Définir ce terme, et expliquer le fonctionnement global d'un type de réseau neuronal permettra au lecteur de relier les différents termes qui suivent à cet exemple afin
   de faciliter leur compréhension.

* Explication des neurones
   * Une connaissance adéquate de l'unité de base d'un réseau neuronal permettra de comprendre d'où proviennent divers concepts mathématiques qui sont nécessaires à son
      fonctionnement.
      * Structure d'un neurone
      * Poids et biais
      * Fonction d'activation (démonstration mathématique)
* Communication entre les couches
   * La manière les valeurs sont entrées dans le réseau, puis sont ensuite modifiées et propagées au travers du réseau est le fondement de comment un réseau neuronal apprend.
      * entrants et extrants
* Lien avec les neurones biologiques
   * L'analogie avec les neurones biologiques permet de visualiser plus facilement le fonctionnement d'un réseau neuronal, et fait un lien avec le terme 
      "intelligence artificielle".
    
## Apprentissage machine (explication à l'aide d'un programme)

Maintenant que le langage et certaines notions de base reliées aux réseaux neurals 
et à la reconnaissance optique ont été expliqués,
cette section-ci va expliquer en profondeur les différentes étapes à l’intérieur
d’un système neuronal. Le phénomène de l'apprentissage machine sera expliqué en 
détail afin de comprendre d'où provient le biais et comment l'intelligence 
artificielle peut être autant pratique et puissante. Certaines démonstrations
mathématiques permettront d’expliquer en profondeur le fonctionnement de chaque 
étape. Enfin, un programme documenté fera un lien
entre la théorie démontrée et l'application pratique.

### Explication de la collecte et traitement de données
   * Les différentes étapes nécessaires avant de pouvoir utiliser les données d’une image comme la
   conversion en binaire. 

   * Expliquer ce fonctionnement est utile puisque sans ce processus, les images
   données ne pourraient pas être fiables en raison des défauts de l’image 
   {cite}`preprocessing`.

### Fonction de coût

   * Le coût est le résultat lors de l'entraînement d’un programme. Cette étape sera
   expliquée et la fonction sigmoïde sera démontrée. Cette étape permet de 
   comprendre le fonctionnement de l’apprentissage machine global {cite}`Michael`.
   
### Explication de la modification des paramètres

   * Un paramètre modifié a un grand impact sur le reste du résultat. Comprenant
   la fonction de coût et en utilisant son résulat cette étape permettra de 
   comprendre l’impact des changements à l’intérieur du système neuronal et ainsi
   mieux comprendre le fonctionnement de l'apprentissage machine.

### Programme sur la reconnaissance optique de caractère

   * Un programme écrit sera documenté afin de comprendre l’aspect pratique des 
   notions théoriques précédemment expliquées. Cela va permettre d’avoir un 
   exemple tangible de l’utilisation et du fonctionnement de l’intelligence 
   artificielle.

### Impact de l'intelligence artificielle

   Il est important de parler de l'**Impact de l'intelligence artificielle** 
   dans le but de répondre à la thèse, c'est pour cela que dans cette section,
   on y retrouve les bienfaits et les inconvénients de l'intelligence artificielle.
   Ceux-ci permettront d'identifier si l'impacte des bienfaits est plus important
   que celui des inconvénients , et donc répondre à la sous-question de notre thèse: 
   Est-ce que l'intelligence artificielle peut constituer un bénéfice pour l'être humain?


### Bienfaits

   La sous-section **Bienfaits** permettra de montrer et expliquer les bienfaits de
   l'intelligence artificielle dans plusieurs sphères pour l'être  humain.
   
### Inconvénients

   La sous-section **Inconvénients** permettra de montrer et expliquer les inconvénients de
   l'intelligence artificielle dans plusieurs sphères pour l'être  humain.

## Conclusion
   Cette section a pour but de rappeler l'hypothèse de départ et de la comparer avec
   les résultats obtenus lors d'expérimentation et d'expliquer les différences entre ceux-ci.
   Tout cela dans le but d'utiliser cette information pour construire et formuler la réponse 
   finale à la thèse: *Quel est le fonctionnent de l'intelligence artificielle et comment devrait-elle 
   être utilisée afin de bénéficier l’être humain?*. Cette section se termine 
   par une ouverture globale de la thèse et du projet pour complémenter celle-ci
   et laisser place à d'autres questions à répondre.
   
### Retour sur l'hypothèse 

   Le **Retour sur l'hypothèse** permet d'avoir une rétroaction sur l'hypothèse dans le but de comparer 
   celle-ci avec les résultats trouvés après les tests et les recherches et en comparer leurs différences
   et leurs ressemblances et d'assembler la réponse à la thèse.

### Réponse à la question

   La **Réponse à la question** est la réponse finale à la thèse émise au début du projet basé sur les 
   tests et les recherches ainsi que l'hypothèse.

### Ouverture

   L'**ouverture** est une question ouverte qui permet, non pas de répondre à une 
   question spécifique, mais d'ouvrir plusieurs champs de réflexions ainsi que 
   des pistes sur  l'intelligence aritifielle dans le monde présent.
    

## Méthodologie

Pour plus d'information sur notre méthodologie, voir le
[README](https://github.com/TrickyTroll/ML-intro/blob/master/README.md)
de notre projet. Nous y mentionnons comment nous avons modifié la
configuration de nos outils afin de les rendre compatibles avec les
normes du cégep.

### Markdown

Pour écrire notre rapport, nous utilisons le langage de *markup*
[MyST](https://myst-parser.readthedocs.io/en/latest/). Beaucoup plus facile
d'accès et efficace que LaTeX, il permet tout de même la réalisation de
rapports scientifiques de qualité possible avec
[ses multiples fonctionnalités](https://jupyterbook.org/content/myst.html).

### Google Colab

[Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
nous permet de réaliser nos calculs dans le nuage sur les serveurs Google.
Il est donc facile de partager notre progrès, et nos calculs peuvent être
réalisés sur n'importe quel ordinateur possédant un navigateur web. Avec
Google Colab, le lecteur pourra aussi facilement lancer son propre notebook
lors de sa lecture afin de tester nos fonctions. La version de base de
l'application est gratuite.

### Jupyter Book

[Jupyter Book](https://jupyterbook.org/intro.html) est un nouveau projet
permettant de créer des publications scientifiques à l'aide de notebooks
[Jupyter](https://jupyter.org/) et du [Markdown](https://commonmark.org/help/).
Nous avons choisi cet outil puisqu'il nous permet d'intégrer nos calculs
réalisés dans un environnement de calcul interactif Python ainsi que notre
texte décrivant les procédés. Il sera donc plus facile pour le lecteur de
comprendre les concepts de programmation puisque des explications pourront
accompagner chaque cellule de code.

### Github

Grâce à Github, la séparation des tâches en sous-section est beaucoup plus 
facile, donc chacun des coéquipiers aura leurs sous-section pour travailler 
dedans. Ces sous-sections seront les grandes lignes de notre plan: Le projet 
consiste aussi un produit qui est un programme, celui-ci ce ferra sur le côté, 
plus comme enrichissement au projet et pourra être travaillé à la guise des 
membres de l'équipe ainsi qu'à leur habilité en programmation.

