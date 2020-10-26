# Plan de rédaction du travail

**Comment allons nous démontrer notre thèse en 3 ou 4 grandes étapes. Comment
ces étapes sont-elles utiles**

*Quel est le fonctionnent de l'intelligence artificielle et comment devrait-elle
être utilisée afin de bénéficier l’être humain?*

## Choix de la thèse

L'intelligence artificielle est au coeur de nos vies présentement et a évolué
a une vitesse fulgurante au courant de la dernière décénie. Les multiples 
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

<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/2213_RC01/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"/m/0h1fn8h","geo":"","time":"all"},{"keyword":"machine learning","geo":"","time":"all"}],"category":0,"property":""}, {"exploreQuery":"date=all&q=%2Fm%2F0h1fn8h,machine%20learning","guestPath":"https://trends.google.com:443/trends/embed/"}); </script> 

Les tendances de recherche à l'aide du moteur [Google](https://www.google.com/) 
pointent aussi dans la même direction. L'intérêt dans le temps pour les termes
«Deep learning» et «Machine learning» a beaucoup augmenté récemment 
{cite}`trendsml`. Par contre, la popularité du terme «Artificial Intelligence» 
ne semble pas varier dans la même direction {cite}`trendsai`. Il faut alors prendre en 
considération que l'intelligence artificielle est un domaine qui date, et que
le moteur de recherche n'existe que depuis le début des années 2000.

En se basant sur les statistiques mentionnées plus haut, il est possible
d'affirmer que l'IA est un domaine d'actualité avec lequel la population se
fait de plus en plus familière. De multiples articles traitent déjà du sujet, 
une grande partie d'entre eux mentionnant comment son utilisation pourrait
bénéficier l'être humain.

Le principal problème avec ces articles est qu'ils ne permettent pas au lecteur
de réellement saisir comment un programme peut être entraîné à réaliser des 
tâches qui témoignent d'une intelligence humaine. Une conaissance de ces
concepts est nécessaire au développement d'un bon jugement critique face
aux bonnes et mauvaises utilisations de la technologie.

C'est pourquoi notre thèse inclus non seulement la manière dont l'IA pourrait
être utilisée pour bénéficier l'être humain, mais aussi quel est le 
fonctionnement de la technologie.

Nous allons d'abord introduire le sujet en mentionnant comment l'intelligence
artificielle a évolée depuis sa création. Nous allons ensuite décrire en détails
les composantes d'un _machine learing pipeline_ ainsi que les mathématiques qui
y sont associées. À l'aide des ces composantes, nous alons créer un modèle
permettant la reconnaissance optique de charactères. Par la suite, nous 
analyserons les capacités de notre modèle, mettant beaucoup d'emphase sur les
limites de celui-ci. Pour conclure, nous utiliserons les connaissances
nouvellements acquises afin de porter un regard critique sur les utilisations
potentielles de l'IA.



## Un bref historique

* Recherche et documentation sur l'histoire de l'intelligence ainsi que ses
utilités dans le passé.

* L'intelligence artificielle présentement

* Formulation de l'hypothèse

### Les débuts de l'IA

### Les grandes avancées

* Big data
* GPUS



## Notion de base des procédés
* Définition du mot OCR

* Explications des neurones
    * Structure d'un neurone
    * poids
    * biais
    * entrants et extrants
* Communication entre les couches
    * Fonction d'activation (démonstration mathématique)
    * lien avec les neurones biologiques
    
<<<<<<< HEAD
## Apprentissage machine (explication à l'aide d'un programme)
* Explication de la collecte et traitement de données
    * Préparation des données
=======
## Apprentissage machine (explication à l'aide d'un programme)  
Maintenant que le langage et certaines notions de base relié aux réseau neural et à la reconnaissance optique expliqué,
cette section-ci va expliquer en profondeur les différentes étapes à l’intérieur d’un système neuronal. Le phénomène de l'apprentissage machine sera expliqué en détail afin de comprendre d'où provient le biais et comment l'intelligence artificielle peut être autant pratique et puissante. Certaines démonstrations
mathématiques permettront d’expliquer en profondeur le fonctionnement de chaque étape. Enfin, un programme documenté fera un lien
entre la théorie démontrée et l'application pratique.
>>>>>>> 4627b15ce5255acbf1fd7b14feca250761ea69d4

* Explication de la collecte et traitement de données
   * Les différentes étapes nécessaires avant de pouvoir utiliser les données d’une image comme la binarisation. 
   Expliquer ce fonctionnement est utile puisque sans ce processus, les images données ne pourraient pas être fiable
   en raison des défauts de l’image.{cite}`preprocessing`
* Fonction de coût
   * Le coût est le résultat lors de l'entraînement d’un programme. Cette étape sera expliqué et la fonction sigmoïde sera démontrée. 
   Cette étape permet de comprendre le fonctionnement de l’apprentissage machine global.{cite}`Michael`
* Explication de la modification des paramètres
   * Un paramètre modifié a un grand impact sur le reste du résultat. Comprenant la fonction de coût et en utilisant
   son résulats cette étape permettra de comprendre l’impact des changements à l’intérieur du système neuronal et ainsi
   mieux comprendre le fonctionnement de l'apprentissage machine
* Programme sur la reconnaissance optique de caractère
   * Un programme écrit sera documenté afin de comprendre l’aspect pratique des notions théoriques précédemment expliquées. Cela va permettre d’avoir un exemple tangible de l’utilisation et du fonctionnement de l’intelligence artificielle.

## Impact de l'intelligence artificielle
   Il est important de parler de l'**Impact de l'intelligence artificielle** 
   dans le but de répondre à la thèse, c'est pour cela que dans cette section,
   on y retrouve les bienfaits et les incovénients de l'intelligence artificielle.
   Ceux-ci permettront d'identifier si l'impacte des bienfaits est plus important
   que celui des incovénients , et donc répondre à la sous-question de notre thèse: 
   Est-que l'intelligence artificielle peut constituer un bénéfice pour l'être humain?


* Bienfaits

   La sous-section **Bienfaits** permettera de montrer et expliquer les bienfaits de
   l'intelligence artificielle dans plusieurs sphères pour l'être  humain.
* Inconvénients

   La sous-section **Inconvénients** permettera de montrer et expliquer les incovénients de
   l'intelligence artificielle dans plusieurs sphères pour l'être  humain.

## Conclusion
   Cette section à pour but de rappeller l'hypothèse de départ et de la comparer avec
   les résultats obtenues lors d'expérimentation et d'expliquer les différences entre ceux-ci.
   Tout cela dans le but d'utiliser cette information pour construire et formuler la réponse 
   finale à la thèse: *Quel est le fonctionnent de l'intelligence artificielle et comment devrait-elle 
   être utilisée afin de bénéficier l’être humain?*. Cette section fini par une ouverture global de 
   la thèse et du projet pour complémenter celle-ci et laisser place à d'autres questions à répondre.
   
* Retour sur l'hypothèse 

   Le **Retour sur l'hypothèse** permet d'avoir une rétroaction sur l'hypothèse dans le but de comparer 
   celle-ci avec les résultats trouvés après les tests et les recherches et en comparer leurs différences
   et leurs ressemblances et d'assembler la réponse à la thèse.
* Réponse à la question

   La **Réponse à la question** est la réponse finale à la thèse émit au début du projet basé sur les 
   test et les recherches ainsi que l'hypothèse.
* Ouverture

   L'**ouverture** est une question ouverte qui permet, non pas de répondre à une question spécifique, mais d'ouvrir
   plusieurs champs de réflections ainsi que des pistes sur  l'intelligence aritifielle dans le monde présent.
    
