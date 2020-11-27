## L'entrainement d'un système neuronal
L'entrainement d'un réseau à l'aide d'une certaine base de données (donnée d'entrainement) permet à celui-ci de prédire le résultat
d'une autre base donnée. En effet, le but d'un réseau neuronal est de réduire l'erreur de l'entrainement ainsi que la différence
entre l'erreur des données entrainées et l'erreur des données de test soient petites. Lorsque le réseau est sous-entrainé,
le réseau de sera pas précis lors de ces résultats. Cependant, lorsque le réseau est sur-entrainé, celui-ci va prendre en compte
tout le bruit des données. Ce bruit peut être, par exemple, le fait de prendre en compte les imperfections d'une image, reconnaitre
seulement certains styles d'écriture, etc. Cela a comme impact d'augmenter l'erreur lorsque le système est exposé à une nouvelle base de données.

```{figure} ./img/overfitting.png
---
name: overfitting
---
Graphiques représentant l'effet de l'entrainement du réseau de neurone
```
L'entrainement d'un réseau neuronal s'effectue à l'inverse. Visuellement, l'entrainement et l'ajustement des différents
paramètres se font de la droite vers la gauche. Ce principe, appelé "backpropagation", va être expliqué à l'aide quelques
démonstrations mathématiques complémentées par quelques explications écrites.

### Fonction d'erreur
Une fonction d'erreur est une fonction permettant de connaitre la précision des résultats des extrants de la dernière
couche. Il peut y avoir plusieurs fonctions d'erreur. En voici un exemple:
  
$E_{SS}=1/2\sum_{i=1}^nE_i^2 $ 

__(1.1)__

où $E_{SS}$= "error sum of square". Cela est tout simplement une de plusieurs fonctions d'erreur.

$E_i =|{t_i-I_i}|$ 

__(1.2)__

où $E_i$ correspond à l'erreur d'un neurone de la dernière couche (extrant). $I_i$ correspond à la valeur numérique
d'un extrant et $t_i$ correspond à la valeur désirée provenant de la base de données fournies.

Combiner les deux équations permet d'obtenir:

$E=1/2\sum_{i=1}^n({T_i-Y_i})^2$ 

__(1.3)__

### Transmition de l'information

>Note: Afin de simplifier les explications, ces dernières seront faites en utilisant un réseau neuronal ayant seulement 1 >neurone par couche. 

D'abord, il faut comprendre comment le réseau transmet son information de cellules en cellule. En effet,
un neurone ayant contenant une certaine valeur $Y$ transmet cette dernière à tous les autres neurones de
la prochaine couche. Cependant, ces transmitions n'ont pas toutes les mêmes poids. Ces poids $p$ diffèrent
afin de favoriser certaines activations et en défavoriser d'autres. Chaque liaison entre chaque neurone possède
un poid propre à chacune. Ces derniers sont multipliés avec l'extrant de la neurone en précédentes.

$Y_{i} = Y_{i-1}\times p_{i}$
 
 __(2.0)__ 
 

où $p_{i}$ correspond au poid de la neurone de la couche i

Ensuite, un biais $b$ est additionné ou soustrait au résultat précédent

$Y_i = Y_{i-1}\times p_{i} + b_i$ 

__(2.1)__  

d’activation sera expliqué en détail plus loin.où $b_i$ correspond au biais de la neurone de la couche i.

Finalement, une fonction d'activation $a$ est ajoutée au reste de la formule. L'utilité et le fonctionnement de
la fonction d'activation sera expliqué en détail plus loin.

$Y_i = a\times(Y_{i-1}\times p_{i} + b_i)$ 

__(2.2)__ 

### *Back propagation*

L'objectif est de comprendre comment le poids et le biais doit être ajuster en débutant de la fonction d'erreur et d'activation.

Dabord, en utilisant la formule de base de transmission d'un neurone (sans le biais) :

$Y = \sum_{i=1}^{n} I_i \times p_i $

Il est possible de comprendre comment le changement d'une variable impact une autre. Les dérivés seront
donc utilisées afin de démontrer ce principe.

$\frac{dY}{dI_i}=\frac{dY}{dI_i}\sum_{i=1}^{n} I_i \times p_{ji} $

$\frac{dY}{dI_i} = p_i$

$\frac{dY}{dp_i} = I_i$


Cela veut donc dire que le poid influence le résultat de l'extrant et que l'intrant influence
le résultat de l'extrant. 

En utlisant la formule (1.4) et le concept de dérivée partielle, il est possible de comprendre
l'impact d'un changement de la valeur de l'intrant $I_i$ sur l'erreur:

$\frac{dE}{dI_i} =  (2/2)(t_i - I_i)(-1) $
$\frac{dE}{dI_i}= -(t_i-I_i)$

Maintenant, il faut calculer la dérivation de la fonction d'activation.
La fonction sigmoïde sera utilisée pour cet exemple.

$a = \frac{1}{1 + e^{-Y}} =(1+e^{-Y})^{-1}$

$\frac{da}{dY} = -1 (-e^{-Y})(1+e^{-Y})^{-2} $

$= \frac{e^{-Y}}{(1+e^{-Y})^2} $

$= \frac{1}{(1+e^{-Y})}\times\frac{e^{-Y}}{(1+e^{-Y})} $

$= a \times \frac{e^{-Y}}{(1+e^{-Y})}$

$= a \times \frac{1+e^{-Y}-1}{(1+e^{-Y})} $

$= a \times (\frac{(1+e^{-Y})}{(1+e^{-Y})} + \frac{-1}{(1+e^{-Y})})$

$\frac{da}{dY}= a \times (1-a)$


Maintenant il est possible, à l'aide de la règle de dérivation en chaine, de trouver l'impact
qu'a $Y$ sur l'erreur $E$. Dans cet exemple, $I_i = a $ puisque la fonction d'activation été appliquée au neurone en question.

$\frac{dE}{dY_i} = \frac{dE}{dI_i} \times \frac{dI_i}{dY_i}$

$=\frac{dE}{dI_i} \times \frac{da}{dY_i}$

$=-(t_i - I_i)  I_i (1- I_i)$  

__(3.0)__

Ensuite il est possible de calculer la dérivation de l'erreur en fonction du poid $p_{ji}$ d'une liaison entre deux neurones.

$\frac{dE}{dp_{ji}} =\frac{dE}{dY_i} \times \frac{dY_i}{dp_{ji}} $

$= (-(t_i - I_i) \times I_i\times (1- I_i))\times I_i$

$= -I_i I_j (1-I_i)(t_i-I_i)$

__(4.0)__


Cette équation signifie que le changement de l'erreur influence le poid et cette influence
correspond à l'extrant d'un neurone négatif multiplié par l'extrant du neurone précédent et
ce tout est multplié par 1 moins la valeur du neurone. À ce résultat est multiplié la valeur
de l'erreur soit : $(t_i-I_i)$
L'équation 3.0 sera représenté par la variable:  $\Delta p$  

Le concept de "backpropagation" se résume donc a:

$p_{ji} = p_{ji} + \Delta p$

Le poids d'un neurone change légèrement en additionnant un $\Delta p$  positif ou négatif. Ce changement
est fait avec une plus grande importance plus le neurone est proche de la couche des extrants. Cela est
dû au fait que l'apprentissage commence par la couche finale pour enfin se rendre jusqu'à la couche débutant
le système neuronal. Les premiers ajustements, donc ceux des couches plus proches de la fin, sont plus importants.
Les couches se situant plus au début du réseau vont plutôt avoir de petits changements à leur poids puisque
rendu à l'ajustement de dce dernier, l'erreur est déja considérablement réduite. Ce concept se nomme descente
de gradient stochastique. 

### Origine du biais
Certains problèmes peuvent survenir avec le concept de descente de gradiant dans un réseau neuronal.
En effet, lorsqu'une couche "n'apprend plus" ou, en d'autres mots, lorsque le poids ne varie plus,
on assiste a un problème se nommant la disparition du gradiant. Cela est un problème pour le réseau puisque
le tout l'entrainement se fait uniquement dans les dernières couches. Un autre problème est que le gradiant
dans une fonction de coût telle une sigoïde, le gradient se situe uniquement au milieu de la fonction comme
le montre le graphique ci-dessous.

```{figure} ./img/tanh.png
---
name: tanh
---
Fonction sigmoïde
```

En effet, les extrémités de la fonction forment un plateau. Il n'y a donc pas de changement
possible puisque soit l'intrant est multiplié par 1, ce qui ne change pas le résultat, ou bien
soit le résultat est multiplié par 0 ce qui rend la valeurr nul et cela est néfaste pour un réseau neuronal.
En effet, une valeur égale a 0 empêche l'entrainement du réseau puisque peut importe la variation
du poid, $0\times p$ sera toujours être égale à 0. C'est pour remédier à cette erreur qu'un biais est ajouté
à la fonction.

$Y =\sum_{i=1}^{n} I_i \times p_i + b$

L'ajout de ce biais va permettre de conserver un apprentissage même lorsque la valeur d'un neurone est figée à 0.
