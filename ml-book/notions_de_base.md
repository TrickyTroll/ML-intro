## Notions de base

### Introduction au réseau neuronal
Un réseau neuronal est une forme d'intelligence artificielle, qui effectue des prédictions basées sur des valeurs qui sont entrées dans le système, 
afin d'accomplir une certaine tâche. Le réseau est constitué d'un ensemble de neurones interconnectés et distribués en plusieurs couches. 

Chaque neurone possède des paramètres qui peuvent être ajustés, afin d'obtenir des résultats plus fiables. C'est ce qu'on appelle l'entrainement.
Le réseau est entrainé à partir d'un jeu de données, qui contient des valeurs associées à une étiquette, qui consiste de la "réponse" attendue. 

Par exemple, un réseau neuronal ayant comme objectif de prédire l'achalandage dans un parc d'amusement pour une journée donnée pourrait recevoir 
comme intrant la température, le niveau d'ensoleillement ainsi que le pourcentage de précipitation et d'humidité. Le jeu de données serait alors 
constitué d'une liste ces quatre valeurs enregistrées à chaque jour des dernières années, avec comme étiquette le nombre de clients cette journée-là. 
Les réponses du réseau sont comparées aux étiquettes, et les paramètres des neurones sont individuellement modifiés de manière à se rapprocher de la réponse attendue.


```{figure} ./img/resneuronalsimp.png
---
name: Réseau neuronal
---
Ceci est un exemple simplifié d'un réseau neuronal. Les composantes du schéma seront expliquées en détail dans cette section.
```
 



### OCR
Le terme OCR, ou ROC en français, signifie "Reconnaissance optique de caractères". Cela désigne un processus au cours duquel du texte est extrait 
d'une image ou d'un document afin d'être transformé en fichier. Pour ce faire, un réseau neuronal reçoit les valeurs des pixels du document de source,
>Note : La valeur d'un pixel en "grayscale" ou échelle de gris, est un nombre entier
>de format 8 bits et peut donc avoir
>une valeur comprise entre 0 et 255 (2^8 - 1), où 0 est noir et 255 est blanc. 
>Un pixel en couleur est représenté sous la forme d'un vecteur de 3 nombres 8
>bits, chaque nombre correspondant à une valeur de rouge, vert et bleu. {cite}`HIPR2`
                    
  traitées afin de les rendre utilisables par le réseau. Ces données se propagent ensuite vers l'avant
dans le réseau, de couche de neurone en couche de neurone, avant d'aboutir à la couche d'extrants, composée de 10 neurones dans le cas de notre 
programme, qui correspondent aux chiffres de 0 à 9. Un de ces neurones de cette couche finale s'active, donnant ainsi le résultat estimé par le réseau. 
Ensuite, divers paramètres sont ajustés par un algorithme d'optimisation afin d'augmenter la précision des réponses du réseau.



### Le neurone
Le neurone est l'unité de base d'un réseau neuronal. C'est un noeud parmis le réseau par lequel transitent des valeurs, qui sont modifiées 
au passage par un procédé qui sera expliqué plus en détail prochainement, avant d'être envoyées vers les prochains neurones. 
Essentiellement, un neurone reçoit une ou des valeurs comme intrant, effectue des opérations sur ces dernières, puis transmet la nouvelle valeur.

La structure d'un neurone est relativement simple. Chaque neurone possède un coefficient, ou un **poids** $p$ dans le jargon, associé à chaque **intrant** $I$ qu'elle reçoit.
La première opération que la neurone effectue est la somme des produits des intrants fois leur poids. À celà est ajouté un **biais** $b$ propre à chaque neurone.
Cette opération peut être représentée par la fonction 

$Y = \sum_{i=1}^{n} I_i \times p_i + b$

, où n correspond au nombre d'intrants.

La dernière opération que les valeurs subissent avant d'être transmises est une fonction d'activation. La fonction d'activation est appliquée à chaque extrant de chaque
neurone de la couche. Les fonctions d'activation, analogues à l'activation
d'un neurone biologique, permettent généralement d'obtenir un extrant compris entre 0 et 1, ou -1 et 1. Elles ont plusieurs utilités, notamment 
pour la modélisation de fonctions non linéeaires, ainsi que pour l'entrainement du réseau, ce qui sera expliqué dans une section ultérieure.

```{figure} ./img/neurone.png
---
name: Neurone
---
Exemple des opérations effectuées au sein d'un neurone.
```

La fonction la plus simple est la fonction à échelons. Elle retourne 1 si l'intrant *x* est plus grand qu'une valeur seuil *s*, et 0 s'il ne l'est pas. Cette fonction peut être représentée par l'équation

$
E(x)=
\begin{cases}
 1 & \quad \text{si } x \text{ > s}\\
 0 & \quad \text{si } x \text{ <= s}
\end{cases}
$

Elle n'est néanmoins pas utilisée, puisqu'elle empêche l'entrainement du réseau.
La fonction d'activation doit être dérivable en une autre fonction, et non en une constante, afin que le processus d'ajustement des paramètres puisse avoir lieu. 
Il est également impossible de représenter des situations non-linéaires avec cette fonction, puisque seulement des fonctions linéaires sont présentes dans le réseau.

La fonction d'activation la plus utilisée est la fonction Unité Linéaire Rectifiée, ou "ReLU" en anglais (Rectified Linear Unit).
Cette fonction peut être représentée par l'équation :  

$
R(x)=
\begin{cases}
 x & \quad \text{si } x \text{ > 0}\\
 0 & \quad \text{si } x \text{ <= 0}
\end{cases}
$
                                                        
ou encore, $ R(x) = max(0, x)$. Cette fonction est peu demandante à calculer pour l'ordinateur, et se fait très rapidement. De plus, malgré son apparence linéaire,
elle peut être dérivée, ce qui est nécessaire pour pouvoir entrainer le réseau. C'est pour ces raisons que c'est la fonction d'activation la plus répandue.
Elle a toutefois comme désavantage de produire parfois une trop grande quantité de "0", ce qui peut entrainer une réaction en chaine, où ces zéros se propagent, 
empêchant le bon fonctionnement du réseau. Cette situation est appelée la "mort du réseau", où l'extrant de plusieurs neurones devient invariablement 0, ce qui 
diminue l'efficacité du réseau. Ce phénomène se produit surtout lorsque le réseau se fait entrainer de manière trop rigoureuse, et que le biais de certains 
neurones devient une très grande valeur négative, ce qui fait que l'intrant dans la fonction d'activation est toujours en dessous de 0, et l'extrant reste ainsi 
invariablement 0.


Une variation de cette fonction, nommée Leaky ReLU, a été créée afin de tenter de régler ce problème de mort du réseau : 

$ 
L(x)=
\begin{cases}
 x & \quad \text{si } x \text{ > 0}\\
 0,01 \times x & \quad \text{si } x \text{ <= 0}
\end{cases}
$
                                                         
Ici, les zéros sont remplacés par de très petits nombres négatifs, qui correspondent généralement à x multiplié par le coefficient 0,01. 


Une autre fonction commune est la sigmoide. Son équation est : 

$ \phi(x) = 
\frac{1}{1 + e^{-x}}
$

La fonction retourne 0 lorsque x tend vers l'infini négatif, et 1 lorsque x tend vers l'infini positif. Cette fonction a comme avantage de 
s'approcher rapidement de 0 ou de 1, lorsque l'intrant *x* est plus petit que -2 ou plus grand que 2, respectivement. Cela permet d'envoyer 
un signal très fort aux prochains neurones. Cela peut toutefois devenir un désavantage lorsque les intrants sont très grands, puisque l'extrant 
reste pratiquement le même, ce qui peut nuire à l'entrainement. Cette fonction est également plus lourde pour l'ordinateur, ce qui peut ralentir 
considérablement le système lorsque ce calcul est effectué des centaines ou des milliers de fois.

Une fonction similaire à la sigmoide et la TanH. Son équation est :

$ tanh(x) = 
\frac{2}{1 + e^{-2x}} - 1
$

Elle retourne -1 lorsque x tend vers l'infini négatif, et 1 lorsque x tend vers l'infini positif. Elle a comme avantage de retourner en moyenne
des valeurs proches de 0, ce qui rend la tâche plus facile pour les couches suivantes, puisque les valeurs auront moins tendance à devenir très grandes, 
ce qui ralentirait les opérations.


### Couches de neurones

Comme mentionné précédemment, les neurones sont organisés en couches. Il y a 3 types de couches différentes. La première est la couche des intrants, dans laquelle 
les données sont rentrées dans le réseau. Dans le cas de notre programme, où les intrants sont des images de format 28x28, 
la première couche est composée de 784 ($28\times28 = 784$) neurones recevant chacun la valeur en échelle de gris d'un pixel de l'image. 


Plus concrètement, ces images sont des matrices carrées $M_{28}$, qui se font vectoriser en un vecteur de taille 784. Par la suite, chacune de ces
données est transmise à chacun des neurones de la couche cachée, puisque le réseau est densément connecté, et les neurones d'une couche sont connectés à
tout ceux des couches adjacentes. Pour la suite de cette explication, le réseau neuronal provenant de la figure affichée plus haut sera utilisé, à des
fins de clarté. Donc, les valeurs des trois neurones de la couche d'intrants sont contenus dans la matrice $I_{1\times3}$. Les poids des neurones de la couche cachée 1 sont
contenus dans la matrice $C_{4\times3}$, où 4 correspond au nombre de neurones dans la couche, et 3 aux poids que possèdent chaque neurones de la couche (un poids par neurone
de la couche précédente). Ici, l'opération à faire serait un produit matriciel 
$A_{m\times p} \times B_{p\times n} = C_{m\times n}$
, afin de multiplier les intrants par chaque ensemble de poids. Toutefois, les matrices ne sont
pas compatibles pour effectuer cette opération, puisque le nombre de colonnes de la première matrice n'est pas égal au nombre de rangées de la seconde. 
Il faut donc faire la transposée de la matrice $C_{4\times3}$, qui devient alors $C_{3\times4}^{t}$. L'opération $I_{1\times3} \times C_{3\times4}^{t}$, où sont multipliés
dans l'ordre, élément par élément, chaque élément d'une ligne de *I* par chaque élément d'une colonne de *C*, puis est effectué la somme de 
ces produits pour obtenir un nouvel élément de la matrice résultante $R_{1\times4}$ {cite}`Alloprof`. 

Par la suite, la matrice $B_{1\times4}$ contenant les biais de chaque neurone 
de la couche est additionnée à la matrice R, dans une opération où s'additionnent entre eux les éléments correspondants de chaque matrice pour 
former une nouvelle matrice de même dimension. Finalement, dans une itération au travers de cette matrice, chaque élément passe par la fonction d'activation, 
pour former encore une nouvelle matrice de même dimension contenant les résultats de cette dernière opération. Cette matrice résultante finale $F_{1\times4}$ devient 
alors l'intrant de la couche suivante de neurones, et ainsi de suite. 

### Réseaux neuronaux et le cerveau humain
Plusieurs liens peuvent être faits entre les réseaux neuronaux et le cerveau humain. Le premier réseau neuronal était un 
système mécanique financé par la marine américaine qui tentait d'émuler les neurones biologiques. La fonction d'activation
à échelons était utilisée, imitant les neurones biologiques qui s'activent *1* ou ne s'activent pas *0*. Le projet a rapidement
été laissé de côté, principalement à cause du fait que le réseau était extrêmement difficile à entrainer, puisque, comme vu plus
tôt, la fonction à échelon ne permet pas l'entrainement du réseau, et les paramètres devaient être ajustés au hasard. 

Voilà donc une première différence fondamentale entre les neurones artificiels et organiques. Les neurones artificiels peuvent 
sortir toutes sortes de valeurs, de manière à mieux servir les intérêts du système, alors que dans le cas d'un neurone organique,
elles ne peuvent envoyer que le signal binaire *activé* ou *non-activé*. 

>Un neurone s'active lorsque son seuil d'excitation est atteint. Le potentiel de repos d'un neurone est d'environ -50mV. Lorsqu'il 
reçoit suffisamment de neurotransmetteurs (des particules envoyées par d'autre neurones et qui possèdent une charge électrique), par 
ses dendrites et que le seuil d'excitation d'environ 15mV est atteint, le potentiel d'action se déclenche, et un influx nerveux se propage
le long de l'axone sous forme de courant électrique. Une fois arrivé aux terminaisons axonales du neurone, d'autres neurotransmetteurs sont
libérés par les synapses, poursuivant ainsi la transmission du signal. La quantité de neurotransmetteurs libérée ne dépend pas de l'intensité
du stimulus initial ; c'est une situation de tout ou rien. {cite}`futura-sciences`

En d'autres termes, l'image de la fonction d'un neurone artificiel *A*, dépendamment de la fonction d'activation, peut être, 
par exemple $\mathbb{R}$ , $\mathbb{R^+}$, ou encore $[-1, 1]$, 
tandis que l'image de la fonction d'un neurone organique *O* est toujours limité à $\text{{0, 1}}$. 

L'aspect où les réseaux neuronaux et le cerveau humain ont le plus en commun est leur état initial. Les deux commencent comme un canevas vierge, ne possédant
aucunes connaissances ou expériences. Les deux se font "entrainer" par des informations extérieures, jusqu'à arriver au point où ils deviennent autonomes.
Les connaissances qu'ils amassent se trouvent, d'une certaine manière, encodées dans leur système, et influencent leurs actions futures.

