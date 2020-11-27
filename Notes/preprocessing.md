<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax: {inlineMath:[['$','$']]}});</script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=default' async></script>

# Traitement antérieur à l’entrainement

Dans cette section, nous discuterons du traitement nécessaire afin d’utiliser des images pour entrainer un réseau neuronal. Nous discuterons aussi de l’importance de ce traitement, ainsi que de la raison pour laquelle il doit aussi être réalisé sur les images que nous voudrons par la suite reconnaître.

## Pourquoi faire du *preprocessing*?

Comme nous en avons discuté dans la section précédente, notre programme utilise des méthodes fournies par la librairie `Tensorflow` afin de charger les données dans le bon format. Malheureusement, dans une majorité des cas, les données ne vous seront pas fournies sur un plateau d’argent. Les programmes d’apprentissage machine visent à faire du calcul statistique sur le jeu de données fourni

```{note}
Dans le domaine de l'IA, un jeu de données représente l'ensemble des données traitées ainsi que leur étiquettes.
```

## Mise en bouche sur l’apprentissage machine

L’entrainement d’un modèle se rapproche beaucoup des mathématiques, plus précisément de la statistique, comme en témoigne le *Deep Learning Book* {cite}`Goodfellow-et-al-2016`. L’apprentissage machine vise à ingérer des quantités massives de données provenant de sources différentes. Par la suite, à l’aide de calculs statistiques, le programme tente de faire une certaine classification du jeu données. Selon le besoin, le programme pourrait alors poser une étiquette sur des données non étiquetées similaires à celles retrouvées dans le jeu de donnée {cite}`mitclassification`.   Le modèle pourrait aussi être entrainer afin de reconnaître des anomalies, grouper des informations similaires par classes et bien d’autres {cite}`wikisupervised`. Toutes ces informations seront discutées plus en détails au courant de la prochaine section. Ce qu’il est important de retenir, c’est qu’entrainer un modèle nécessite ***beaucoup*** de données.

```{admonition} Sur la signification de «beaucoup de données»
```python
In [ ]: train_data.shape
Out[ ]: (60000, 28, 28)
```
``

## Traiter beaucoup de données

Pour reprendre l’exemple précédent, la `shape` de l’objet `train_data`est une liste de 60 000 images représentées par des matrices carrées de dimension 28. Dans notre cas, si le programme passait tous les pixels un à un, il faudrait qu’il réalise séquentiellement $28 \times 28 \times 60 000 = 47 040 000$ opérations. Ce serait par exemple le cas dans une `for loop`. Bien que les ordinateurs modernes sont particulièrement rapides[^1], les modèles récents sont eux aussi entrainés avec des jeux de données de plus en plus massifs. Celui utilisé pour le service [Google Translate](https://translate.google.ca), par exemple, compte des milliards d’exemples {cite}`googledatasize.

Heureusement, il existe des méthodes permettant de paralléliser[^2] les opérations statistiques réalisées sur notre modèle.

### Le parallélisme

Pour réduire le coût monétaire et temporel de l’entrainement d’un modèle, la tâche peut être séparée sur plusieurs des coeurs[^3] de la machine. Pour se faire, nous profiterons des propriétés des matrices.

#### Les propriétés des matrices

Afin de trouver comment il serait possible de réaliser nos calculs en parallèle, analysons les propriétés des matrices.

##### La multiplication

Assumons les matrices de dimensions compatibles[^4] $A$ et $B$:

$[A \times B]_{i,j} = \displaystyle\sum_{k=1}A_{i,k}B_{k,j}$

Assumons aussi que la matrice $C$ est produite par l’opération $A \times B$ et que les matrices $A$ et $B$ sont carrées[^5].  Le calcul de $C$ pourrait alors être implémenté de la manière suivante.

```python
# Initialisation de la matrice de départ.
# Nous assumons que les matrices sont 3x3.
C = [[0,0,0],
    [0,0,0],
    [0,0,0]]
# Pour chaque rangée de la matrice A.
for i in range(len(A)):
	# Pour chaque valeur d'une rangée de la matrice B.
	for j in range(len(B[0])):
		# Pour chaque rangée de la matrice B
		for k in range(len(B)):
			# [AxB]_{i,j} += A_{i,k} * B_{k, j}
			C[i][j] += A[i][k] * B[k][j]
```

Quoi qu’assez simple à implémenter, cette façon de calculer $C$ est particulièrement inefficace. Alors que les matrices A et B augmentent en taille, le nombre d’opérations requises augmente…**au cube!** Si $A$ passe d’une matrice $2X2$ à une matrice $3X3$, chaque `for loop` doit être réalisée $n$[^6] fois de plus. Comme le programme contient 3 for loops imbriquées, si la première doit être faite $n$ fois de plus, alors c’est de même pour la deuxième, puis la troisième. Le calcul est alors $n \times n \times n = n^3$ fois plus complexe à réaliser {cite}`wikimatrixmulti`.

Heureusement, ce problème n’est pas sans issues. Reprenons l’équation de la multiplication de deux matrices.
$[A \times B]_{i,j} = \displaystyle\sum_{k=1}A_{i,k}B_{k,j}$
Dans ce cas, chaque élément de $C$ est produit par une sommation sur des multiplications d’éléments de $A$ et $B$. Il est aussi important de noter qu’aucun calcul pour un élément de $C$ dépend d’un calcul pour un autre élément de $C$[^7]. Il serait donc possible de calculer plusieurs éléments de $C$ en même temps!

Bien que le calcul en parallèle ne réduit pas l’ordre de complexité, il permet tout de même de diviser le temps requis par le nombre de coeurs utilisés[^8].

##### L’addition

L’addition de deux matrices compatibles[^9] se définit par l’addition de chacun des éléments homologues des deux matrices. Si nous reprenons les matrices carrées $A$ et $B$ utilisées plus haut, la somme de ces deux matrices serait:

$[A+B]_{i,j} = A_{i,j} + B_{i,j}$

Supposons que la matrice $C$ résulte de la somme de $A$ et $B$. Il est encore une fois possible d’affirmer que la valeur de $C_{i,j}$ ne dépend pas de la valeur de $C_{k,l}$. Il serait possible d’additionner chaque composante des deux matrices dans n’importe quel ordre en obtenant toujours le même résultat.

Encore une fois, l’addition de deux matrices peut être parallélisé afin de réduire le temps de calcul[^10].

##### La multiplication par un scalaire

Bien que la multiplication par un scalaire s’avère facile à réaliser à la main pour de petites matrices, l’opération doit tout de même être réalisée sur chaque élément de la matrice.
$\lambda \begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{d1} & x_{d2} & x_{d3} & \dots  & x_{dn}
\end{bmatrix} = 
\begin{bmatrix}
    \lambda x_{11} & \lambda x_{12} & \lambda x_{13} & \dots  & \lambda x_{1n} \\
    \lambda x_{21} & \lambda x_{22} & \lambda x_{23} & \dots  & \lambda x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \lambda x_{d1} & \lambda x_{d2} & \lambda x_{d3} & \dots  & \lambda x_{dn}
\end{bmatrix}$
Encore une fois, aucun résultat n’est dépendant d’un autre. Il serait donc possible d’effectuer plusieurs multiplications en même temps, puis grouper les résultats dans une matrice.

##### La sommation

Nous avons ici un calcul légèrement différent des autres. Dans le cas de la sommation des éléments d’une matrice de dimension $(1,n)$, le calcul s’avère commutatif en plus d’être associatif[^11]. La commutativité de l’addition permet à notre programme d’utiliser l’opérateur de réduction.
```{admonition} L'opérateur de réduction
Un opérateur de réduction permet de réduire les éléments d'un [tableau](https://fr.wikipedia.org/wiki/Tableau_(structure_de_données)) à un seul résultat. {cite}`wikireducop`
```


En premier lieu, voici comme une addition séquentielle d’un tableau pourrait être réalisé. Assumons un tableau de 8 entiers comme suit: `tableau = [2,9,6,4,1,3,8,8]`. L’addition pourrait alors être réalisée en ajoutant chaque nombre un par un jusqu’à obtenir le total.
```python
# Création du tableau.
tableau = [2,9,6,4,1,3,8,8]
# Initiation de la variable `somme`.
somme = 0
# Pour chaque chiffre dans le tableau.
for chiffre in tableau:
    # Sommation de l'ancienne somme avec le nouveau chiffre.
    somme += chiffre
# Affichage de la somme.
print(somme)
```
Cet exemple permettrait de réaliser une sommation séquentielle sur tous les chiffres contenus dans le tableau. Ce reviendrait à réaliser le calcul suivant:
$(((((((2+9)+6)+4)+1)+3)+8)+8)$
Bien que la moindre performance de cette méthode ne se fait pas ressentir pour des petites sommations, ce programme ne s’adapte pas bien à de grands tableaux[^12].

Ensuite, si l’addition n’était qu’associative, l’opération pourrait tout de même être parallélisée. Les sommes partielles pourraient être calculées indépendamment les unes des autres comme pour les autres opérations matricielles. Le calcul serait similaire à celui ci:
$((2+1)+(9+3))+((6+8)+(4+8))$
Dans cet exemple, les sommes $2+1$, $9+3$, $6+8$ et $4+8$ sont calculées en même temps. Par contre, lorsque l’une des opérations est complétée avant une autre, il arrive que l’ordinateur ait à attendre {cite}`stackoverflowcommutativity`. Le coeur ne pourrait alors pas se libérer pour faire d’autres opérations. Par exemple, assumons un ordinateur possédant deux unités de calcul disponibles et une addition non commutative. Si le calcul de $2+9$ était complété avant celui de $9+3$, l’ordinateur devrait attendre que les deux calculs soient complétés avant de calculer la somme partielle $(2+9)+(9+3)$. Heureusement, l’addition est associative. L’ordinateur ira donc écrire le résultat de la première opération complétée à la somme partielle, sans se soucier de la complétion de l’autre opération. L’unité de calcul sera alors libérée pour calculer, par exemple, la somme $6+8$.

Finalement , l’opérateur de réduction est beaucoup plus adapté aux échelles de l’intelligence artificielle. Encore une fois, l’ordinateur sépare la sommation en plusieurs petites opérations qui peuvent être exécutés en parallèle. De plus, l’opérateur profite de l’associativité pour optimiser la tâche au maximum. À la fin de la réduction, il ne reste qu’une seule addition. Le calcul mathématique serait par contre identique au précédent.
$((2+1)+(9+3))+((6+8)+(4+8))$
Il est possible de remarquer que l’addition est fait dans un ordre particulier. Cet ordre donne un meilleur modèle d’accès à la mémoire.

Python utilise l’opérateur de réduction lors du calcul de sommations. Implémenter ce genre de solution s’avère donc assez simple.
```python
tableau = [2,9,6,4,1,3,8,8]
somme = sum(tableau)
print(somme)
```

`NumPy` possède aussi une fonction de sommation optimisée pour les `numpy arrays`. Elle peut être implémentée tout aussi simplement.

```python
import numpy as np
tableau = np.array([2,9,6,4,1,3,8,8])
somme = np.sum(tableau)
print(somme)
```

##### En bref

En bref, une majorité des opérations matricielles peuvent être parallélisées. Les matrices sont donc la représentation de choix pour les jeux de données dans le domaine de l’intelligence artificielle. La transformation du jeu de données en matrices est une partie majeure du *preprocessing*. Elle permet d’accélérer le calcul d’un facteur non-négligeable.

### NumPy

C’est pour les opérations parallèles que la librairie `numpy`, mentionnée lors de la section précédente, entre en jeu. Les opérations matricielles réalisées à l’aide de méthodes implémentés par `numpy`profitent aussi de l’implémentation des `BLAS`[^13]. Les `BLAS` permettent de grandement accélérer nos calculs sans même nécessiter de coeurs supplémentaires. Elles exploitent plutôt les différentes architectures de processeur ainsi que leur différents niveau de cache[^14].

#### `BLAS`

Les sous-routines d’algèbre linéaire permettent de nettement réduire l’ordre de complexité des opérations d’algèbre linéaire. Elles permettent, par exemple, de décomposer des matrices en blocs afin d’accélérer la multiplication.

Ces sous-programmes sont extrêmement populaires. Ils sont implémentés dans une majorité des programmes de calcul scientifique {cite}`blaswebsite`.

#### Quelques résultats concrets

Voici quelques résultats plus concrets permettant d’obtenir une meilleure idée de l’ampleur de l’accélération des calculs.

##### Multiplication de matrices à l’aide de `NumPy`.

Dans ce programme, deux matrices de dimensions $1000 x 1000$ sont multipliées. La méthode de base avec les itérations imbriquées est comparée avec l’implémentation de l’opération par la librairie `NumPy`.

```python
import time
import numpy as np

# Création de nos matrices
A = np.random.rand(1000,1000)
B = np.random.rand(1000,1000)
C = np.zeros((1000,1000))

# Test de la première implémentation

start_time = time.time()

for i in range(len(A)):
	for j in range(len(B[0])):
		for k in range(len(B)):
		# 	print(A[i][k])
# 			print(B[k][j])
# 			print(i,j)
			C[i][j] += A[i][k] * B[k][j]
			
end_time = time.time()

time_1 = end_time - start_time

# Test de l'implémentation avec Numpy
start_time = time.time()

C = A*B

end_time = time.time()

time_2 = end_time - start_time

print("Run time 1 = {} seconds".format(time_1))
print("Run time numpy = {} seconds".format(time_2))
```

```
Run time 1 = 1838.9336512088776 seconds
Run time numpy = 0.005700111389160156 seconds
```

Alors que l’implémentation de base prend plus de 30 minutes à faire le calcul, `Numpy` n’a besoin de que moins de 6 millièmes de secondes[^15]!

#### Sommations de tableaux à l’aide de `NumPy`

Dans cet autre programme démontre quant à lui la différence entre notre implémentation de base de la sommation avec celle de `NumPy` sur un `numpy array`. La sommation se fait sur 1 milliard d’éléments aléatoires entre $0$ et $1$.

```python
import time
import numpy as np

# Création du tableau et initiation de la somme.
tableau = np.random.rand(1000000000)
somme = 0

# Test de la première implémentation.

start_time = time.time()

# Pour chaque chiffre dans le tableau.
for chiffre in tableau:
    # Sommation de l'ancienne somme avec le nouveau chiffre.
    somme += chiffre
			
end_time = time.time()

time_1 = end_time - start_time

# Test de l'implémentation avec Numpy.

# Réinitialisation de la somme.
somme = 0

start_time = time.time()

somme += np.sum(tableau)

end_time = time.time()

time_2 = end_time - start_time

print("Run time 1 = {} seconds".format(time_1))
print("Run time numpy = {} seconds".format(time_2))
```

La différence entre les résultats est encore une fois majeure.

```
Run time 1 = 361.87627387046814 seconds
Run time numpy = 5.082181215286255 seconds
```


## Représenter des images

Pour notre programme, l’intrant de notre réseau neuronal sera constitué d’images. Par contre, ces images ne seront pas directement passées au travers de notre programme dans leur format d’origine.

Comme discuté dans la section précédente, il serait préférable que les images soient représentées sous forme de matrices. C’est heureusement déjà le cas de nos données d’entrainement.

```python
In [ ]: type(train_data)
Out[ ]: numpy.ndarray
```
 
Dans cet exemple, `train_data` est un `array` contenant l’ensemble de nos images. Pour obtenir le nombre d’éléments dans cet `array`, la méthode `shape` peut être utilisée.

```python
In [ ]: train_data.shape
Out[ ]: (60000, 28, 28)
```

La première valeur correspond au nombre d’image dans nos données d’entrainement. Les deux valeurs suivantes sont le nombre de rangées et de colonnes des matrices utilisées pour représenter ces mêmes images. Ce sont donc des matrices carrées de dimensions $n = 28$.

Afin d’analyser précisément l’une de ces images, imprimons l’une des rangées de pixels.

```python
In [ ]: train_data[0][20]
Out[ ]: array([  0,   0,   0,   0,   0,   
                 0,   0,   0,   0,   0,  
                 24, 114, 221,  253, 253, 
                 253, 253, 201,  78,   0,
                 0,   0,   0,   0,   0,   0,
                 0,   0], dtype=uint8)
```

Les pixels sont représentés par des valeurs *grayscale* inversées. Traditionnellement, une valeur *grayscale* est élevée lorsque le pixel est très illuminé. La valeur maximale de `255`signifie un blanc, alors que le `0`correspond au noir. Dans notre jeu de données, ces deux valeurs sont inversées. Une valeur élevée signifie un pixel plus sombre.

Le paramètre `dtype=uint8`signifie que nos pixels sont représentés par des entiers de 8 bits. Chaque bit ne pouvant avoir une valeur que de 1 ou 0, le plus grand nombre pouvant être représenté par ce type d’entier est 255.


### Explications plus détaillées sur la représentation des images.
Cette section c’est seulement si j’ai le temps
#### Pourquoi le *grayscale*?
##### Pourquoi est-il inversé?
#### Pourquoi seulement 784 pixels?
```{figure} ./img/toto.png
---
name: toto
---
this is just toto
```

[^1]:	Un ordinateur moderne possédant un processeur de 2GHz peut réaliser 2 000 000 000 opérations par seconde sur chacun de ses coeurs.

[^2]:	Exécuter plusieurs opérations parallèlement plutôt que séquentiellement.

[^3]:	Un «coeur» est une unité du processeur pouvant faire du calcul indépendamment des autres coeurs. Un processeur d’ordinateur portable moderne possède de deux à quatre coeurs. Une carte graphique moderne en possède quelques milliers, quoique moins performants que ceux du processeur.

[^4]:	Pour réaliser une multiplication entre deux matrices, il faut que le nombre de colonnes de la première matrice soit égal au nombre de rangées de la deuxième.

[^5]:	Une matrice carrée est une matrice qui contient autant de rangées que de colonnes.

[^6]:	$n$ représente le nombre de rangées et de colonnes d’une matrice carrée.

[^7]:	Ils peuvent être calculés dans n’importe quel ordre et le résultat sera toujours le même. Un résultat pourrait aussi

[^8]:	Plus ou moins. Voir [1.1 Amdahl's law and Gustafson's law](https://en.wikipedia.org/wiki/Parallel_computing#Amdahl's_law_and_Gustafson's_law "1.1 Amdahl's law and Gustafson's law")

[^9]:	Pour que deux matrices puissent être additionnées, elles doivent avoir le même nombre de rangées et de colonnes.

[^10]:	En assumant que le calcul est réalisé sur une machine possédant plusieurs coeurs.

[^11]:	Dans le cas des autres opérations matricielles présentées, elles étaient seulement associatives. Les calculs étaient, comme pour la sommation, indépendants les un des autres. Par contre, pour les autres opérations les résultats devaient être placés de manière ordonnée dans la matrice résultante.

[^12]:	Voir la {section} sur les résultats concrets.

[^13]:	`BLAS` signifie **B**asic **L**inear **A**lgebra **S**ubroutines, ou sous-routines de base d’algèbre linéaire.

[^14]:	Petite mémoire rapide allouée au processeur.

[^15]:	Testé sur un processeur *2.9 GHz Dual-Core Intel Core i5*.