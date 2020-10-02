# Énoncé du sujet

## Mise en contexte

L'intelligence artificielle est au coeur de l'actualité depuis près d'une
décénnie. Elle est déjà entrain de changer le monde , et ce, dans plusieurs 
secteurs incluant la finance, la sécurité, la santé, la justice criminelle, 
les moyens de transport, la publicité, et plusieurs autres. 

Que ça soit des décisions sur l'investissement d'un portefeuille
d'un individu ou de la détection de fraude en identifiant des anormalités, l'intelligence
artificielle est de plus en plus présente dans le secteur de la finance. 

Du côté de la 
sécurité, un excellent exemple serait [Project Maven](https://en.wikipedia.org/wiki/Project_Maven)
un projet d' intelligence artificielle du [Pentagon](https://en.wikipedia.org/wiki/The_Pentagon) 
des États-Unis qui est capable de passer à travers plusieurs informations, 
vidéos et photos pour détecter des dangers potentiels.

L'intelligence artificielle est très importante dans la santé avec des compagnies comme 
[Merantix](https://www.merantix.com/), une compagnie allemande qui a permis de detecter 
des ganglions lymphatiques ainsi que des problèmes liés à ceux-ci tels que des lésions 
ou des cancers. L'étude de séquence d'ADN par l'intelligence artificielle permet de détecter
des maladies génétiques et des cancers.

Un des domaines le plus importants en ce moment serait, les moyens de transport avec plus de $80
milliards investis dans des véhicules de conduite autonome entre 2014 et 2017. L'intelligence 
artificielle dans ce domaine aurait pour but de diminuer grandement l'erreur humaine dans les transports
et réduire à presque zéro les accidents si la majorité des autos était intelligente. De plus, cela réduirait
aussi grandement le trafic grâce à la communication entre les automobiles intelligentes. La compagnie [Tesla](https://www.tesla.com/)
en est déjà très avancée pour ce qui est de leur auto intelligente.


Comme on peut le voir, cette technologie a permis de multiples avancées dans des domaines où 
il se fait extrêmement difficile de modéliser la problématique selon une
fonction mathématique particulière. L'analyse de langage en est un bon exemple.
Le travail ne peut être modélisé par une seule fonction mathématique puisque
les conditions souvent changeantes nécessiteraient une multitude de fonctions
différentes pour chaque environnement qui n'est pas réaliste. La solution est
plutôt « d'entraîner » un ordinateur à comprendre le monde qui l'entoure.
Pour continuer avec l'exemple de l'analyse du langage, une solution serait
de fournir à l'ordinateur une immense quantité d'exemples et de solutions afin
qu'il développe la capacité de prédire la solution à de nouveaux exemples.
[GPT-3](https://github.com/openai/gpt-3), 
un nouveau modèle d'intelligence artificielle produit par 
[OpenAI](https://openai.com), a permis à des développeurs de créer un programme
lui-même capable de programmer à partir de demandes spécifiques faites par un
utilisateur.

## Le début de la découverte des inconvénients

Malgré les avancées incroyables que l'intelligence artificielle a déjà permis et
continuera de permettre dans le futur, elle n'est pas sans ses inconvénients. 

## Le biais
Au
courant des dernières années, les systèmes intelligents sont de plus en plus
reconnus coupables de discrimination envers certains groupes d'individus. Une
étude réalisée par le [NIST](https://www.nist.gov/) à étudié le taux d'erreur de
différents programmes de reconnaissance faciale en fonction des différences de
sexe et d'ethnicité des individus sur les photos analysées. L'étude 
présente des taux d'erreur
jusqu'à cent fois plus élevés pour des personnes d'origine asiatique ou 
africaine lorsque comparé à des personnes d'origine européenne {cite}`nistbias`.
Le taux d'erreur est aussi plus élevé chez les femmes que chez les hommes, et
ce, peut importe l'origine.

Un autre résultat important de cette étude est que le taux d'erreur associé à la
reconnaissance de personnes asiatiques n'est pas présent dans des programmes
réalisés dans des pays d'Asie. Cette observation permet de déduire l'un des plus
grands problèmes liés à l'intelligence artificielle: le biais.

Contrairement à une fonction mathématique qui transforme un chiffre de manière
définie, les procédés menant à la reconnaissance faciale sont beaucoup plus
flous et souvent très mal compris. Plusieurs considèrent les programmes
entraînés comme des « boîtes noires ». Il est difficile de prédire ce qui sortira
de la boîte lorsque l'on y insère quelque chose, et il est encore plus difficile
de comprendre pourquoi le programme prend certaines décisions plus que d'autres.

Cette imprévisibilité inquiète plusieurs. Elle rend la tâche de corriger le
biais assez ardue. Elle fait aussi en sorte qu'il est difficile de prédire le
comportement du programme dans des cas extrêmes sans avoir à lui faire passer
des tests dans ces conditions. Le biais est donc un phénomène difficile à
corriger, ce qui entraîne des questionnements en rapport aux bienfaits de
l'utilisation de l'intelligence artificielle. 

Certaines régions du monde
commencent à banir l'utilisation de la reconnaissance faciale par les forces
de l'ordre. C'est le cas de la ville de Portland, en Oregon {cite}`cnnportland`.
La ville a décidé de bannir l'utilisation de la technologie suite à des craintes
en liées à son manque de précision, surtout lorsqu'utilisée sur des individus
appartenant à une minorité visible.

```{figure} ./img/black_box.png
---
name: boite-noire
---
L'analogie de la boîte noire.
```

## Une deuxième révolution industrielle

Une autre inquiétude liée à l'intelligence artificielle est l'importante 
quantité d'emplois qui risque de disparaître puisqu'ils seront maintenant
occupés par des ordinateurs. Ces inquiétudes sont justifiées. Plusieurs articles,
dont 
[celui-ci](https://www.cnbc.com/2019/01/14/the-oracle-of-ai-these-kinds-of-jobs-will-not-be-replaced-by-robots-.html)
publié par CNN ainsi que 
[cette publication](https://medium.com/@ChanPriya/15-jobs-that-will-never-be-replaced-by-ai-512bfbbed0d6)
sur Medium tentent de rassurer la population en mentionnant des emplois qui ne
pourraient apparemment jamais être remplacés par des ordinateurs. Ils mentionnent
entre autres les emplois créatifs, accompagnés des emplois nécessitant beaucoup
d'interactions humaines.

Pourtant, le domaine de l'IA avance chaque année, et il existe maintenant une
panoplie de programmes capable de
[composer de la musique](https://openai.com/blog/musenet/),
[maîtriser les arts visuels](https://www.nvidia.com/en-us/research/ai-playground/)
ainsi qu'[entretenir des conversations au téléphone](https://www.youtube.com/watch?v=D5VN56jQMWM).

```{figure} ./img/duplex.jpeg
---
name: duplex-presentation
---
Le PDG de Google présentant une démonstration de Google Duplex.
```

Il est dangereux d'extrapoler le progrès qui a été fait au courant des dernières 
années sur les décénnies à venir. Certaines lois limitant le développement de 
l'IA, ou des limitations physiques au présent rythme d'augmentation de la 
puissance de calcul des ordinateurs pourraient survenir grandement ralentir
le développement de la technologie. Si nous tentons tout de même de le faire,
les inquiétudes vécues par plusieurs semblent raisonnables.

## Comprendre la technologie pour démystifier les inquiétudes

Bien que les précédentes inquiétudes face à l'intelligence artificielle soient
totalement justifiées, elles ne sont pas sans solution. Si son développement
est fait de manière éthique et s'il est bien encadré, nous pourrions en retirer
plus d'avantages que d'inconvénient. Pour bien comprendre les inquiétudes, il
faut d'abord comprendre les enjeux. C'est pourquoi nous tenterons de répondre
à la question suivante.

*Quel est le fonctionnent de l'intelligence artificielle et comment devrait-elle
être utilisée afin de bénéficier l’être humain?*

