# ML-intro

Introduction aux concepts de base de l'apprentissage machine à l'aide de Python.

## Comment travailler avec LaTeX

Cette section vise à accélérer la transition de Word vers LaTeX.

### Informations générales

Pour l'instant, je pense que la meilleure source d'information sur LaTeX sont
[Overleaf](https://www.overleaf.com/learn) ainsi que le 
[StackExchange](https://tex.stackexchange.com/) dédié au langage.

### Rapport préliminaire

Pour le rapport préliminaire, les fichiers devraient tous être situés dans le
répertoire `Rapport_preliminaire`.

#### Représentation de l'arbre du répertoire:

```
.
├── Bibliography
│   └── bibliography.bib
├── Body
│   ├── description.tex
│   ├── enonce_sujet.tex
│   └── plan.tex
├── Page_presentation
│   └── pp_pre.tex
└── rapport_preleminaire.tex
```

##### Bibliography

Cette section ne devrait contenir que des fichiers `.bib`. Ce sont des fichiers
qui pourront facilement être lus par le [programme](http://www.bibtex.org/) 
générant la bibliographie. La syntaxe dans le fichier devrait être similaire à
l'exemple ci-dessous:

``` 
@book{[bib_id], 
author = "[FirstName] [Lastname]",
title = "[tile]",
publisher = "[maison de publication]",
year = "[année]"
}
```

Les crochets signifient que la valeur doit être remplacée par la vôtre (enlever 
les crochets).

Ce qui remplace `bib_id` est arbitraire. Seulement choisir un mot sans espace.
Il serait préférable que ce soit le nom de famille de l'auteur suivi de la
date de publication.

Pour plus d'information sur [BibTeX](http://www.bibtex.org/), voir la
documentation sur le site officiel ou celui l'Overleaf.
(https://www.overleaf.com/learn/latex/Bibliography_management_with_bibtex)

Pour référencer l'auteur dans vos textes, il suffit d'insérer `\cite{bib_id}`.
La référence sera insérée automatiquement lorsque le texte sera compilé.

##### Body

Ce répertoire contient les fichiers de chacune des sections qui sont à remplir.
Remplir chaque fichier individuellement. Ces fichiers sont ensuite inclus dans
le fichier principal. Lorsque celui-ci sera compilé, le contenu de ce répertoire
sera automatiquement ajouté au fichier principal.

##### Page_presentation

Contiens le fichier de la page de présentation.

##### rapport_preliminaire.tex

C'est le fichier principal qui sera compilé à la fin du travail. Plus rien ne
devrait avoir à être ajouté à ce fichier.

