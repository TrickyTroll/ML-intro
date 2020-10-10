# ML-intro

Introduction aux concepts de base de l'apprentissage machine à l'aide de Python.

## Utilisation de Jupyter Book

Pour la réalisation de notre projet, nous avons décidé d'utiliser 
[Jupyter Book](https://jupyterbook.org/intro.html). Cet outil nous permettra
de n'avoir qu'une seule source de fichiers afin de réaliser un site internet
sur notre projet ainsi qu'un fichier PDF pour les remises. Jupyter Book permet
aussi l'utilisation du langage de markup 
[MyST](https://myst-parser.readthedocs.io/en/latest/), ce qui laisse beaucoup
de flexibilité lors de la stylisation de notre texte. L'intégration entre les
fichiers Markdown et les notebooks Jupyter se fait aussi sans friction, ce qui
nous permet de présenter des exemples concrets en Python pour supporter nos
propos.

### Respect des normes de présentation du cégep

Par défaut, Jupyter Book est configuré en anglais et le style de citation ne
convient pas aux normes du Cégep de Sainte-Foy. Afin de respecter ces normes,
certaines modifications ont dû être faites à la configuration.

#### Jupyter Book en français

Jupyter Book utilise [Sphinx](https://www.sphinx-doc.org/en/master/) pour
générer le layout du site. Il n'est donc pas particulièrement compliqué de
changer le langage. Dans le fichier `_config.yml`, il faut ajouter `langage: fr`
sous les paramètres de configuration de Sphinx comme suit.

```yaml
sphinx:
  config:
    language: fr
```

#### Citations style APA

Pour l'instant, 
[très peu](https://jupyterbook.org/content/citations.html#selecting-your-reference-style)
de styles de référence et citation sont disponibles dans Jupyter Book. 
Heureusement, l'outil se sert de 
[sphinxcontrib-bibtex](https://jupyterbook.org/content/citations.html#selecting-your-reference-style)
pour générer les références. Les extensions faites pour ce programme sont donc 
compatibles avec Jupyter Book.

Pour utiliser le style de citations APA, nous avons décidé d'utiliser
[pybtex-apa-style](https://github.com/Naeka/pybtex-apa-style). L'extension est
disponible comme un package sur [Pip](https://pypi.org/project/pybtex-apa-style/).

Afin d'utiliser cette extension, il vous faut:

1. L'installer avec `pip` dans le même environnement que Jupyter Book.
2. Ajouter la mention `:style: apa` sous l'appel à votre bibliographie.

#### Conclusion

En seulement quelques minutes, il est possible d'avoir un nnement permettant
de générer des rapports scientifiques reproductibles, de qualité et suivant les
normes du Cégep de Sainte-Foy.

Pour l'instant, la page de présentation générée automatiquement par Jupyter Book
ne suit pas les conventions du Cégep. Pour l'instant, nous générons la page
de présentation à part et joignons les deux fichiers PDF à l'aide du programme
[pdfjam](https://github.com/DavidFirth/pdfjam), mais il serait beaucoup plus
efficace d'ajuster le générateur de page de présentation.

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

