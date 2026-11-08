# Math Trainer

Apprendre les maths **par la pratique**, des équations du premier degré jusqu'aux maths utilisées en IA.
Chaque exercice s'affiche dans une fenêtre matplotlib avec sa représentation géométrique (courbes, triangle,
cercle trigonométrique, vecteurs…) ; on répond dans le terminal.

## Prérequis

- **[uv](https://docs.astral.sh/uv/)** (gestionnaire de projets Python). Installation :
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # ou, sur macOS avec Homebrew
  brew install uv
  ```
- **Python ≥ 3.10**. Rien à installer à la main : uv télécharge automatiquement la version indiquée dans
  `.python-version` si elle est absente.
- **git**, pour récupérer le projet.
- Un **environnement graphique** pour les fenêtres matplotlib. Le Python fourni par uv inclut Tkinter. Si tu
  utilises un Python système sous Linux et que la fenêtre ne s'ouvre pas, installe Tk
  (`sudo apt install python3-tk` sur Debian/Ubuntu).

Les dépendances (matplotlib, numpy, sympy) sont installées automatiquement par uv.

## Installation

```bash
git clone https://github.com/My-sidequests/math_trainer.git
cd math_trainer
uv sync
```

## Utilisation

```bash
uv run math-trainer      # lancer l'entraîneur
uv run pytest            # lancer les tests (vérifie les 600 modèles d'exercices)
```

Pour quitter à tout moment : `q` dans les menus, ou `Ctrl+C`.

## Déroulement

1. Choisis une **matière**.
2. Choisis un **niveau** de 1 à 5.
3. Un exercice tiré au hasard parmi les 10 modèles du niveau s'affiche (les valeurs changent à chaque fois).
4. Tape ta réponse. Tentatives illimitées.
   - `m` : affiche la **méthode**, puis une **démonstration complète sur les nombres de l'exercice** (étapes
     de calcul, vérification et réponse). Les valeurs étant tirées au hasard, l'exercice suivant reste à faire.
   - `q` : abandonne l'exercice
5. Si la réponse est juste : **SUCCESS**, puis `[a]` autre exercice du même niveau, `[m]` menu principal, `[q]` quitter.

## Formats de réponse

| Type | Exemples |
|---|---|
| Nombre | `3`, `-2.5`, `7/3`, `sqrt(2)/2`, `√3/2`, `pi/4`, `π/4`, `2^10`, `ln(2)`, `e^2`, `inf` |
| Plusieurs nombres | `2, -3` (ordre libre pour des solutions, imposé pour des coordonnées `x, y`) |
| Aucune solution | `aucune` |
| Expression | `2*x + 3`, `3x^2 - 1`, `e^(2x)`, `4*x*y + 6` (en x, ou en x et y pour les dérivées partielles ; comparée mathématiquement, pas caractère par caractère) |
| Vecteur / matrice | `2, -4` ; une matrice s'écrit ligne par ligne : `1, 2, 3, 4` pour [[1, 2], [3, 4]] |
| Mot | `oui` / `non`, `croissante` / `décroissante`… |

Les décimaux s'écrivent avec un **point** (la virgule sépare plusieurs valeurs).
Quand une calculatrice est autorisée, l'énoncé le précise (3 décimales suffisent) ; sinon, une réponse exacte est
attendue (une valeur décimale à 3 chiffres significatifs est aussi acceptée).

## Parcours

12 matières × 5 niveaux × 10 modèles d'exercices = 600 modèles, des bases jusqu'aux maths du deep learning.

| # | Matière | Du niveau 1 … | … au niveau 5 |
|---|---|---|---|
| 1 | Équations | x + a = b | systèmes, équations rationnelles |
| 2 | Fonctions & graphes | calculer f(k) | limites, asymptotes |
| 3 | Trigonométrie | Pythagore, SOH-CAH-TOA | Al-Kashi, polaires, angle entre vecteurs |
| 4 | Exponentielle & logarithme | puissances | sigmoïde, softmax, entropie croisée |
| 5 | Dérivées | dérivée d'un polynôme | dérivée seconde, Newton, convexité |
| 6 | Règle de la chaîne & gradient | (ax + b)² | Hessienne, Jacobienne, points critiques |
| 7 | Vecteurs | addition, norme | similarité cosinus, embeddings |
| 8 | Matrices | addition, transposée | couches denses, gradients de W, ACP |
| 9 | Probabilités | dés, urnes | entropie, KL, vraisemblance |
| 10 | Statistiques | moyenne, médiane | intervalles de confiance, batch norm |
| 11 | Optimisation | un pas de descente | Adam, conditionnement, régression logistique |
| 12 | Maths des réseaux de neurones | un neurone | backprop complète, convolution, attention |

## Structure

```
src/math_trainer/
├── main.py           # menus, fenêtre matplotlib, boucle d'exercices
├── answers.py        # types de réponses (Number, Numbers, Expr, Choice) et lecture de la saisie (sympy)
├── core.py           # Exercise, formatage, aides au dessin (courbes, vecteurs, matrices, réseaux...)
└── subjects/
    ├── __init__.py   # registre des matières (ordre du parcours)
    ├── equations.py   # 1. Équations
    ├── fonctions.py   # 2. Fonctions & graphes
    ├── trigo.py       # 3. Trigonométrie
    ├── explog.py      # 4. Exponentielle & logarithme
    ├── derivees.py    # 5. Dérivées
    ├── gradient.py    # 6. Règle de la chaîne & gradient
    ├── vecteurs.py    # 7. Vecteurs
    ├── matrices.py    # 8. Matrices
    ├── probas.py      # 9. Probabilités
    ├── stats.py       # 10. Statistiques
    ├── optim.py       # 11. Optimisation
    └── nn.py          # 12. Maths des réseaux de neurones
tests/test_generators.py
```

## Ajouter une matière

1. Crée `subjects/<matiere>.py` avec `DESCRIPTIONS = {1: "...", ..., 5: "..."}` et
   `LEVELS = {1: [gen, ...], ..., 5: [...]}` (10 générateurs par niveau).
2. Un générateur prend un `random.Random` et renvoie un `Exercise(statement, answer, method, draw, solution)`.
   `solution` est la démonstration pas à pas (les helpers `lines()` et `steps_lin()` de `core.py` aident à
   la rédiger) ; la ligne « Réponse : … » est ajoutée automatiquement à partir de `answer`.
3. Branche le module dans `subjects/__init__.py`.
4. Lance `uv run pytest` : le test vérifie, pour chaque générateur et plusieurs tirages, que la bonne réponse
   est acceptée, qu'une mauvaise est refusée, que la figure se dessine et que la démonstration existe.

## Notes

- Sous Linux, matplotlib a besoin d'un backend graphique (Tk est inclus avec le Python installé par uv).
- La fenêtre de la figure peut ne pas réagir (zoom, déplacement) pendant que le terminal attend ta réponse : c'est normal.
