"""Registre des matières, dans l'ordre du parcours."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import derivees, equations, explog, fonctions, gradient, matrices, nn, optim, probas, stats, trigo, vecteurs


@dataclass
class Subject:
    name: str
    levels: dict | None = None            # {niveau: [générateurs]} ; None = pas encore disponible
    descriptions: dict = field(default_factory=dict)


def _s(name, module=None):
    if module is None:
        return Subject(name)
    return Subject(name, module.LEVELS, module.DESCRIPTIONS)


SUBJECTS = [
    _s("Équations", equations),
    _s("Fonctions & graphes", fonctions),
    _s("Trigonométrie", trigo),
    _s("Exponentielle & logarithme", explog),
    _s("Dérivées", derivees),
    _s("Règle de la chaîne & gradient", gradient),
    _s("Vecteurs", vecteurs),
    _s("Matrices", matrices),
    _s("Probabilités", probas),
    _s("Statistiques", stats),
    _s("Optimisation", optim),
    _s("Maths des réseaux de neurones", nn),
]
