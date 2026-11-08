"""Vérifie que chaque générateur produit un exercice cohérent : la bonne réponse est acceptée,
une mauvaise est refusée, et le dessin s'exécute sans erreur."""

import math
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from math_trainer.answers import Choice, Expr, Number, Numbers  # noqa: E402
from math_trainer.subjects import SUBJECTS  # noqa: E402


def good_and_bad(ans):
    if isinstance(ans, Number):
        v = ans.value
        if math.isinf(v):
            return ("inf" if v > 0 else "-inf"), "0"
        return repr(v), repr(v + 1.5 + 0.05 * abs(v))
    if isinstance(ans, Numbers):
        if not ans.values:
            return "aucune", "0"
        return ", ".join(repr(v) for v in ans.values), ", ".join(repr(v + 1.5 + 0.05 * abs(v)) for v in ans.values)
    if isinstance(ans, Expr):
        return str(ans.expr).replace("**", "^"), str(ans.expr) + " + 1"
    if isinstance(ans, Choice):
        return ans.correct, "zzz"
    raise TypeError(ans)


def test_all_generators():
    n = 0
    for subject in SUBJECTS:
        if not subject.levels:
            continue
        assert sorted(subject.levels) == [1, 2, 3, 4, 5], subject.name
        for level, gens in subject.levels.items():
            assert len(gens) == 10, (subject.name, level, len(gens))
            for gen in gens:
                for seed in range(25):
                    ex = gen(random.Random(seed))
                    good, bad = good_and_bad(ex.answer)
                    where = f"{subject.name} L{level} {gen.__name__} seed={seed}: {ex.statement!r} / {good}"
                    assert ex.answer.check(good), where
                    assert not ex.answer.check(bad), where + " (mauvaise réponse acceptée)"
                    assert ex.method
                    if ex.draw and seed < 3:
                        fig, ax = plt.subplots()
                        ex.draw(ax)
                        plt.close(fig)
                    n += 1
    assert n > 0


def test_parsing():
    assert Number(math.sqrt(2) / 2).check("√2/2")
    assert Number(math.sqrt(2) / 2).check("sqrt(2)/2")
    assert Number(math.pi / 3).check("π/3")
    assert Number(math.pi / 3).check("pi/3")
    assert Number(7 / 3).check("7/3")
    assert Number(7 / 3).check("2.3333")
    assert not Number(7 / 3).check("2.3")
    assert Numbers([2, -3]).check("-3, 2")
    assert not Numbers([2, -3], ordered=True).check("-3, 2")
    assert Expr("2*x + 3").check("3 + 2x")
    assert Expr("x**2").check("x^2")
    assert Choice("decroissante", ["croissante", "décroissante"]).check("Décroissante")
