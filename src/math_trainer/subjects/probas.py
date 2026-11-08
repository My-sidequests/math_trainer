"""Probabilités : dénombrement, conditionnelles, Bayes, lois discrètes et continues, théorie de l'information."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Number
from ..core import Exercise, bars, curves

from ..core import fmt, lines, par  # noqa: E402,F811

DESCRIPTIONS = {
    1: "Probabilités simples : dés, pièces, urnes, complémentaire",
    2: "Événements combinés, arbres, dénombrement (n!, combinaisons)",
    3: "Conditionnelles, Bayes, loi binomiale, espérance et variance",
    4: "Lois continues : uniforme, exponentielle, normale ; Poisson",
    5: "Vers l'IA : entropie, entropie croisée, KL, vraisemblance",
}

M_BASIC = ("Quand toutes les issues sont équiprobables : P(A) = (nombre d'issues favorables) / (nombre d'issues possibles).\n"
           "Complémentaire : P(non A) = 1 − P(A). Réponds en fraction (ex : 1/6) ou en décimal.")
M_COUNT = ("n! = n × (n−1) × ... × 1 : nombre d'ordres possibles de n objets.\n"
           "Combinaisons C(n, k) = n! / (k!(n−k)!) : nombre de façons de CHOISIR k objets parmi n (sans ordre).\n"
           "Arrangements A(n, k) = n! / (n−k)! : choisir k objets parmi n AVEC ordre.")
M_COND = ("P(A | B) = P(A ∩ B) / P(B) : probabilité de A sachant que B est arrivé.\n"
          "Bayes : P(A | B) = P(B | A)·P(A) / P(B), avec P(B) = P(B | A)P(A) + P(B | non A)P(non A).")
M_BINOM = ("Loi binomiale B(n, p) : nombre de succès sur n essais indépendants de probabilité p.\n"
           "P(X = k) = C(n, k)·pᵏ·(1 − p)ⁿ⁻ᵏ ;  E[X] = np ;  Var(X) = np(1 − p).")
M_INFO = ("Entropie H(p) = −Σ pᵢ·log₂(pᵢ) (en bits) : l'incertitude moyenne.\n"
          "Entropie croisée H(p, q) = −Σ pᵢ·log(qᵢ) ; KL(p‖q) = Σ pᵢ·log(pᵢ/qᵢ) = H(p, q) − H(p).\n"
          "(log₂ → bits, ln → nats ; l'énoncé précise lequel.)")
CALC = "\n(calculatrice autorisée ; 3 décimales)"


def binom_pmf(n, k, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def dist_bars(probs, labels=None, title=""):
    return bars(labels or [str(i + 1) for i in range(len(probs))], [float(p) for p in probs], title, ylim=(0, 1))


# ================================================================ NIVEAU 1

def l1_die(r):
    k = r.randint(1, 5)
    kind = r.choice(["≤", "≥", "pair", "impair"])
    if kind == "≤":
        fav, txt = k, f"un nombre ≤ {k}"
    elif kind == "≥":
        fav, txt = 7 - k, f"un nombre ≥ {k}"
    else:
        fav, txt = 3, f"un nombre {kind}"
    _list = {"≤": list(range(1, k + 1)), "≥": list(range(k, 7)), "pair": [2, 4, 6], "impair": [1, 3, 5]}[kind]
    _sol = lines(f"Issues favorables : {_list} → {fav} issue(s) sur 6", f"P = {fav}/6 = {fmt(F(fav, 6))}")
    ex = Exercise(f"On lance un dé équilibré à 6 faces. Probabilité d'obtenir {txt} ?", Number(F(fav, 6)), M_BASIC,
                    dist_bars([F(1, 6)] * 6, title="loi du dé"))
    ex.solution = _sol
    return ex


def l1_coin(r):
    n = r.randint(2, 4)
    _sol = lines("Chaque lancer : P(pile) = 1/2, lancers indépendants → on multiplie.", f"P = (1/2)^{n} = 1/{2 ** n}")
    ex = Exercise(f"On lance {n} fois une pièce équilibrée. Probabilité d'obtenir {n} fois pile ?", Number(F(1, 2 ** n)),
                    "Les lancers sont indépendants : on multiplie les probabilités (½ à chaque fois).", None)
    ex.solution = _sol
    return ex


def l1_cards(r):
    q = r.choice([("un cœur", 13), ("un as", 4), ("une figure (valet, dame, roi)", 12), ("une carte rouge", 26),
                  ("l'as de pique", 1)])
    _sol = lines(f"Cartes favorables : {q[1]} sur 52", f"P = {q[1]}/52 = {fmt(F(q[1], 52))}")
    ex = Exercise(f"On tire une carte au hasard dans un jeu de 52 cartes. Probabilité de tirer {q[0]} ?",
                    Number(F(q[1], 52)), M_BASIC + "\nUn jeu : 4 couleurs × 13 valeurs.", None)
    ex.solution = _sol
    return ex


def l1_complement(r):
    p = r.choice([F(1, 3), F(2, 5), F(1, 10), F(3, 8), F(7, 20)])
    _sol = lines("P(non A) = 1 − P(A)", f"= 1 − {fmt(p)} = {fmt(1 - p)}")
    ex = Exercise(f"La probabilité qu'il pleuve demain est {p}. Probabilité qu'il ne pleuve pas ?", Number(1 - p), M_BASIC,
                    dist_bars([p, 1 - p], ["pluie", "pas de pluie"]))
    ex.solution = _sol
    return ex


def l1_urn(r):
    rd, bl, gr = r.randint(1, 8), r.randint(1, 8), r.randint(0, 6)
    _sol = lines(f"Total : {rd} + {bl} + {gr} = {rd + bl + gr} boules, dont {bl} bleues", f"P = {bl}/{rd + bl + gr} = {fmt(F(bl, rd + bl + gr))}")
    ex = Exercise(f"Une urne contient {rd} boules rouges, {bl} bleues et {gr} vertes. Probabilité de tirer une bleue ?",
                    Number(F(bl, rd + bl + gr)), M_BASIC,
                    bars(["rouge", "bleue", "verte"], [rd, bl, gr], "contenu de l'urne"))
    ex.solution = _sol
    return ex


def l1_disjoint(r):
    a, b = r.sample(range(1, 7), 2)
    _sol = lines("A et B sont incompatibles : P(A ou B) = P(A) + P(B)", "= 1/6 + 1/6 = 2/6 = 1/3")
    ex = Exercise(f"Dé à 6 faces. A = « obtenir {a} », B = « obtenir {b} ». Calcule P(A ou B).", Number(F(2, 6)),
                    "A et B ne peuvent pas arriver ensemble (incompatibles) : P(A ou B) = P(A) + P(B).",
                    dist_bars([F(1, 6)] * 6, title="loi du dé"))
    ex.solution = _sol
    return ex


def l1_two_dice(r):
    s = r.randint(2, 12)
    fav = 6 - abs(s - 7)
    _pairs = [(i, s - i) for i in range(1, 7) if 1 <= s - i <= 6]
    _sol = lines(f"Couples (dé1, dé2) de somme {s} : {', '.join(f'({a},{b})' for a, b in _pairs)} → {fav} couples",
                 f"P = {fav}/36 = {fmt(F(fav, 36))}")
    ex = Exercise(f"On lance deux dés. Probabilité que la somme fasse {s} ?", Number(F(fav, 36)),
                    "Il y a 6 × 6 = 36 couples équiprobables (dé1, dé2). Compte ceux dont la somme vaut " + str(s) + ".",
                    bars(list(range(2, 13)), [6 - abs(t - 7) for t in range(2, 13)], "nombre de couples par somme"))
    ex.solution = _sol
    return ex


def l1_at_least_one(r):
    n = r.randint(2, 4)
    _sol = lines(f"Contraire : « jamais face » = {n} fois pile, P = (1/2)^{n} = 1/{2 ** n}", f"P(au moins une face) = 1 − 1/{2 ** n} = {fmt(1 - F(1, 2 ** n))}")
    ex = Exercise(f"On lance {n} fois une pièce. Probabilité d'obtenir AU MOINS une fois face ?", Number(1 - F(1, 2 ** n)),
                    "« Au moins une fois » : passe par le complémentaire « jamais » (= toujours pile), puis 1 − P(jamais).",
                    None)
    ex.solution = _sol
    return ex


def l1_expect_die(r):
    n = r.choice([4, 6, 8, 10, 12, 20])
    _sol = lines(f"E = (1 + 2 + ... + {n}) / {n}", f"1 + ... + {n} = {n} × {n + 1} / 2 = {n * (n + 1) // 2}", f"E = {n * (n + 1) // 2}/{n} = {fmt(F(n + 1, 2))}")
    ex = Exercise(f"Espérance (valeur moyenne) d'un dé équilibré à {n} faces numérotées de 1 à {n} ?", Number(F(n + 1, 2)),
                    "E[X] = Σ valeur × probabilité. Ici chaque face a probabilité 1/" + str(n) + ".",
                    dist_bars([F(1, n)] * n))
    ex.solution = _sol
    return ex


def l1_percent(r):
    p = r.choice([5, 12, 25, 40, 75, 2.5])
    _sol = lines(f"{fmt(p)} % = {fmt(p)}/100 = {fmt(p / 100)}")
    ex = Exercise(f"Un événement a {p} % de chances de se produire. Écris cette probabilité sous forme décimale.",
                    Number(p / 100), "Pourcentage → probabilité : divise par 100.", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 2

def l2_indep(r):
    a, b = r.choice([F(1, 2), F(1, 3), F(2, 5), F(3, 4)]), r.choice([F(1, 2), F(1, 4), F(3, 5), F(1, 6)])
    _sol = lines("Indépendants : P(A et B) = P(A) × P(B)", f"= {fmt(a)} × {fmt(b)} = {fmt(a * b)}")
    ex = Exercise(f"A et B sont indépendants, P(A) = {a}, P(B) = {b}. Calcule P(A et B).", Number(a * b),
                    "Indépendants ⇔ P(A ∩ B) = P(A)·P(B).", None)
    ex.solution = _sol
    return ex


def l2_union(r):
    a, b = F(r.randint(2, 6), 10), F(r.randint(2, 6), 10)
    ab = F(r.randint(1, min(int(a * 10), int(b * 10))), 10)
    _sol = lines("P(A ou B) = P(A) + P(B) − P(A et B)", f"= {float(a):g} + {float(b):g} − {float(ab):g} = {float(a + b - ab):g}")
    ex = Exercise(f"P(A) = {float(a)}, P(B) = {float(b)}, P(A et B) = {float(ab)}. Calcule P(A ou B).", Number(a + b - ab),
                    "P(A ∪ B) = P(A) + P(B) − P(A ∩ B) (on retire l'intersection comptée deux fois).", None)
    ex.solution = _sol
    return ex


def l2_cond_table(r):
    a, b, c, d = (r.randint(5, 40) for _ in range(4))
    _sol = lines(f"Fumeurs : {a} + {b} = {a + b}, dont {a} malades", f"P(malade | fumeur) = {a}/{a + b} ≈ {a / (a + b):.4f}")
    ex = Exercise(f"Sur un groupe : {a} fumeurs malades, {b} fumeurs non malades, {c} non-fumeurs malades, "
                    f"{d} non-fumeurs non malades. Probabilité d'être malade SACHANT qu'on est fumeur ?",
                    Number(F(a, a + b)), M_COND + "\nSachant « fumeur » : on ne regarde que la ligne des fumeurs.",
                    bars(["fum. malade", "fum. sain", "non-fum. malade", "non-fum. sain"], [a, b, c, d]))
    ex.solution = _sol
    return ex


def l2_tree_without(r):
    rd, bl = r.randint(2, 6), r.randint(2, 6)
    n = rd + bl
    _sol = lines(f"1re rouge : {rd}/{n}", f"2e rouge sachant la 1re rouge : il reste {rd - 1} rouges sur {n - 1} → {rd - 1}/{n - 1}",
                 f"P = {rd}/{n} × {rd - 1}/{n - 1} = {rd * (rd - 1)}/{n * (n - 1)} = {fmt(F(rd * (rd - 1), n * (n - 1)))}")
    ex = Exercise(f"Urne : {rd} rouges, {bl} bleues. On tire 2 boules SANS remise. Probabilité de 2 rouges ?",
                    Number(F(rd, n) * F(rd - 1, n - 1)),
                    "Arbre : P(1re rouge) × P(2e rouge sachant 1re rouge). Après le 1er tirage, il reste une boule de moins.",
                    None)
    ex.solution = _sol
    return ex


def l2_tree_with(r):
    rd, bl = r.randint(1, 6), r.randint(1, 6)
    n = rd + bl
    _sol = lines(f"P(rouge) = {rd}/{n}, P(bleue) = {bl}/{n} (avec remise : rien ne change entre les tirages)",
                 f"P = {rd}/{n} × {bl}/{n} = {rd * bl}/{n * n} = {fmt(F(rd * bl, n * n))}")
    ex = Exercise(f"Urne : {rd} rouges, {bl} bleues. On tire 2 boules AVEC remise. Probabilité d'une rouge puis une bleue ?",
                    Number(F(rd, n) * F(bl, n)), "Avec remise, les tirages sont indépendants : multiplie les probabilités.", None)
    ex.solution = _sol
    return ex


def l2_at_least(r):
    p, n = r.choice([0.1, 0.2, 0.05, 0.5]), r.randint(2, 6)
    _sol = lines(f"P(aucun succès) = (1 − {p})^{n} = {1 - p:g}^{n} ≈ {(1 - p) ** n:.4f}", f"P(au moins un) = 1 − {(1 - p) ** n:.4f} ≈ {1 - (1 - p) ** n:.4f}")
    ex = Exercise(f"Un essai réussit avec probabilité {p}. On fait {n} essais indépendants. Probabilité d'au moins un succès ?" + CALC,
                    Number(1 - (1 - p) ** n, 1e-3), "1 − P(aucun succès) = 1 − (1 − p)ⁿ.", None)
    ex.solution = _sol
    return ex


def l2_factorial(r):
    n = r.randint(3, 7)
    _sol = lines(f"{n} choix pour la 1re place, {n - 1} pour la 2e, etc.", f"{n}! = {' × '.join(str(i) for i in range(n, 0, -1))} = {math.factorial(n)}")
    ex = Exercise(f"De combien de façons peut-on ranger {n} livres différents sur une étagère ?", Number(math.factorial(n)),
                    M_COUNT, None)
    ex.solution = _sol
    return ex


def l2_comb(r):
    n = r.randint(4, 10)
    k = r.randint(2, n - 1)
    _sol = lines(f"C({n}, {k}) = {n}! / ({k}! × {n - k}!)", f"= {math.perm(n, k)} / {math.factorial(k)} = {math.comb(n, k)}")
    ex = Exercise(f"Combien de groupes de {k} personnes peut-on former parmi {n} ? (l'ordre ne compte pas)",
                    Number(math.comb(n, k)), M_COUNT, None)
    ex.solution = _sol
    return ex


def l2_arr(r):
    n = r.randint(4, 9)
    k = r.randint(2, 3)
    _sol = lines(f"{n} choix pour le 1er, {n - 1} pour le 2e" + (f", {n - 2} pour le 3e" if k == 3 else ""),
                 f"{' × '.join(str(n - i) for i in range(k))} = {math.perm(n, k)}")
    ex = Exercise(f"Course de {n} coureurs : combien de podiums différents (1er, 2e{', 3e' if k == 3 else ''}) possibles ?",
                    Number(math.perm(n, k)), M_COUNT + "\nIci l'ordre compte.", None)
    ex.solution = _sol
    return ex


def l2_lottery(r):
    n, k = r.choice([(10, 3), (8, 2), (12, 3), (20, 2)])
    _sol = lines(f"Nombre de combinaisons : C({n}, {k}) = {math.comb(n, k)}", f"Une seule gagnante : P = 1/{math.comb(n, k)}")
    ex = Exercise(f"Loterie : on choisit {k} numéros parmi {n}. Probabilité d'avoir la bonne combinaison ?",
                    Number(F(1, math.comb(n, k))), M_COUNT + "\nUne seule combinaison gagnante parmi C(n, k).", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

def l3_bayes(r):
    prev = r.choice([0.01, 0.02, 0.05, 0.1])
    sens, fp = r.choice([0.9, 0.95, 0.99]), r.choice([0.05, 0.1, 0.02])
    post = sens * prev / (sens * prev + fp * (1 - prev))
    _pb = sens * prev + fp * (1 - prev)
    _sol = lines(f"P(positif) = P(+|malade)P(malade) + P(+|sain)P(sain) = {sens} × {prev} + {fp} × {1 - prev:g} = {_pb:.5f}",
                 f"P(malade | +) = P(+|malade)·P(malade) / P(+) = {sens * prev:.5f} / {_pb:.5f} ≈ {post:.4f}",
                 "Même avec un bon test, si la maladie est rare, beaucoup de positifs sont des faux positifs.")
    ex = Exercise(f"Une maladie touche {prev * 100:g} % de la population. Un test détecte {sens * 100:g} % des malades, "
                    f"mais est positif chez {fp * 100:g} % des personnes saines. Si le test est positif, probabilité "
                    "d'être malade ?" + CALC, Number(post, 5e-3),
                    M_COND + "\nA = malade, B = test positif. Calcule d'abord P(positif) par la formule des probabilités totales.",
                    bars(["malades", "sains"], [prev, 1 - prev], "avant le test"))
    ex.solution = _sol
    return ex


def l3_total(r):
    pa = r.choice([0.3, 0.4, 0.6])
    pba, pbna = r.choice([0.2, 0.5, 0.9]), r.choice([0.1, 0.3, 0.6])
    _sol = lines("P(B) = P(B|A)P(A) + P(B|non A)P(non A)", f"= {pba} × {pa} + {pbna} × {1 - pa:g} = {pba * pa:g} + {pbna * (1 - pa):g} = {pba * pa + pbna * (1 - pa):g}")
    ex = Exercise(f"P(A) = {pa}, P(B | A) = {pba}, P(B | non A) = {pbna}. Calcule P(B).", Number(pba * pa + pbna * (1 - pa)),
                    "Probabilités totales : P(B) = P(B | A)·P(A) + P(B | non A)·P(non A).", None)
    ex.solution = _sol
    return ex


def l3_binom(r):
    n = r.randint(3, 6)
    k = r.randint(0, n)
    p = r.choice([0.5, 0.2, 0.3])
    _sol = lines(f"P(X = {k}) = C({n}, {k}) × {p}^{k} × {1 - p:g}^{n - k}", f"= {math.comb(n, k)} × {p ** k:.5g} × {(1 - p) ** (n - k):.5g}",
                 f"≈ {binom_pmf(n, k, p):.4f}")
    ex = Exercise(f"X suit B(n = {n}, p = {p}). Calcule P(X = {k})." + CALC, Number(binom_pmf(n, k, p), 5e-3), M_BINOM,
                    dist_bars([binom_pmf(n, i, p) for i in range(n + 1)], [str(i) for i in range(n + 1)], "loi de X"))
    ex.solution = _sol
    return ex


def l3_binom_mean(r):
    n, p = r.choice([10, 20, 50, 100]), r.choice([0.1, 0.25, 0.5, 0.3])
    _sol = lines("E[X] = n·p", f"= {n} × {p} = {n * p:g}")
    ex = Exercise(f"X suit B(n = {n}, p = {p}). Espérance E[X] ?", Number(n * p), M_BINOM, None)
    ex.solution = _sol
    return ex


def l3_binom_var(r):
    n, p = r.choice([10, 20, 50, 100]), r.choice([0.1, 0.25, 0.5, 0.2])
    _sol = lines("Var(X) = n·p·(1 − p)", f"= {n} × {p} × {1 - p:g} = {n * p * (1 - p):g}")
    ex = Exercise(f"X suit B(n = {n}, p = {p}). Variance Var(X) ?", Number(n * p * (1 - p)), M_BINOM, None)
    ex.solution = _sol
    return ex


def l3_geometric(r):
    p, k = r.choice([F(1, 2), F(1, 6), F(1, 3)]), r.randint(1, 4)
    _sol = lines(f"{k - 1} échec(s) puis 1 succès : ({fmt(1 - p)})^{k - 1} × {fmt(p)}", f"= {fmt((1 - p) ** (k - 1) * p)}")
    ex = Exercise(f"On répète un essai de probabilité de succès {p} jusqu'au premier succès. Probabilité que le premier "
                    f"succès arrive exactement au {k}e essai ?", Number((1 - p) ** (k - 1) * p),
                    "Il faut k − 1 échecs puis un succès : (1 − p)ᵏ⁻¹·p.", None)
    ex.solution = _sol
    return ex


def _table(r):
    vals = sorted(r.sample(range(-3, 8), 3))
    ws = [r.randint(1, 5) for _ in range(3)]
    s = sum(ws)
    return vals, [F(w, s) for w in ws]


def l3_expect(r):
    vals, ps = _table(r)
    _sol = lines("E[X] = Σ xᵢ·P(X = xᵢ)", " + ".join(f"{par(v)} × {fmt(p)}" for v, p in zip(vals, ps)), f"= {fmt(sum(v * p for v, p in zip(vals, ps)))}")
    ex = Exercise(f"X prend les valeurs {vals} avec probabilités {[str(p) for p in ps]}. Calcule E[X].".replace("'", ""),
                    Number(sum(v * p for v, p in zip(vals, ps))), "E[X] = Σ xᵢ·P(X = xᵢ).",
                    dist_bars(ps, [str(v) for v in vals]))
    ex.solution = _sol
    return ex


def l3_var(r):
    vals, ps = _table(r)
    m = sum(v * p for v, p in zip(vals, ps))
    _e2 = sum(v * v * p for v, p in zip(vals, ps))
    _sol = lines(f"E[X] = {fmt(m)}", "E[X²] = " + " + ".join(f"{v * v} × {fmt(p)}" for v, p in zip(vals, ps)) + f" = {fmt(_e2)}",
                 f"Var(X) = E[X²] − E[X]² = {fmt(_e2)} − ({fmt(m)})² = {fmt(_e2 - m * m)}")
    ex = Exercise(f"X prend les valeurs {vals} avec probabilités {[str(p) for p in ps]}. Calcule Var(X).".replace("'", ""),
                    Number(sum(p * (v - m) ** 2 for v, p in zip(vals, ps))),
                    "Var(X) = E[(X − E[X])²] = E[X²] − E[X]². Calcule E[X] puis E[X²].", dist_bars(ps, [str(v) for v in vals]))
    ex.solution = _sol
    return ex


def l3_cdf(r):
    n, k = r.randint(3, 5), r.randint(0, 2)
    _sol = lines(*[f"P(X = {i}) = C({n}, {i}) / 2^{n} = {math.comb(n, i)}/{2 ** n}" for i in range(k + 1)],
                 f"P(X ≤ {k}) = {sum(math.comb(n, i) for i in range(k + 1))}/{2 ** n} ≈ {sum(binom_pmf(n, i, 0.5) for i in range(k + 1)):.4f}")
    ex = Exercise(f"X suit B(n = {n}, p = 0.5). Calcule P(X ≤ {k})." + CALC,
                    Number(sum(binom_pmf(n, i, 0.5) for i in range(k + 1)), 5e-3),
                    M_BINOM + "\nP(X ≤ k) = P(X = 0) + ... + P(X = k).",
                    dist_bars([binom_pmf(n, i, 0.5) for i in range(n + 1)], [str(i) for i in range(n + 1)]))
    ex.solution = _sol
    return ex


def l3_indep_check(r):
    pa, pb = r.choice([0.5, 0.4, 0.2]), r.choice([0.5, 0.3, 0.6])
    pab = pa * pb if r.random() < 0.5 else round(pa * pb + r.choice([0.05, -0.04]), 2)
    _sol = lines(f"P(A) × P(B) = {pa} × {pb} = {pa * pb:g}", f"P(A ∩ B) = {round(pab, 4):g} : " + ("égal → indépendants." if abs(pab - pa * pb) < 1e-9 else "différent → pas indépendants."))
    ex = Exercise(f"P(A) = {pa}, P(B) = {pb}, P(A ∩ B) = {round(pab, 4)}. A et B sont-ils indépendants ? (oui / non)",
                    Choice("oui" if abs(pab - pa * pb) < 1e-9 else "non", ["oui", "non"]),
                    "Indépendants ⇔ P(A ∩ B) = P(A)·P(B). Calcule le produit et compare.", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 4

def l4_uniform(r):
    a, b = 0, r.choice([10, 20, 4, 60])
    c, d = sorted(r.sample(range(a, b + 1), 2))
    _sol = lines(f"Loi uniforme sur [{a}, {b}] : P = longueur de l'intervalle / longueur totale", f"= ({d} − {c}) / ({b} − {a}) = {d - c}/{b - a} = {fmt(F(d - c, b - a))}")
    ex = Exercise(f"X suit une loi uniforme sur [{a}, {b}]. Calcule P({c} < X < {d}).", Number(F(d - c, b - a)),
                    "Loi uniforme : la probabilité est proportionnelle à la longueur de l'intervalle : (d − c)/(b − a).",
                    curves(lambda t: np.where((t >= a) & (t <= b), 1 / (b - a), 0), xlim=(a - 2, b + 2)))
    ex.solution = _sol
    return ex


def l4_density(r):
    a, b = r.randint(-5, 2), r.randint(3, 10)
    _sol = lines(f"Aire sous la densité = 1 : hauteur × ({b} − {par(a)}) = 1", f"hauteur = 1/{b - a}")
    ex = Exercise(f"Densité d'une loi uniforme sur [{a}, {b}] (valeur constante de f sur l'intervalle) ?", Number(F(1, b - a)),
                    "L'aire sous une densité vaut 1 : hauteur × largeur = 1.", None)
    ex.solution = _sol
    return ex


def l4_expo(r):
    lam, t = r.choice([0.5, 1, 2, 0.1]), r.choice([1, 2, 3, 5])
    _sol = lines("P(X > t) = e^(−λt)", f"= e^(−{lam} × {t}) = e^(−{lam * t:g}) ≈ {math.exp(-lam * t):.4f}")
    ex = Exercise(f"X suit une loi exponentielle de paramètre λ = {lam}. Calcule P(X > {t})." + CALC,
                    Number(math.exp(-lam * t), 5e-3), "Loi exponentielle : P(X > t) = e^(−λt).",
                    curves(lambda u: lam * np.exp(-lam * u), xlim=(0, 4 / lam + 1)))
    ex.solution = _sol
    return ex


def l4_expo_mean(r):
    lam = r.choice([0.5, 2, 4, 0.25, 0.1])
    _sol = lines("E[X] = 1/λ", f"= 1/{lam} = {1 / lam:g}")
    ex = Exercise(f"Espérance d'une loi exponentielle de paramètre λ = {lam} ?", Number(1 / lam), "E[X] = 1/λ.", None)
    ex.solution = _sol
    return ex


def l4_zscore(r):
    mu, s = r.choice([100, 50, 170, 0]), r.choice([10, 15, 5, 2])
    x = mu + s * r.choice([-2, -1, -0.5, 1, 1.5, 2, 3])
    _sol = lines("z = (x − μ) / σ", f"= ({x:g} − {mu}) / {s} = {x - mu:g} / {s} = {(x - mu) / s:g}")
    ex = Exercise(f"X suit une loi normale N(μ = {mu}, σ = {s}). Valeur centrée réduite z de x = {x:g} ?", Number((x - mu) / s),
                    "z = (x − μ)/σ : de combien d'écarts-types x est loin de la moyenne.",
                    curves(lambda t: np.exp(-((t - mu) / s) ** 2 / 2), xlim=(mu - 4 * s, mu + 4 * s), points=[(x, 0)]))
    ex.solution = _sol
    return ex


def l4_68(r):
    k, p = r.choice([(1, 0.68), (2, 0.95), (3, 0.997)])
    _sol = lines("Règle 68 – 95 – 99,7 pour la loi normale", f"À ±{k} écart(s)-type(s) : environ {p * 100:g} %, soit {p}")
    ex = Exercise(f"X suit une loi normale. Probabilité (approximative) que X soit à moins de {k} écart(s)-type(s) de la "
                    "moyenne ?", Number(p, 1e-2), "Règle 68 – 95 – 99,7 : ±1σ → 68 %, ±2σ → 95 %, ±3σ → 99,7 %.",
                    curves(lambda t: np.exp(-t * t / 2) / math.sqrt(2 * math.pi), xlim=(-4, 4)))
    ex.solution = _sol
    return ex


def l4_half(r):
    mu = r.choice([0, 10, 100, -3])
    _sol = lines("La courbe en cloche est symétrique autour de sa moyenne.", f"Il y a autant de probabilité au-dessus de {mu} qu'en dessous : 0.5")
    ex = Exercise(f"X suit une loi normale de moyenne {mu}. Calcule P(X > {mu}).", Number(0.5),
                    "La loi normale est symétrique autour de sa moyenne.", None)
    ex.solution = _sol
    return ex


def l4_poisson(r):
    lam, k = r.choice([1, 2, 3, 4]), r.randint(0, 4)
    _sol = lines(f"P(X = {k}) = e^(−{lam}) × {lam}^{k} / {k}!", f"= {math.exp(-lam):.4f} × {lam ** k} / {math.factorial(k)}", f"≈ {math.exp(-lam) * lam ** k / math.factorial(k):.4f}")
    ex = Exercise(f"X suit une loi de Poisson de paramètre λ = {lam}. Calcule P(X = {k})." + CALC,
                    Number(math.exp(-lam) * lam ** k / math.factorial(k), 5e-3),
                    "Poisson : P(X = k) = e^(−λ)·λᵏ / k!. (Nombre d'événements rares dans un intervalle.)",
                    dist_bars([math.exp(-lam) * lam ** i / math.factorial(i) for i in range(9)], [str(i) for i in range(9)]))
    ex.solution = _sol
    return ex


def l4_linear_e(r):
    m, a, b = r.randint(-5, 10), nz_(r), r.randint(-9, 9)
    _sol = lines("E[aX + b] = a·E[X] + b", f"= {a} × {par(m)} {'+' if b >= 0 else '−'} {abs(b)} = {a * m + b}")
    ex = Exercise(f"E[X] = {m}. Calcule E[{a}X + {b}].".replace("+ -", "− "), Number(a * m + b),
                    "L'espérance est linéaire : E[aX + b] = a·E[X] + b.", None)
    ex.solution = _sol
    return ex


def l4_linear_var(r):
    v, a, b = r.randint(1, 9), nz_(r), r.randint(-9, 9)
    _sol = lines("Var(aX + b) = a²·Var(X) (le décalage b ne change rien)", f"= {par(a)}² × {v} = {a * a} × {v} = {a * a * v}")
    ex = Exercise(f"Var(X) = {v}. Calcule Var({a}X + {b}).".replace("+ -", "− "), Number(a * a * v),
                    "Var(aX + b) = a²·Var(X) : décaler ne change pas la dispersion, multiplier la change au carré.", None)
    ex.solution = _sol
    return ex


def nz_(r):
    return r.choice([-3, -2, 2, 3, 4, 5])


# ================================================================ NIVEAU 5

def l5_entropy(r):
    ps = r.choice([[0.5, 0.5], [0.25, 0.25, 0.5], [0.25] * 4, [0.125] * 8, [0.5, 0.25, 0.125, 0.125], [1.0]])
    _terms = " + ".join(f"{p:g} × {-math.log2(p):g}" for p in ps)
    _sol = lines("H = −Σ pᵢ·log₂(pᵢ) = Σ pᵢ·log₂(1/pᵢ)", f"= {_terms}" if len(ps) > 1 else "Une seule issue certaine : aucune incertitude.",
                 f"= {-sum(p * math.log2(p) for p in ps):g} bit(s)")
    ex = Exercise(f"Entropie (en bits) de la distribution {ps} ?", Number(-sum(p * math.log2(p) for p in ps)), M_INFO,
                    dist_bars(ps))
    ex.solution = _sol
    return ex


def l5_cross_entropy(r):
    q = r.choice([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.25, 0.5, 0.25]])
    k = r.randint(0, 2)
    _sol = lines(f"p one-hot : seul le terme de la classe {k + 1} reste", f"H = −ln(q_{k + 1}) = −ln({q[k]}) ≈ {-math.log(q[k]):.4f}")
    ex = Exercise(f"Vraie classe = {k + 1} (one-hot). Le modèle prédit q = {q}. Entropie croisée −Σ pᵢ·ln(qᵢ) ?" + CALC,
                    Number(-math.log(q[k]), 5e-3), M_INFO + "\nAvec p one-hot, un seul terme reste : −ln(q_vraie classe).",
                    dist_bars(q, title="prédiction du modèle"))
    ex.solution = _sol
    return ex


def l5_kl(r):
    p, q = r.choice([([0.5, 0.5], [0.25, 0.75]), ([0.5, 0.5], [0.9, 0.1]), ([0.75, 0.25], [0.5, 0.5])])
    _sol = lines("KL(p‖q) = Σ pᵢ·ln(pᵢ/qᵢ)", " + ".join(f"{a} × ln({a}/{b})" for a, b in zip(p, q)),
                 " + ".join(f"({a * math.log(a / b):.4f})" for a, b in zip(p, q)) + f" ≈ {sum(a * math.log(a / b) for a, b in zip(p, q)):.4f}")
    ex = Exercise(f"KL(p‖q) en nats avec p = {p}, q = {q} ?" + CALC,
                    Number(sum(a * math.log(a / b) for a, b in zip(p, q)), 5e-3),
                    M_INFO + "\nKL ≥ 0, et vaut 0 seulement si p = q. Elle n'est pas symétrique.", None)
    ex.solution = _sol
    return ex


def l5_likelihood(r):
    p = r.choice([0.3, 0.5, 0.6, 0.8])
    data = [r.choice([0, 1]) for _ in range(4)]
    k = sum(data)
    _sol = lines(f"{k} pile(s) (prob {p}) et {4 - k} face(s) (prob {1 - p:g})", f"P = {p}^{k} × {1 - p:g}^{4 - k} ≈ {p ** k * (1 - p) ** (4 - k):.4f}")
    ex = Exercise(f"Pièce de probabilité de pile p = {p}. Observations (1 = pile) : {data}. Vraisemblance "
                    "P(données | p) ?" + CALC, Number(p ** k * (1 - p) ** (4 - k), 5e-3),
                    "Observations indépendantes : produit de p pour chaque 1 et de (1 − p) pour chaque 0.", None)
    ex.solution = _sol
    return ex


def l5_mle(r):
    data = [r.choice([0, 1]) for _ in range(r.choice([5, 8, 10]))]
    _sol = lines(f"Nombre de succès : {sum(data)} sur {len(data)}", f"p̂ = {sum(data)}/{len(data)} = {fmt(F(sum(data), len(data)))}")
    ex = Exercise(f"Observations (1 = succès) : {data}. Estimateur du maximum de vraisemblance de p ?",
                    Number(F(sum(data), len(data))),
                    "Pour une loi de Bernoulli, maximiser la (log-)vraisemblance donne p̂ = (nombre de succès) / n "
                    "(dérive le log-vraisemblance et annule-le).", None)
    ex.solution = _sol
    return ex


def l5_naive_bayes(r):
    ps, _pa = r.choice([0.3, 0.4, 0.5]), r.choice([0.6, 0.8])
    pws, pwh = r.choice([0.6, 0.8, 0.5]), r.choice([0.05, 0.1, 0.2])
    post = pws * ps / (pws * ps + pwh * (1 - ps))
    _pw = pws * ps + pwh * (1 - ps)
    _sol = lines(f"P(« gratuit ») = {pws} × {ps} + {pwh} × {1 - ps:g} = {_pw:g}", f"P(spam | « gratuit ») = {pws} × {ps} / {_pw:g} = {pws * ps:g} / {_pw:g} ≈ {post:.4f}")
    ex = Exercise(f"{ps * 100:g} % des mails sont des spams. Le mot « gratuit » apparaît dans {pws * 100:g} % des spams "
                    f"et {pwh * 100:g} % des non-spams. Un mail contient « gratuit » : probabilité que ce soit un spam ?" + CALC,
                    Number(post, 5e-3), M_COND, None)
    ex.solution = _sol
    return ex


def l5_softmax_prob(r):
    z = [r.randint(-1, 3) for _ in range(3)]
    k = r.randint(0, 2)
    s = sum(math.exp(v) for v in z)
    _e = [math.exp(v) for v in z]
    _sol = lines("e^z : " + ", ".join(f"{ev:.4f}" for ev in _e), f"Somme ≈ {s:.4f}", f"p_{k + 1} = {_e[k]:.4f} / {s:.4f} ≈ {_e[k] / s:.4f}")
    ex = Exercise(f"Logits z = {z}. Probabilité de la classe {k + 1} après softmax ?" + CALC, Number(math.exp(z[k]) / s, 5e-3),
                    "softmax(z)ₖ = e^(zₖ) / Σ e^(zⱼ).", bars(["1", "2", "3"], z, "logits"))
    ex.solution = _sol
    return ex


def l5_expected_loss(r):
    ps = [F(1, 2), F(1, 3), F(1, 6)]
    losses = [r.randint(0, 9) for _ in range(3)]
    _sol = lines(" + ".join(f"{fmt(p)} × {l}" for p, l in zip(ps, losses)), f"= {fmt(sum(p * l for p, l in zip(ps, losses)))}")
    ex = Exercise(f"Trois scénarios de probabilités {[str(p) for p in ps]} entraînent des pertes {losses}. "
                    "Perte espérée ?".replace("'", ""), Number(sum(p * l for p, l in zip(ps, losses))),
                    "Perte espérée = Σ P(scénario)·perte (une espérance).", None)
    ex.solution = _sol
    return ex


def l5_entropy_coin(r):
    p = r.choice([0.5, 0.1, 0.25, 0.9, 0.75])
    _sol = lines(f"H = −[{p} × log₂({p}) + {1 - p:g} × log₂({1 - p:g})]", f"= −[{p} × ({math.log2(p):.4f}) + {1 - p:g} × ({math.log2(1 - p):.4f})]",
                 f"≈ {-(p * math.log2(p) + (1 - p) * math.log2(1 - p)):.4f} bit")
    ex = Exercise(f"Entropie (en bits) d'une pièce biaisée avec P(pile) = {p} ?" + CALC,
                    Number(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)), 5e-3), M_INFO,
                    curves(lambda t: -(t * np.log2(t) + (1 - t) * np.log2(1 - t)), xlim=(0.001, 0.999),
                           points=[(p, -(p * math.log2(p) + (1 - p) * math.log2(1 - p)))]))
    ex.solution = _sol
    return ex


def l5_perplexity(r):
    h = r.choice([1, 2, 3, 4, 0.5])
    _sol = lines(f"Perplexité = 2^H = 2^{h}", f"= {2 ** h:g}")
    ex = Exercise(f"Un modèle de langage a une entropie croisée de {h} bits par mot. Sa perplexité vaut 2^H. Combien ?",
                    Number(2 ** h), "Perplexité = 2^H : le nombre « effectif » de choix équiprobables entre lesquels le modèle hésite.",
                    None)
    ex.solution = _sol
    return ex


LEVELS = {
    1: [l1_die, l1_coin, l1_cards, l1_complement, l1_urn, l1_disjoint, l1_two_dice, l1_at_least_one, l1_expect_die,
        l1_percent],
    2: [l2_indep, l2_union, l2_cond_table, l2_tree_without, l2_tree_with, l2_at_least, l2_factorial, l2_comb, l2_arr,
        l2_lottery],
    3: [l3_bayes, l3_total, l3_binom, l3_binom_mean, l3_binom_var, l3_geometric, l3_expect, l3_var, l3_cdf, l3_indep_check],
    4: [l4_uniform, l4_density, l4_expo, l4_expo_mean, l4_zscore, l4_68, l4_half, l4_poisson, l4_linear_e, l4_linear_var],
    5: [l5_entropy, l5_cross_entropy, l5_kl, l5_likelihood, l5_mle, l5_naive_bayes, l5_softmax_prob, l5_expected_loss,
        l5_entropy_coin, l5_perplexity],
}
