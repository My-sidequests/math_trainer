"""Probabilités : dénombrement, conditionnelles, Bayes, lois discrètes et continues, théorie de l'information."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Number
from ..core import Exercise, bars, curves

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
    return Exercise(f"On lance un dé équilibré à 6 faces. Probabilité d'obtenir {txt} ?", Number(F(fav, 6)), M_BASIC,
                    dist_bars([F(1, 6)] * 6, title="loi du dé"))


def l1_coin(r):
    n = r.randint(2, 4)
    return Exercise(f"On lance {n} fois une pièce équilibrée. Probabilité d'obtenir {n} fois pile ?", Number(F(1, 2 ** n)),
                    "Les lancers sont indépendants : on multiplie les probabilités (½ à chaque fois).", None)


def l1_cards(r):
    q = r.choice([("un cœur", 13), ("un as", 4), ("une figure (valet, dame, roi)", 12), ("une carte rouge", 26),
                  ("l'as de pique", 1)])
    return Exercise(f"On tire une carte au hasard dans un jeu de 52 cartes. Probabilité de tirer {q[0]} ?",
                    Number(F(q[1], 52)), M_BASIC + "\nUn jeu : 4 couleurs × 13 valeurs.", None)


def l1_complement(r):
    p = r.choice([F(1, 3), F(2, 5), F(1, 10), F(3, 8), F(7, 20)])
    return Exercise(f"La probabilité qu'il pleuve demain est {p}. Probabilité qu'il ne pleuve pas ?", Number(1 - p), M_BASIC,
                    dist_bars([p, 1 - p], ["pluie", "pas de pluie"]))


def l1_urn(r):
    rd, bl, gr = r.randint(1, 8), r.randint(1, 8), r.randint(0, 6)
    return Exercise(f"Une urne contient {rd} boules rouges, {bl} bleues et {gr} vertes. Probabilité de tirer une bleue ?",
                    Number(F(bl, rd + bl + gr)), M_BASIC,
                    bars(["rouge", "bleue", "verte"], [rd, bl, gr], "contenu de l'urne"))


def l1_disjoint(r):
    a, b = r.sample(range(1, 7), 2)
    return Exercise(f"Dé à 6 faces. A = « obtenir {a} », B = « obtenir {b} ». Calcule P(A ou B).", Number(F(2, 6)),
                    "A et B ne peuvent pas arriver ensemble (incompatibles) : P(A ou B) = P(A) + P(B).",
                    dist_bars([F(1, 6)] * 6, title="loi du dé"))


def l1_two_dice(r):
    s = r.randint(2, 12)
    fav = 6 - abs(s - 7)
    return Exercise(f"On lance deux dés. Probabilité que la somme fasse {s} ?", Number(F(fav, 36)),
                    "Il y a 6 × 6 = 36 couples équiprobables (dé1, dé2). Compte ceux dont la somme vaut " + str(s) + ".",
                    bars(list(range(2, 13)), [6 - abs(t - 7) for t in range(2, 13)], "nombre de couples par somme"))


def l1_at_least_one(r):
    n = r.randint(2, 4)
    return Exercise(f"On lance {n} fois une pièce. Probabilité d'obtenir AU MOINS une fois face ?", Number(1 - F(1, 2 ** n)),
                    "« Au moins une fois » : passe par le complémentaire « jamais » (= toujours pile), puis 1 − P(jamais).",
                    None)


def l1_expect_die(r):
    n = r.choice([4, 6, 8, 10, 12, 20])
    return Exercise(f"Espérance (valeur moyenne) d'un dé équilibré à {n} faces numérotées de 1 à {n} ?", Number(F(n + 1, 2)),
                    "E[X] = Σ valeur × probabilité. Ici chaque face a probabilité 1/" + str(n) + ".",
                    dist_bars([F(1, n)] * n))


def l1_percent(r):
    p = r.choice([5, 12, 25, 40, 75, 2.5])
    return Exercise(f"Un événement a {p} % de chances de se produire. Écris cette probabilité sous forme décimale.",
                    Number(p / 100), "Pourcentage → probabilité : divise par 100.", None)


# ================================================================ NIVEAU 2

def l2_indep(r):
    a, b = r.choice([F(1, 2), F(1, 3), F(2, 5), F(3, 4)]), r.choice([F(1, 2), F(1, 4), F(3, 5), F(1, 6)])
    return Exercise(f"A et B sont indépendants, P(A) = {a}, P(B) = {b}. Calcule P(A et B).", Number(a * b),
                    "Indépendants ⇔ P(A ∩ B) = P(A)·P(B).", None)


def l2_union(r):
    a, b = F(r.randint(2, 6), 10), F(r.randint(2, 6), 10)
    ab = F(r.randint(1, min(int(a * 10), int(b * 10))), 10)
    return Exercise(f"P(A) = {float(a)}, P(B) = {float(b)}, P(A et B) = {float(ab)}. Calcule P(A ou B).", Number(a + b - ab),
                    "P(A ∪ B) = P(A) + P(B) − P(A ∩ B) (on retire l'intersection comptée deux fois).", None)


def l2_cond_table(r):
    a, b, c, d = (r.randint(5, 40) for _ in range(4))
    return Exercise(f"Sur un groupe : {a} fumeurs malades, {b} fumeurs non malades, {c} non-fumeurs malades, "
                    f"{d} non-fumeurs non malades. Probabilité d'être malade SACHANT qu'on est fumeur ?",
                    Number(F(a, a + b)), M_COND + "\nSachant « fumeur » : on ne regarde que la ligne des fumeurs.",
                    bars(["fum. malade", "fum. sain", "non-fum. malade", "non-fum. sain"], [a, b, c, d]))


def l2_tree_without(r):
    rd, bl = r.randint(2, 6), r.randint(2, 6)
    n = rd + bl
    return Exercise(f"Urne : {rd} rouges, {bl} bleues. On tire 2 boules SANS remise. Probabilité de 2 rouges ?",
                    Number(F(rd, n) * F(rd - 1, n - 1)),
                    "Arbre : P(1re rouge) × P(2e rouge sachant 1re rouge). Après le 1er tirage, il reste une boule de moins.",
                    None)


def l2_tree_with(r):
    rd, bl = r.randint(1, 6), r.randint(1, 6)
    n = rd + bl
    return Exercise(f"Urne : {rd} rouges, {bl} bleues. On tire 2 boules AVEC remise. Probabilité d'une rouge puis une bleue ?",
                    Number(F(rd, n) * F(bl, n)), "Avec remise, les tirages sont indépendants : multiplie les probabilités.", None)


def l2_at_least(r):
    p, n = r.choice([0.1, 0.2, 0.05, 0.5]), r.randint(2, 6)
    return Exercise(f"Un essai réussit avec probabilité {p}. On fait {n} essais indépendants. Probabilité d'au moins un succès ?" + CALC,
                    Number(1 - (1 - p) ** n, 1e-3), "1 − P(aucun succès) = 1 − (1 − p)ⁿ.", None)


def l2_factorial(r):
    n = r.randint(3, 7)
    return Exercise(f"De combien de façons peut-on ranger {n} livres différents sur une étagère ?", Number(math.factorial(n)),
                    M_COUNT, None)


def l2_comb(r):
    n = r.randint(4, 10)
    k = r.randint(2, n - 1)
    return Exercise(f"Combien de groupes de {k} personnes peut-on former parmi {n} ? (l'ordre ne compte pas)",
                    Number(math.comb(n, k)), M_COUNT, None)


def l2_arr(r):
    n = r.randint(4, 9)
    k = r.randint(2, 3)
    return Exercise(f"Course de {n} coureurs : combien de podiums différents (1er, 2e{', 3e' if k == 3 else ''}) possibles ?",
                    Number(math.perm(n, k)), M_COUNT + "\nIci l'ordre compte.", None)


def l2_lottery(r):
    n, k = r.choice([(10, 3), (8, 2), (12, 3), (20, 2)])
    return Exercise(f"Loterie : on choisit {k} numéros parmi {n}. Probabilité d'avoir la bonne combinaison ?",
                    Number(F(1, math.comb(n, k))), M_COUNT + "\nUne seule combinaison gagnante parmi C(n, k).", None)


# ================================================================ NIVEAU 3

def l3_bayes(r):
    prev = r.choice([0.01, 0.02, 0.05, 0.1])
    sens, fp = r.choice([0.9, 0.95, 0.99]), r.choice([0.05, 0.1, 0.02])
    post = sens * prev / (sens * prev + fp * (1 - prev))
    return Exercise(f"Une maladie touche {prev * 100:g} % de la population. Un test détecte {sens * 100:g} % des malades, "
                    f"mais est positif chez {fp * 100:g} % des personnes saines. Si le test est positif, probabilité "
                    "d'être malade ?" + CALC, Number(post, 5e-3),
                    M_COND + "\nA = malade, B = test positif. Calcule d'abord P(positif) par la formule des probabilités totales.",
                    bars(["malades", "sains"], [prev, 1 - prev], "avant le test"))


def l3_total(r):
    pa = r.choice([0.3, 0.4, 0.6])
    pba, pbna = r.choice([0.2, 0.5, 0.9]), r.choice([0.1, 0.3, 0.6])
    return Exercise(f"P(A) = {pa}, P(B | A) = {pba}, P(B | non A) = {pbna}. Calcule P(B).", Number(pba * pa + pbna * (1 - pa)),
                    "Probabilités totales : P(B) = P(B | A)·P(A) + P(B | non A)·P(non A).", None)


def l3_binom(r):
    n = r.randint(3, 6)
    k = r.randint(0, n)
    p = r.choice([0.5, 0.2, 0.3])
    return Exercise(f"X suit B(n = {n}, p = {p}). Calcule P(X = {k})." + CALC, Number(binom_pmf(n, k, p), 5e-3), M_BINOM,
                    dist_bars([binom_pmf(n, i, p) for i in range(n + 1)], [str(i) for i in range(n + 1)], "loi de X"))


def l3_binom_mean(r):
    n, p = r.choice([10, 20, 50, 100]), r.choice([0.1, 0.25, 0.5, 0.3])
    return Exercise(f"X suit B(n = {n}, p = {p}). Espérance E[X] ?", Number(n * p), M_BINOM, None)


def l3_binom_var(r):
    n, p = r.choice([10, 20, 50, 100]), r.choice([0.1, 0.25, 0.5, 0.2])
    return Exercise(f"X suit B(n = {n}, p = {p}). Variance Var(X) ?", Number(n * p * (1 - p)), M_BINOM, None)


def l3_geometric(r):
    p, k = r.choice([F(1, 2), F(1, 6), F(1, 3)]), r.randint(1, 4)
    return Exercise(f"On répète un essai de probabilité de succès {p} jusqu'au premier succès. Probabilité que le premier "
                    f"succès arrive exactement au {k}e essai ?", Number((1 - p) ** (k - 1) * p),
                    "Il faut k − 1 échecs puis un succès : (1 − p)ᵏ⁻¹·p.", None)


def _table(r):
    vals = sorted(r.sample(range(-3, 8), 3))
    ws = [r.randint(1, 5) for _ in range(3)]
    s = sum(ws)
    return vals, [F(w, s) for w in ws]


def l3_expect(r):
    vals, ps = _table(r)
    return Exercise(f"X prend les valeurs {vals} avec probabilités {[str(p) for p in ps]}. Calcule E[X].".replace("'", ""),
                    Number(sum(v * p for v, p in zip(vals, ps))), "E[X] = Σ xᵢ·P(X = xᵢ).",
                    dist_bars(ps, [str(v) for v in vals]))


def l3_var(r):
    vals, ps = _table(r)
    m = sum(v * p for v, p in zip(vals, ps))
    return Exercise(f"X prend les valeurs {vals} avec probabilités {[str(p) for p in ps]}. Calcule Var(X).".replace("'", ""),
                    Number(sum(p * (v - m) ** 2 for v, p in zip(vals, ps))),
                    "Var(X) = E[(X − E[X])²] = E[X²] − E[X]². Calcule E[X] puis E[X²].", dist_bars(ps, [str(v) for v in vals]))


def l3_cdf(r):
    n, k = r.randint(3, 5), r.randint(0, 2)
    return Exercise(f"X suit B(n = {n}, p = 0.5). Calcule P(X ≤ {k})." + CALC,
                    Number(sum(binom_pmf(n, i, 0.5) for i in range(k + 1)), 5e-3),
                    M_BINOM + "\nP(X ≤ k) = P(X = 0) + ... + P(X = k).",
                    dist_bars([binom_pmf(n, i, 0.5) for i in range(n + 1)], [str(i) for i in range(n + 1)]))


def l3_indep_check(r):
    pa, pb = r.choice([0.5, 0.4, 0.2]), r.choice([0.5, 0.3, 0.6])
    pab = pa * pb if r.random() < 0.5 else round(pa * pb + r.choice([0.05, -0.04]), 2)
    return Exercise(f"P(A) = {pa}, P(B) = {pb}, P(A ∩ B) = {round(pab, 4)}. A et B sont-ils indépendants ? (oui / non)",
                    Choice("oui" if abs(pab - pa * pb) < 1e-9 else "non", ["oui", "non"]),
                    "Indépendants ⇔ P(A ∩ B) = P(A)·P(B). Calcule le produit et compare.", None)


# ================================================================ NIVEAU 4

def l4_uniform(r):
    a, b = 0, r.choice([10, 20, 4, 60])
    c, d = sorted(r.sample(range(a, b + 1), 2))
    return Exercise(f"X suit une loi uniforme sur [{a}, {b}]. Calcule P({c} < X < {d}).", Number(F(d - c, b - a)),
                    "Loi uniforme : la probabilité est proportionnelle à la longueur de l'intervalle : (d − c)/(b − a).",
                    curves(lambda t: np.where((t >= a) & (t <= b), 1 / (b - a), 0), xlim=(a - 2, b + 2)))


def l4_density(r):
    a, b = r.randint(-5, 2), r.randint(3, 10)
    return Exercise(f"Densité d'une loi uniforme sur [{a}, {b}] (valeur constante de f sur l'intervalle) ?", Number(F(1, b - a)),
                    "L'aire sous une densité vaut 1 : hauteur × largeur = 1.", None)


def l4_expo(r):
    lam, t = r.choice([0.5, 1, 2, 0.1]), r.choice([1, 2, 3, 5])
    return Exercise(f"X suit une loi exponentielle de paramètre λ = {lam}. Calcule P(X > {t})." + CALC,
                    Number(math.exp(-lam * t), 5e-3), "Loi exponentielle : P(X > t) = e^(−λt).",
                    curves(lambda u: lam * np.exp(-lam * u), xlim=(0, 4 / lam + 1)))


def l4_expo_mean(r):
    lam = r.choice([0.5, 2, 4, 0.25, 0.1])
    return Exercise(f"Espérance d'une loi exponentielle de paramètre λ = {lam} ?", Number(1 / lam), "E[X] = 1/λ.", None)


def l4_zscore(r):
    mu, s = r.choice([100, 50, 170, 0]), r.choice([10, 15, 5, 2])
    x = mu + s * r.choice([-2, -1, -0.5, 1, 1.5, 2, 3])
    return Exercise(f"X suit une loi normale N(μ = {mu}, σ = {s}). Valeur centrée réduite z de x = {x:g} ?", Number((x - mu) / s),
                    "z = (x − μ)/σ : de combien d'écarts-types x est loin de la moyenne.",
                    curves(lambda t: np.exp(-((t - mu) / s) ** 2 / 2), xlim=(mu - 4 * s, mu + 4 * s), points=[(x, 0)]))


def l4_68(r):
    k, p = r.choice([(1, 0.68), (2, 0.95), (3, 0.997)])
    return Exercise(f"X suit une loi normale. Probabilité (approximative) que X soit à moins de {k} écart(s)-type(s) de la "
                    "moyenne ?", Number(p, 1e-2), "Règle 68 – 95 – 99,7 : ±1σ → 68 %, ±2σ → 95 %, ±3σ → 99,7 %.",
                    curves(lambda t: np.exp(-t * t / 2) / math.sqrt(2 * math.pi), xlim=(-4, 4)))


def l4_half(r):
    mu = r.choice([0, 10, 100, -3])
    return Exercise(f"X suit une loi normale de moyenne {mu}. Calcule P(X > {mu}).", Number(0.5),
                    "La loi normale est symétrique autour de sa moyenne.", None)


def l4_poisson(r):
    lam, k = r.choice([1, 2, 3, 4]), r.randint(0, 4)
    return Exercise(f"X suit une loi de Poisson de paramètre λ = {lam}. Calcule P(X = {k})." + CALC,
                    Number(math.exp(-lam) * lam ** k / math.factorial(k), 5e-3),
                    "Poisson : P(X = k) = e^(−λ)·λᵏ / k!. (Nombre d'événements rares dans un intervalle.)",
                    dist_bars([math.exp(-lam) * lam ** i / math.factorial(i) for i in range(9)], [str(i) for i in range(9)]))


def l4_linear_e(r):
    m, a, b = r.randint(-5, 10), nz_(r), r.randint(-9, 9)
    return Exercise(f"E[X] = {m}. Calcule E[{a}X + {b}].".replace("+ -", "− "), Number(a * m + b),
                    "L'espérance est linéaire : E[aX + b] = a·E[X] + b.", None)


def l4_linear_var(r):
    v, a, b = r.randint(1, 9), nz_(r), r.randint(-9, 9)
    return Exercise(f"Var(X) = {v}. Calcule Var({a}X + {b}).".replace("+ -", "− "), Number(a * a * v),
                    "Var(aX + b) = a²·Var(X) : décaler ne change pas la dispersion, multiplier la change au carré.", None)


def nz_(r):
    return r.choice([-3, -2, 2, 3, 4, 5])


# ================================================================ NIVEAU 5

def l5_entropy(r):
    ps = r.choice([[0.5, 0.5], [0.25, 0.25, 0.5], [0.25] * 4, [0.125] * 8, [0.5, 0.25, 0.125, 0.125], [1.0]])
    return Exercise(f"Entropie (en bits) de la distribution {ps} ?", Number(-sum(p * math.log2(p) for p in ps)), M_INFO,
                    dist_bars(ps))


def l5_cross_entropy(r):
    q = r.choice([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.25, 0.5, 0.25]])
    k = r.randint(0, 2)
    return Exercise(f"Vraie classe = {k + 1} (one-hot). Le modèle prédit q = {q}. Entropie croisée −Σ pᵢ·ln(qᵢ) ?" + CALC,
                    Number(-math.log(q[k]), 5e-3), M_INFO + "\nAvec p one-hot, un seul terme reste : −ln(q_vraie classe).",
                    dist_bars(q, title="prédiction du modèle"))


def l5_kl(r):
    p, q = r.choice([([0.5, 0.5], [0.25, 0.75]), ([0.5, 0.5], [0.9, 0.1]), ([0.75, 0.25], [0.5, 0.5])])
    return Exercise(f"KL(p‖q) en nats avec p = {p}, q = {q} ?" + CALC,
                    Number(sum(a * math.log(a / b) for a, b in zip(p, q)), 5e-3),
                    M_INFO + "\nKL ≥ 0, et vaut 0 seulement si p = q. Elle n'est pas symétrique.", None)


def l5_likelihood(r):
    p = r.choice([0.3, 0.5, 0.6, 0.8])
    data = [r.choice([0, 1]) for _ in range(4)]
    k = sum(data)
    return Exercise(f"Pièce de probabilité de pile p = {p}. Observations (1 = pile) : {data}. Vraisemblance "
                    "P(données | p) ?" + CALC, Number(p ** k * (1 - p) ** (4 - k), 5e-3),
                    "Observations indépendantes : produit de p pour chaque 1 et de (1 − p) pour chaque 0.", None)


def l5_mle(r):
    data = [r.choice([0, 1]) for _ in range(r.choice([5, 8, 10]))]
    return Exercise(f"Observations (1 = succès) : {data}. Estimateur du maximum de vraisemblance de p ?",
                    Number(F(sum(data), len(data))),
                    "Pour une loi de Bernoulli, maximiser la (log-)vraisemblance donne p̂ = (nombre de succès) / n "
                    "(dérive le log-vraisemblance et annule-le).", None)


def l5_naive_bayes(r):
    ps, _pa = r.choice([0.3, 0.4, 0.5]), r.choice([0.6, 0.8])
    pws, pwh = r.choice([0.6, 0.8, 0.5]), r.choice([0.05, 0.1, 0.2])
    post = pws * ps / (pws * ps + pwh * (1 - ps))
    return Exercise(f"{ps * 100:g} % des mails sont des spams. Le mot « gratuit » apparaît dans {pws * 100:g} % des spams "
                    f"et {pwh * 100:g} % des non-spams. Un mail contient « gratuit » : probabilité que ce soit un spam ?" + CALC,
                    Number(post, 5e-3), M_COND, None)


def l5_softmax_prob(r):
    z = [r.randint(-1, 3) for _ in range(3)]
    k = r.randint(0, 2)
    s = sum(math.exp(v) for v in z)
    return Exercise(f"Logits z = {z}. Probabilité de la classe {k + 1} après softmax ?" + CALC, Number(math.exp(z[k]) / s, 5e-3),
                    "softmax(z)ₖ = e^(zₖ) / Σ e^(zⱼ).", bars(["1", "2", "3"], z, "logits"))


def l5_expected_loss(r):
    ps = [F(1, 2), F(1, 3), F(1, 6)]
    losses = [r.randint(0, 9) for _ in range(3)]
    return Exercise(f"Trois scénarios de probabilités {[str(p) for p in ps]} entraînent des pertes {losses}. "
                    "Perte espérée ?".replace("'", ""), Number(sum(p * l for p, l in zip(ps, losses))),
                    "Perte espérée = Σ P(scénario)·perte (une espérance).", None)


def l5_entropy_coin(r):
    p = r.choice([0.5, 0.1, 0.25, 0.9, 0.75])
    return Exercise(f"Entropie (en bits) d'une pièce biaisée avec P(pile) = {p} ?" + CALC,
                    Number(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)), 5e-3), M_INFO,
                    curves(lambda t: -(t * np.log2(t) + (1 - t) * np.log2(1 - t)), xlim=(0.001, 0.999),
                           points=[(p, -(p * math.log2(p) + (1 - p) * math.log2(1 - p)))]))


def l5_perplexity(r):
    h = r.choice([1, 2, 3, 4, 0.5])
    return Exercise(f"Un modèle de langage a une entropie croisée de {h} bits par mot. Sa perplexité vaut 2^H. Combien ?",
                    Number(2 ** h), "Perplexité = 2^H : le nombre « effectif » de choix équiprobables entre lesquels le modèle hésite.",
                    None)


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
