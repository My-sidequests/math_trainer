"""Fonctions & graphes : images, affines, transformations, composition, limites."""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Expr, Number, Numbers
from ..core import Exercise, coef, curves, lin, nz, par, poly, signed

DESCRIPTIONS = {
    1: "Calculer des images, lire un graphe",
    2: "Fonctions affines (droites)",
    3: "Fonctions de référence et transformations",
    4: "Composition, réciproque, polynômes",
    5: "Limites et asymptotes",
}

M_IMAGE = ("Calculer f(k) = remplacer chaque x de l'expression par k (entre parenthèses si k est négatif), "
           "puis calculer en respectant les priorités : puissances, puis × et ÷, puis + et −.")


# ================================================================ NIVEAU 1

def l1_image_lin(r):
    a, b, k = nz(r, -6, 6), r.randint(-9, 9), r.randint(-6, 6)
    return Exercise(f"f(x) = {lin(a, b)}. Calcule f({k}).", Number(a * k + b), M_IMAGE,
                    curves(lambda t: a * t + b, xlim=(-8, 8)))


def l1_image_sq(r):
    k = r.randint(-9, 9)
    return Exercise(f"f(x) = x². Calcule f({k}).", Number(k * k),
                    M_IMAGE + "\nPiège classique : (−3)² = 9, alors que −3² = −9.",
                    curves(lambda t: t ** 2, xlim=(-10, 10)))


def l1_read_b(r):
    a, b = nz(r, -3, 3), r.randint(-5, 5)
    return Exercise("Lis sur le graphe l'ordonnée à l'origine de la droite (la valeur de y quand x = 0).",
                    Number(b), "Regarde où la droite coupe l'axe vertical (x = 0).",
                    curves(lambda t: a * t + b, xlim=(-6, 6), ylim=(-10, 10)))


def l1_read_slope(r):
    a, b = nz(r, -3, 3), r.randint(-4, 4)
    x1, x2 = -2, 2
    return Exercise("Quelle est la pente (coefficient directeur) de la droite ? Deux points sont marqués.",
                    Number(a),
                    "Pente = (variation de y) / (variation de x) = (y₂ − y₁) / (x₂ − x₁).\n"
                    "Lis les coordonnées des deux points rouges et applique la formule.",
                    curves(lambda t: a * t + b, xlim=(-5, 5), points=[(x1, a * x1 + b), (x2, a * x2 + b)]))


def l1_antecedent(r):
    a, b, x = nz(r, -6, 6), r.randint(-9, 9), r.randint(-6, 6)
    return Exercise(f"f(x) = {lin(a, b)}. Pour quelle valeur de x a-t-on f(x) = {a * x + b} ?", Number(x),
                    "On cherche l'antécédent : écris l'équation " + lin(a, b) + " = valeur, et résous-la.",
                    curves(lambda t: a * t + b, lambda t: a * x + b + 0 * t, xlim=(-8, 8)))


def l1_belongs(r):
    a, b, p = nz(r, -5, 5), r.randint(-8, 8), r.randint(-5, 5)
    q = a * p + b + r.choice([0, 0, 1, -1, 2])
    return Exercise(f"Le point A({p} ; {q}) est-il sur la droite y = {lin(a, b)} ? (oui / non)",
                    Choice("oui" if q == a * p + b else "non", ["oui", "non"]),
                    "Un point est sur la courbe si ses coordonnées vérifient l'équation : "
                    "calcule y avec x = abscisse du point et compare à son ordonnée.",
                    None)


def l1_monotony(r):
    a, b = nz(r, -7, 7), r.randint(-9, 9)
    return Exercise(f"La fonction f(x) = {lin(a, b)} est-elle croissante ou décroissante ?",
                    Choice("croissante" if a > 0 else "decroissante", ["croissante", "décroissante"]),
                    "Pour une droite y = ax + b, seul le signe de a compte : a > 0 elle monte, a < 0 elle descend. "
                    "b ne change rien.", None)


def l1_image_quad(r):
    a, c, k = nz(r, -4, 4), r.randint(-9, 9), r.randint(-5, 5)
    return Exercise(f"f(x) = {poly({2: a, 0: c})}. Calcule f({k}).", Number(a * k * k + c), M_IMAGE,
                    curves(lambda t: a * t ** 2 + c, xlim=(-7, 7)))


def l1_image_paren(r):
    a, b, c, k = nz(r, -5, 5), r.randint(-6, 6), r.randint(-9, 9), r.randint(-6, 6)
    return Exercise(f"f(x) = {coef(a)}(x {signed(-b)}) {signed(c)}. Calcule f({k}).", Number(a * (k - b) + c), M_IMAGE,
                    curves(lambda t: a * (t - b) + c, xlim=(-8, 8)))


def l1_rate(r):
    x1, x2 = sorted(r.sample(range(-6, 7), 2))
    a, b = nz(r, -4, 4), r.randint(-6, 6)
    return Exercise(f"f est une fonction affine avec f({x1}) = {a * x1 + b} et f({x2}) = {a * x2 + b}. "
                    "Quelle est sa pente ?", Number(a),
                    "Pente = (f(x₂) − f(x₁)) / (x₂ − x₁).", curves(lambda t: a * t + b, xlim=(-8, 8),
                                                                  points=[(x1, a * x1 + b), (x2, a * x2 + b)]))


# ================================================================ NIVEAU 2

def l2_line_two_points(r):
    a, b = nz(r, -4, 4), r.randint(-6, 6)
    x1, x2 = r.sample(range(-5, 6), 2)
    return Exercise(f"Donne l'équation (y = ax + b) de la droite passant par A({x1} ; {a * x1 + b}) "
                    f"et B({x2} ; {a * x2 + b}).", Expr(f"{a}*x + {b}"),
                    "1. Pente a = (y_B − y_A) / (x_B − x_A).\n2. Remplace x et y par les coordonnées de A dans y = ax + b "
                    "pour trouver b.\nRéponse sous forme d'expression, ex : 2*x - 3",
                    curves(lambda t: a * t + b, xlim=(-7, 7), points=[(x1, a * x1 + b, "A"), (x2, a * x2 + b, "B")]))


def l2_slope_frac(r):
    x1, x2 = r.sample(range(-6, 7), 2)
    y1, y2 = r.sample(range(-6, 7), 2)
    return Exercise(f"Calcule la pente de la droite passant par A({x1} ; {y1}) et B({x2} ; {y2}).",
                    Number(F(y2 - y1, x2 - x1)), "a = (y_B − y_A) / (x_B − x_A). Simplifie la fraction si possible.",
                    curves(lambda t: (y2 - y1) / (x2 - x1) * (t - x1) + y1, xlim=(-8, 8),
                           points=[(x1, y1, "A"), (x2, y2, "B")]))


def l2_zero(r):
    a, b = nz(r, -6, 6), r.randint(-12, 12)
    return Exercise(f"Pour quelle valeur de x la fonction f(x) = {lin(a, b)} s'annule-t-elle ?", Number(F(-b, a)),
                    "Résous f(x) = 0. Sur le graphe, c'est là où la droite coupe l'axe horizontal.",
                    curves(lambda t: a * t + b, xlim=(-10, 10)))


def l2_intersect_x(r):
    a, c = r.sample(range(-5, 6), 2)
    b, d = r.randint(-8, 8), r.randint(-8, 8)
    return Exercise(f"f(x) = {lin(a, b)} et g(x) = {lin(c, d)}. Pour quelle valeur de x a-t-on f(x) = g(x) ?",
                    Number(F(d - b, a - c)), "Écris f(x) = g(x) et résous l'équation du premier degré.",
                    curves(lambda t: a * t + b, lambda t: c * t + d, xlim=(-10, 10), labels=["f", "g"]))


def l2_parallel(r):
    a, b, p = nz(r, -4, 4), r.randint(-6, 6), r.randint(-5, 5)
    q = r.randint(-8, 8)
    return Exercise(f"La droite D est parallèle à y = {lin(a, b)} et passe par le point ({p} ; {q}). "
                    "Quelle est son ordonnée à l'origine b ?", Number(q - a * p),
                    "Parallèles = même pente. Donc D : y = " + str(a) + "x + b. Remplace x et y par les "
                    "coordonnées du point et résous en b.",
                    curves(lambda t: a * t + b, xlim=(-8, 8), points=[(p, q)]))


def l2_read_line(r):
    a, b = nz(r, -3, 3), r.randint(-5, 5)
    return Exercise("Donne l'expression de la fonction affine représentée (forme a*x + b).", Expr(f"{a}*x + {b}"),
                    "b = où la droite coupe l'axe vertical. a = de combien y monte (ou descend) quand x avance de 1.",
                    curves(lambda t: a * t + b, xlim=(-6, 6), ylim=(-12, 12)))


def l2_rate_constant(r):
    a, b, k = nz(r, -9, 9), r.randint(-9, 9), r.randint(-20, 20)
    return Exercise(f"f(x) = {lin(a, b)}. Calcule f({k + 1}) − f({k}).", Number(a),
                    "Tu peux calculer les deux images puis soustraire... ou remarquer ce qui se passe "
                    "quand x augmente de 1 dans une fonction affine.", None)


def l2_mean_rate(r):
    x1, x2 = sorted(r.sample(range(-5, 6), 2))
    return Exercise(f"f(x) = x². Calcule le taux de variation moyen de f entre {x1} et {x2}, "
                    f"c'est-à-dire (f({x2}) − f({x1})) / ({x2} − {par(x1)}).", Number(x1 + x2),
                    "C'est la pente de la droite qui relie les deux points de la courbe (une « corde »). "
                    "Calcule les deux images, puis le quotient.",
                    curves(lambda t: t ** 2, lambda t: (x1 + x2) * (t - x1) + x1 * x1, xlim=(-7, 7), ylim=(-5, 40),
                           points=[(x1, x1 * x1), (x2, x2 * x2)]))


def l2_perp(r):
    a = r.choice([F(n) for n in (-4, -3, -2, 2, 3, 4)] + [F(1, 2), F(-1, 3), F(2, 3)])
    return Exercise(f"Quelle est la pente d'une droite perpendiculaire à une droite de pente {a} ?", Number(-1 / a),
                    "Deux droites (non verticales) sont perpendiculaires si le produit de leurs pentes vaut −1.",
                    curves(lambda t: float(a) * t, xlim=(-5, 5), ylim=(-5, 5), equal=True))


def l2_word(r):
    fixed, per = r.randint(5, 30), r.randint(2, 9)
    n = r.randint(3, 20)
    return Exercise(f"Un abonnement coûte {fixed} € fixes plus {per} € par séance. "
                    f"Combien de séances pour un total de {fixed + per * n} € ?", Number(n),
                    "Écris le coût comme une fonction affine du nombre de séances x, puis résous coût(x) = total.",
                    curves(lambda t: fixed + per * t, xlim=(0, 25)))


# ================================================================ NIVEAU 3

def l3_vertex(r):
    h, k = r.randint(-5, 5), r.randint(-6, 6)
    return Exercise(f"f(x) = (x {signed(-h)})² {signed(k)}. Donne les coordonnées du sommet (x, y).",
                    Numbers([h, k], ordered=True),
                    "(x − h)² est minimal (= 0) quand x = h. La forme (x − h)² + k est x² décalée de h "
                    "horizontalement et de k verticalement. Attention au signe dans la parenthèse.",
                    curves(lambda t: (t - h) ** 2 + k, xlim=(h - 5, h + 5)))


def l3_read_parabola(r):
    h, k = r.randint(-4, 4), r.randint(-5, 5)
    return Exercise("La courbe est la parabole y = x² décalée. Donne son expression.",
                    Expr(f"(x - ({h}))**2 + ({k})"),
                    "Lis les coordonnées (h, k) du sommet : la courbe est alors y = (x − h)² + k.",
                    curves(lambda t: t ** 2, lambda t: (t - h) ** 2 + k, xlim=(-7, 7), ylim=(-7, 15),
                           labels=["x² (référence)", "?"]))


def l3_inverse_fn(r):
    a, b = r.choice([(1, 2), (2, 3), (1, 4), (3, 4), (2, 5), (1, 3)])
    s = r.choice([1, -1])
    return Exercise(f"f(x) = 1/x. Calcule f({s * a}/{b}).".replace("-", "−"), Number(F(s * b, a)),
                    "Diviser par une fraction, c'est multiplier par son inverse : 1 / (a/b) = b/a.",
                    curves(lambda t: 1 / t, xlim=(-4, 4), ylim=(-6, 6)))


def l3_sqrt(r):
    k = r.randint(0, 12)
    return Exercise(f"f(x) = √x. Calcule f({k * k}).", Number(k),
                    "√A est le nombre POSITIF dont le carré vaut A.", curves(lambda t: np.sqrt(t), xlim=(0, 150)))


def l3_domain_sqrt(r):
    a = r.randint(-9, 9)
    return Exercise(f"f(x) = √(x {signed(-a)}). Quelle est la plus petite valeur de x pour laquelle f(x) existe ?",
                    Number(a), "Une racine carrée n'existe que pour un nombre ≥ 0. Résous (x ...) ≥ 0.",
                    curves(lambda t: np.sqrt(t - a), xlim=(a - 4, a + 12)))


def l3_forbidden(r):
    a = r.randint(-9, 9)
    return Exercise(f"f(x) = 1/(x {signed(-a)}). Quelle valeur de x est interdite ?", Number(a),
                    "On ne divise jamais par 0 : cherche la valeur qui annule le dénominateur.",
                    curves(lambda t: 1 / (t - a), xlim=(a - 6, a + 6), ylim=(-6, 6)))


def l3_shift(r):
    h = nz(r, -6, 6)
    return Exercise(f"g(x) = f(x {signed(-h)}). La courbe de g est celle de f décalée horizontalement. "
                    "De combien (positif = vers la droite, négatif = vers la gauche) ?", Number(h),
                    "Contre-intuitif : f(x − 3) est décalée de +3 (vers la droite). Pour retrouver une même "
                    "valeur de f, il faut un x plus grand de 3.",
                    curves(lambda t: np.exp(-t ** 2), lambda t: np.exp(-(t - h) ** 2), xlim=(-8, 8),
                           labels=["f", "g"]))


def l3_abs_min(r):
    a, b = r.randint(-6, 6), r.randint(-6, 6)
    return Exercise(f"f(x) = |x {signed(-a)}| {signed(b)}. Quelle est la valeur minimale de f ?", Number(b),
                    "Une valeur absolue est toujours ≥ 0, et elle vaut 0 quand son contenu est nul.",
                    curves(lambda t: np.abs(t - a) + b, xlim=(a - 6, a + 6)))


def l3_parity(r):
    kind = r.choice(["paire", "impaire", "ni"])
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    expr = {"paire": poly({4: a, 2: b, 0: r.randint(-5, 5)}),
            "impaire": poly({3: a, 1: b}),
            "ni": poly({2: a, 1: b, 0: nz(r, -5, 5)})}[kind]
    return Exercise(f"f(x) = {expr}. f est-elle paire, impaire ou ni l'une ni l'autre ? (paire / impaire / ni)",
                    Choice(kind, ["paire", "impaire", "ni"]),
                    "Calcule f(−x). Si f(−x) = f(x) : paire (symétrique par rapport à l'axe vertical). "
                    "Si f(−x) = −f(x) : impaire (symétrique par rapport à l'origine).", None)


def l3_image_poly(r):
    a, b, c, k = nz(r, -3, 3), r.randint(-6, 6), r.randint(-9, 9), r.randint(-4, 4)
    return Exercise(f"f(x) = {poly({2: a, 1: b, 0: c})}. Calcule f({k}).", Number(a * k * k + b * k + c),
                    "Remplace chaque x par " + par(k) + " et calcule. Commence par le carré.",
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(-6, 6)))


# ================================================================ NIVEAU 4

def l4_compose_num(r):
    a, b, c, k = nz(r, -4, 4), r.randint(-5, 5), r.randint(-5, 5), r.randint(-4, 4)
    return Exercise(f"f(x) = {lin(a, b)} et g(x) = {poly({2: 1, 0: c})}. Calcule f(g({k})).",
                    Number(a * (k * k + c) + b),
                    "On part de l'intérieur : calcule d'abord g(" + str(k) + "), puis applique f au résultat.",
                    curves(lambda t: a * t + b, lambda t: t ** 2 + c, xlim=(-6, 6), labels=["f", "g"]))


def l4_compose_expr(r):
    a, b, c = nz(r, -4, 4), r.randint(-5, 5), nz(r, -5, 5)
    if r.random() < 0.5:
        label, target = "f(g(x))", f"({a})*(x**2 + ({c})) + ({b})"
    else:
        label, target = "g(f(x))", f"(({a})*x + ({b}))**2 + ({c})"
    return Exercise(f"f(x) = {lin(a, b)} et g(x) = {poly({2: 1, 0: c})}. Donne l'expression de {label}.",
                    Expr(target),
                    "Remplace chaque x de la fonction EXTÉRIEURE par l'expression entière de la fonction INTÉRIEURE "
                    "(entre parenthèses). Tu n'es pas obligé de développer.",
                    curves(lambda t: a * t + b, lambda t: t ** 2 + c, xlim=(-6, 6), labels=["f", "g"]))


def l4_inverse_expr(r):
    a, b = r.choice([2, 3, 4, 5, -2, -3]), r.randint(-8, 8)
    return Exercise(f"f(x) = {lin(a, b)}. Donne l'expression de la fonction réciproque f⁻¹(x).",
                    Expr(f"(x - ({b})) / ({a})"),
                    "Écris y = " + lin(a, b) + ", puis isole x en fonction de y. Enfin, renomme y en x.\n"
                    "(La courbe de f⁻¹ est la symétrique de celle de f par rapport à la droite y = x.)",
                    curves(lambda t: a * t + b, lambda t: t, xlim=(-8, 8), ylim=(-8, 8), labels=["f", "y = x"],
                           equal=True))


def l4_inverse_num(r):
    a, b, x = nz(r, -5, 5), r.randint(-8, 8), r.randint(-6, 6)
    return Exercise(f"f(x) = {lin(a, b)}. Calcule f⁻¹({a * x + b}).", Number(x),
                    "f⁻¹(k) est le nombre x tel que f(x) = k. Résous donc f(x) = k.",
                    curves(lambda t: a * t + b, lambda t: a * x + b + 0 * t, xlim=(-8, 8), labels=["f", "y = k"]))


def l4_expand(r):
    a, b = nz(r, -7, 7), nz(r, -7, 7)
    return Exercise(f"Développe (x {signed(a)})(x {signed(b)}).", Expr(f"(x + {a})*(x + {b})"),
                    "Double distributivité : chaque terme de la 1re parenthèse multiplie chaque terme de la 2e "
                    "(4 produits), puis regroupe les termes semblables.", None)


def l4_cubic_roots(r):
    p, q, s = r.sample(range(-4, 5), 3)
    c2, c1, c0 = -(p + q + s), p * q + p * s + q * s, -p * q * s
    return Exercise(f"P(x) = {poly({3: 1, 2: c2, 1: c1, 0: c0})}. Trouve ses 3 racines (indice : elles sont entières).",
                    Numbers([p, q, s]),
                    "Teste de petites valeurs entières (diviseurs du terme constant) : si P(a) = 0, a est racine.\n"
                    "Tu peux ensuite factoriser par (x − a) pour obtenir un polynôme de degré 2.",
                    curves(lambda t: (t - p) * (t - q) * (t - s), xlim=(-6, 6), ylim=(-40, 40)))


def l4_shift_expr(r):
    h, k = nz(r, -5, 5), nz(r, -5, 5)
    dh = "droite" if h > 0 else "gauche"
    dv = "haut" if k > 0 else "bas"
    return Exercise(f"On décale la courbe de f(x) = x² de {abs(h)} vers la {dh} et de {abs(k)} vers le {dv}. "
                    "Donne l'expression de la nouvelle fonction g.", Expr(f"(x - ({h}))**2 + ({k})"),
                    "Décalage horizontal de h : on remplace x par (x − h). Décalage vertical de k : on ajoute k.",
                    curves(lambda t: t ** 2, xlim=(-8, 8), ylim=(-8, 20)))


def l4_ff(r):
    a, b, k = nz(r, -3, 3), r.randint(-5, 5), r.randint(-4, 4)
    return Exercise(f"f(x) = {lin(a, b)}. Calcule f(f({k})).", Number(a * (a * k + b) + b),
                    "Calcule f(" + str(k) + ") d'abord, puis applique f une seconde fois au résultat.",
                    curves(lambda t: a * t + b, xlim=(-10, 10)))


def l4_count_sol(r):
    a, h, k = r.choice([1, -1]), r.randint(-3, 3), r.randint(-4, 4)
    m = k + r.choice([-2, -1, 0, 1, 2]) * a
    n = 1 if m == k else (2 if (m - k) * a > 0 else 0)
    sign = "" if a > 0 else "−"
    return Exercise(f"f(x) = {sign}(x {signed(-h)})² {signed(k)}. Combien de solutions a l'équation f(x) = {m} ?",
                    Number(n),
                    "Trace mentalement la droite horizontale y = " + str(m) + " : combien de fois coupe-t-elle la "
                    "parabole ? Repère le sommet et le sens d'ouverture.",
                    curves(lambda t: a * (t - h) ** 2 + k, xlim=(h - 5, h + 5)))


def l4_remainder(r):
    a = r.randint(-3, 3)
    c3, c2, c1, c0 = nz(r, -3, 3), r.randint(-5, 5), r.randint(-5, 5), r.randint(-9, 9)
    val = c3 * a ** 3 + c2 * a ** 2 + c1 * a + c0
    return Exercise(f"Quel est le reste de la division de P(x) = {poly({3: c3, 2: c2, 1: c1, 0: c0})} "
                    f"par (x {signed(-a)}) ?", Number(val),
                    "Théorème du reste : le reste de la division de P(x) par (x − a) est P(a). "
                    "Inutile de poser la division.",
                    curves(lambda t: c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0, xlim=(-4, 4), ylim=(-40, 40)))


# ================================================================ NIVEAU 5

def l5_lim_ratio(r):
    a, c = nz(r, -6, 6), nz(r, -6, 6)
    b, d = r.randint(-9, 9), r.randint(-9, 9)
    return Exercise(f"Calcule la limite quand x → +∞ de ({lin(a, b)}) / ({lin(c, d)}).", Number(F(a, c)),
                    "Quand x est énorme, les constantes deviennent négligeables. Factorise x en haut et en bas "
                    "(ou ne garde que les termes de plus haut degré).",
                    curves(lambda t: (a * t + b) / (c * t + d), lambda t: a / c + 0 * t, xlim=(-40, 40), ylim=(-10, 10)))


def l5_vertical_asym(r):
    a, b, c = nz(r, -4, 4), r.randint(-6, 6), r.randint(-6, 6)
    if a * c + b == 0:
        b += 1
    return Exercise(f"f(x) = ({lin(a, b)}) / (x {signed(-c)}). Pour quelle valeur de x la courbe a-t-elle une "
                    "asymptote verticale ?", Number(c),
                    "Asymptote verticale = la fonction « explose » vers ±∞ : là où le dénominateur s'annule "
                    "(et pas le numérateur).",
                    curves(lambda t: (a * t + b) / (t - c), xlim=(c - 8, c + 8), ylim=(-15, 15)))


def l5_horizontal_asym(r):
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    p, q = r.randint(-6, 6), r.randint(1, 9)
    return Exercise(f"f(x) = ({poly({2: a, 1: p})}) / ({poly({2: b, 0: q})}). "
                    "Quelle est l'asymptote horizontale (la limite quand x → +∞) ?", Number(F(a, b)),
                    "Même degré en haut et en bas : la limite est le quotient des coefficients des termes de plus "
                    "haut degré.", curves(lambda t: (a * t ** 2 + p * t) / (b * t ** 2 + q), xlim=(-30, 30), ylim=(-8, 8)))


def l5_lim_factor(r):
    a = nz(r, -7, 7)
    return Exercise(f"Calcule la limite quand x → {a} de (x² − {a * a}) / (x {signed(-a)}).", Number(2 * a),
                    "En remplaçant directement, on obtient 0/0 (forme indéterminée). Factorise le numérateur "
                    "(A² − B² = (A − B)(A + B)), simplifie, puis remplace.",
                    curves(lambda t: (t ** 2 - a * a) / (t - a), xlim=(a - 5, a + 5)))


def l5_lim_derivative(r):
    a = nz(r, -6, 6)
    return Exercise(f"Calcule la limite quand h → 0 de (({par(a)} + h)² − {par(a)}²) / h.", Number(2 * a),
                    "Développe (a + h)², simplifie le numérateur, factorise par h et simplifie avant de faire h → 0.\n"
                    "(C'est exactement la définition de la dérivée de x² en a !)",
                    curves(lambda t: t ** 2, xlim=(a - 3, a + 3), points=[(a, a * a)]))


def l5_continuity(r):
    a, b, c = nz(r, -3, 3), r.randint(-5, 5), r.randint(-3, 3)
    k = a * c + b - c * c
    return Exercise(f"f(x) = {lin(a, b)} si x < {c}, et f(x) = x² + k si x ≥ {c}.\n"
                    "Quelle valeur de k rend f continue (sans « saut ») ?", Number(k),
                    "Les deux morceaux doivent se rejoindre au point de raccord : égalise leurs valeurs en x = "
                    + str(c) + " et résous en k.",
                    curves(lambda t: np.where(t < c, a * t + b, np.nan), lambda t: np.where(t >= c, t ** 2, np.nan),
                           xlim=(c - 4, c + 4), labels=[f"{lin(a, b)} (x < {c})", f"x² + k avec k = 0 (x ≥ {c})"]))


def l5_lim_inf(r):
    da, db = r.randint(1, 3), r.randint(1, 3)
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    if da > db:
        ans = Number(float("inf") if a * b > 0 else float("-inf"))
    elif da < db:
        ans = Number(0)
    else:
        ans = Number(F(a, b))
    p0, q0 = r.randint(1, 5), r.randint(1, 5)
    return Exercise(f"Calcule la limite quand x → +∞ de ({poly({da: a, 0: p0})}) / "
                    f"({poly({db: b, 0: q0})}).\n(Tu peux répondre inf ou -inf.)", ans,
                    "Compare les degrés : degré du haut > degré du bas → ±∞ (signe à étudier) ; "
                    "degré du haut < degré du bas → 0 ; égaux → quotient des coefficients dominants.",
                    curves(lambda t: (a * t ** da + p0) / (b * t ** db + q0), xlim=(2, 30)))


def l5_mean_rate_h(r):
    a = r.randint(-4, 4)
    h = r.choice([0.1, 0.5, 1, 2])
    return Exercise(f"f(x) = x². Calcule (f({par(a)} + {h}) − f({par(a)})) / {h}.", Number(2 * a + h),
                    "Calcule les deux images puis le quotient. Puis observe : qu'arriverait-il si h devenait "
                    "de plus en plus petit ?",
                    curves(lambda t: t ** 2, lambda t: (2 * a + h) * (t - a) + a * a, xlim=(a - 3, a + 3),
                           points=[(a, a * a), (a + h, (a + h) ** 2)]))


def l5_bump_max(r):
    a, b = r.randint(1, 9), r.randint(-4, 4)
    return Exercise(f"f(x) = {a} / (1 + (x {signed(-b)})²). Quelle est la valeur maximale de f ?", Number(a),
                    "Le dénominateur est le plus PETIT possible quand le carré vaut 0. Une fraction à numérateur "
                    "positif est d'autant plus grande que son dénominateur est petit.",
                    curves(lambda t: a / (1 + (t - b) ** 2), xlim=(b - 6, b + 6)))


def l5_cubic_roots(r):
    a = r.randint(1, 6)
    return Exercise(f"Trouve toutes les racines de f(x) = x³ − {a * a}x.", Numbers([0, a, -a]),
                    "Factorise par x, puis reconnais une différence de carrés. Produit nul.",
                    curves(lambda t: t ** 3 - a * a * t, xlim=(-a - 2, a + 2), ylim=(-2 * a ** 3 - 5, 2 * a ** 3 + 5)))


LEVELS = {
    1: [l1_image_lin, l1_image_sq, l1_read_b, l1_read_slope, l1_antecedent, l1_belongs, l1_monotony, l1_image_quad,
        l1_image_paren, l1_rate],
    2: [l2_line_two_points, l2_slope_frac, l2_zero, l2_intersect_x, l2_parallel, l2_read_line, l2_rate_constant,
        l2_mean_rate, l2_perp, l2_word],
    3: [l3_vertex, l3_read_parabola, l3_inverse_fn, l3_sqrt, l3_domain_sqrt, l3_forbidden, l3_shift, l3_abs_min,
        l3_parity, l3_image_poly],
    4: [l4_compose_num, l4_compose_expr, l4_inverse_expr, l4_inverse_num, l4_expand, l4_cubic_roots, l4_shift_expr,
        l4_ff, l4_count_sol, l4_remainder],
    5: [l5_lim_ratio, l5_vertical_asym, l5_horizontal_asym, l5_lim_factor, l5_lim_derivative, l5_continuity,
        l5_lim_inf, l5_mean_rate_h, l5_bump_max, l5_cubic_roots],
}
