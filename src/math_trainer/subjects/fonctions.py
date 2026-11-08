"""Fonctions & graphes : images, affines, transformations, composition, limites."""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Expr, Number, Numbers
from ..core import Exercise, coef, curves, lin, nz, par, poly, signed

from ..core import fmt, lines, par, steps_lin, term  # noqa: E402,F811

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
    _sol = lines(f"On remplace x par {par(k)} : f({k}) = {a} × {par(k)} {signed(b)}", f"= {fmt(a * k)} {signed(b)} = {fmt(a * k + b)}")
    ex = Exercise(f"f(x) = {lin(a, b)}. Calcule f({k}).", Number(a * k + b), M_IMAGE,
                    curves(lambda t: a * t + b, xlim=(-8, 8)))
    ex.solution = _sol
    return ex


def l1_image_sq(r):
    k = r.randint(-9, 9)
    _sol = lines(f"f({k}) = {par(k)}² = {par(k)} × {par(k)} = {k * k}")
    ex = Exercise(f"f(x) = x². Calcule f({k}).", Number(k * k),
                    M_IMAGE + "\nPiège classique : (−3)² = 9, alors que −3² = −9.",
                    curves(lambda t: t ** 2, xlim=(-10, 10)))
    ex.solution = _sol
    return ex


def l1_read_b(r):
    a, b = nz(r, -3, 3), r.randint(-5, 5)
    _sol = lines("On regarde où la droite coupe l'axe vertical (x = 0).", f"Elle le coupe en y = {b} : c'est l'ordonnée à l'origine.",
                 f"(La droite tracée est y = {lin(a, b)}.)")
    ex = Exercise("Lis sur le graphe l'ordonnée à l'origine de la droite (la valeur de y quand x = 0).",
                    Number(b), "Regarde où la droite coupe l'axe vertical (x = 0).",
                    curves(lambda t: a * t + b, xlim=(-6, 6), ylim=(-10, 10)))
    ex.solution = _sol
    return ex


def l1_read_slope(r):
    a, b = nz(r, -3, 3), r.randint(-4, 4)
    x1, x2 = -2, 2
    _y1, _y2 = a * x1 + b, a * x2 + b
    _sol = lines(f"Points lus : ({x1} ; {fmt(_y1)}) et ({x2} ; {fmt(_y2)})",
                 f"Pente = ({fmt(_y2)} − {par(_y1)}) / ({x2} − {par(x1)}) = {fmt(_y2 - _y1)} / {x2 - x1} = {fmt(a)}")
    ex = Exercise("Quelle est la pente (coefficient directeur) de la droite ? Deux points sont marqués.",
                    Number(a),
                    "Pente = (variation de y) / (variation de x) = (y₂ − y₁) / (x₂ − x₁).\n"
                    "Lis les coordonnées des deux points rouges et applique la formule.",
                    curves(lambda t: a * t + b, xlim=(-5, 5), points=[(x1, a * x1 + b), (x2, a * x2 + b)]))
    ex.solution = _sol
    return ex


def l1_antecedent(r):
    a, b, x = nz(r, -6, 6), r.randint(-9, 9), r.randint(-6, 6)
    _sol = lines(f"On résout {lin(a, b)} = {fmt(a * x + b)}", steps_lin(a, b, a * x + b),
                 f"Vérification : f({fmt(x)}) = {a} × {par(x)} {signed(b)} = {fmt(a * x + b)} ✔")
    ex = Exercise(f"f(x) = {lin(a, b)}. Pour quelle valeur de x a-t-on f(x) = {a * x + b} ?", Number(x),
                    "On cherche l'antécédent : écris l'équation " + lin(a, b) + " = valeur, et résous-la.",
                    curves(lambda t: a * t + b, lambda t: a * x + b + 0 * t, xlim=(-8, 8)))
    ex.solution = _sol
    return ex


def l1_belongs(r):
    a, b, p = nz(r, -5, 5), r.randint(-8, 8), r.randint(-5, 5)
    q = a * p + b + r.choice([0, 0, 1, -1, 2])
    _sol = lines(f"On calcule y pour x = {p} : {a} × {par(p)} {signed(b)} = {fmt(a * p + b)}",
                 f"L'ordonnée du point est {q} : " + ("égalité, donc le point est sur la droite." if q == a * p + b
                                                      else f"{q} ≠ {fmt(a * p + b)}, donc le point n'est pas sur la droite."))
    ex = Exercise(f"Le point A({p} ; {q}) est-il sur la droite y = {lin(a, b)} ? (oui / non)",
                    Choice("oui" if q == a * p + b else "non", ["oui", "non"]),
                    "Un point est sur la courbe si ses coordonnées vérifient l'équation : "
                    "calcule y avec x = abscisse du point et compare à son ordonnée.",
                    None)
    ex.solution = _sol
    return ex


def l1_monotony(r):
    a, b = nz(r, -7, 7), r.randint(-9, 9)
    _sol = lines(f"f(x) = {lin(a, b)} : le coefficient de x est a = {a}.",
                 f"a {'> 0 : la droite monte, f est croissante' if a > 0 else '< 0 : la droite descend, f est décroissante'}.")
    ex = Exercise(f"La fonction f(x) = {lin(a, b)} est-elle croissante ou décroissante ?",
                    Choice("croissante" if a > 0 else "decroissante", ["croissante", "décroissante"]),
                    "Pour une droite y = ax + b, seul le signe de a compte : a > 0 elle monte, a < 0 elle descend. "
                    "b ne change rien.", None)
    ex.solution = _sol
    return ex


def l1_image_quad(r):
    a, c, k = nz(r, -4, 4), r.randint(-9, 9), r.randint(-5, 5)
    _sol = lines(f"f({k}) = {a} × {par(k)}² {signed(c)}", f"= {a} × {k * k} {signed(c)}", f"= {fmt(a * k * k)} {signed(c)} = {fmt(a * k * k + c)}")
    ex = Exercise(f"f(x) = {poly({2: a, 0: c})}. Calcule f({k}).", Number(a * k * k + c), M_IMAGE,
                    curves(lambda t: a * t ** 2 + c, xlim=(-7, 7)))
    ex.solution = _sol
    return ex


def l1_image_paren(r):
    a, b, c, k = nz(r, -5, 5), r.randint(-6, 6), r.randint(-9, 9), r.randint(-6, 6)
    _sol = lines(f"f({k}) = {coef(a)}({par(k)} {signed(-b)}) {signed(c)}", f"La parenthèse vaut {fmt(k - b)}",
                 f"= {a} × {par(k - b)} {signed(c)} = {fmt(a * (k - b))} {signed(c)} = {fmt(a * (k - b) + c)}")
    ex = Exercise(f"f(x) = {coef(a)}(x {signed(-b)}) {signed(c)}. Calcule f({k}).", Number(a * (k - b) + c), M_IMAGE,
                    curves(lambda t: a * (t - b) + c, xlim=(-8, 8)))
    ex.solution = _sol
    return ex


def l1_rate(r):
    x1, x2 = sorted(r.sample(range(-6, 7), 2))
    a, b = nz(r, -4, 4), r.randint(-6, 6)
    _sol = lines(f"Pente = (f({x2}) − f({x1})) / ({x2} − {par(x1)})",
                 f"= ({fmt(a * x2 + b)} − {par(a * x1 + b)}) / {x2 - x1} = {fmt(a * (x2 - x1))} / {x2 - x1} = {a}")
    ex = Exercise(f"f est une fonction affine avec f({x1}) = {a * x1 + b} et f({x2}) = {a * x2 + b}. "
                    "Quelle est sa pente ?", Number(a),
                    "Pente = (f(x₂) − f(x₁)) / (x₂ − x₁).", curves(lambda t: a * t + b, xlim=(-8, 8),
                                                                  points=[(x1, a * x1 + b), (x2, a * x2 + b)]))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 2

def l2_line_two_points(r):
    a, b = nz(r, -4, 4), r.randint(-6, 6)
    x1, x2 = r.sample(range(-5, 6), 2)
    _ya, _yb = a * x1 + b, a * x2 + b
    _sol = lines(f"Pente a = ({fmt(_yb)} − {par(_ya)}) / ({x2} − {par(x1)}) = {fmt(_yb - _ya)} / {x2 - x1} = {a}",
                 f"On remplace A dans y = {term(a, 'x')} + b : {fmt(_ya)} = {a} × {par(x1)} + b ⇒ b = {fmt(_ya)} − {par(a * x1)} = {b}",
                 f"Équation : y = {lin(a, b)}")
    ex = Exercise(f"Donne l'équation (y = ax + b) de la droite passant par A({x1} ; {a * x1 + b}) "
                    f"et B({x2} ; {a * x2 + b}).", Expr(f"{a}*x + {b}"),
                    "1. Pente a = (y_B − y_A) / (x_B − x_A).\n2. Remplace x et y par les coordonnées de A dans y = ax + b "
                    "pour trouver b.\nRéponse sous forme d'expression, ex : 2*x - 3",
                    curves(lambda t: a * t + b, xlim=(-7, 7), points=[(x1, a * x1 + b, "A"), (x2, a * x2 + b, "B")]))
    ex.solution = _sol
    return ex


def l2_slope_frac(r):
    x1, x2 = r.sample(range(-6, 7), 2)
    y1, y2 = r.sample(range(-6, 7), 2)
    _sol = lines(f"a = (y_B − y_A) / (x_B − x_A) = ({y2} − {par(y1)}) / ({x2} − {par(x1)})",
                 f"= {y2 - y1} / {x2 - x1} = {fmt(F(y2 - y1, x2 - x1))}")
    ex = Exercise(f"Calcule la pente de la droite passant par A({x1} ; {y1}) et B({x2} ; {y2}).",
                    Number(F(y2 - y1, x2 - x1)), "a = (y_B − y_A) / (x_B − x_A). Simplifie la fraction si possible.",
                    curves(lambda t: (y2 - y1) / (x2 - x1) * (t - x1) + y1, xlim=(-8, 8),
                           points=[(x1, y1, "A"), (x2, y2, "B")]))
    ex.solution = _sol
    return ex


def l2_zero(r):
    a, b = nz(r, -6, 6), r.randint(-12, 12)
    _sol = lines(f"On résout {lin(a, b)} = 0", steps_lin(a, b, 0))
    ex = Exercise(f"Pour quelle valeur de x la fonction f(x) = {lin(a, b)} s'annule-t-elle ?", Number(F(-b, a)),
                    "Résous f(x) = 0. Sur le graphe, c'est là où la droite coupe l'axe horizontal.",
                    curves(lambda t: a * t + b, xlim=(-10, 10)))
    ex.solution = _sol
    return ex


def l2_intersect_x(r):
    a, c = r.sample(range(-5, 6), 2)
    b, d = r.randint(-8, 8), r.randint(-8, 8)
    _sol = lines(f"{lin(a, b)} = {lin(c, d)}", f"{term(a - c, 'x')} = {fmt(d)} − {par(b)} = {fmt(d - b)}",
                 f"x = {fmt(d - b)} / {par(a - c)} = {fmt(F(d - b, a - c))}")
    ex = Exercise(f"f(x) = {lin(a, b)} et g(x) = {lin(c, d)}. Pour quelle valeur de x a-t-on f(x) = g(x) ?",
                    Number(F(d - b, a - c)), "Écris f(x) = g(x) et résous l'équation du premier degré.",
                    curves(lambda t: a * t + b, lambda t: c * t + d, xlim=(-10, 10), labels=["f", "g"]))
    ex.solution = _sol
    return ex


def l2_parallel(r):
    a, b, p = nz(r, -4, 4), r.randint(-6, 6), r.randint(-5, 5)
    q = r.randint(-8, 8)
    _sol = lines(f"Même pente : D a pour équation y = {term(a, 'x')} + b", f"Le point ({p} ; {q}) : {q} = {a} × {par(p)} + b = {fmt(a * p)} + b",
                 f"b = {q} − {par(a * p)} = {fmt(q - a * p)}")
    ex = Exercise(f"La droite D est parallèle à y = {lin(a, b)} et passe par le point ({p} ; {q}). "
                    "Quelle est son ordonnée à l'origine b ?", Number(q - a * p),
                    "Parallèles = même pente. Donc D : y = " + str(a) + "x + b. Remplace x et y par les "
                    "coordonnées du point et résous en b.",
                    curves(lambda t: a * t + b, xlim=(-8, 8), points=[(p, q)]))
    ex.solution = _sol
    return ex


def l2_read_line(r):
    a, b = nz(r, -3, 3), r.randint(-5, 5)
    _sol = lines(f"La droite coupe l'axe vertical en y = {b}, donc b = {b}.",
                 f"Quand x augmente de 1, y {'augmente' if a > 0 else 'diminue'} de {abs(a)}, donc a = {a}.", f"f(x) = {lin(a, b)}")
    ex = Exercise("Donne l'expression de la fonction affine représentée (forme a*x + b).", Expr(f"{a}*x + {b}"),
                    "b = où la droite coupe l'axe vertical. a = de combien y monte (ou descend) quand x avance de 1.",
                    curves(lambda t: a * t + b, xlim=(-6, 6), ylim=(-12, 12)))
    ex.solution = _sol
    return ex


def l2_rate_constant(r):
    a, b, k = nz(r, -9, 9), r.randint(-9, 9), r.randint(-20, 20)
    _sol = lines(f"f({k + 1}) − f({k}) = [{a}({k + 1}) {signed(b)}] − [{a}({k}) {signed(b)}]",
                 f"= {a} × ({k + 1} − {par(k)}) = {a} × 1 = {a}", "Pour une fonction affine, avancer de 1 fait toujours varier f de a.")
    ex = Exercise(f"f(x) = {lin(a, b)}. Calcule f({k + 1}) − f({k}).", Number(a),
                    "Tu peux calculer les deux images puis soustraire... ou remarquer ce qui se passe "
                    "quand x augmente de 1 dans une fonction affine.", None)
    ex.solution = _sol
    return ex


def l2_mean_rate(r):
    x1, x2 = sorted(r.sample(range(-5, 6), 2))
    _sol = lines(f"f({x2}) = {x2 * x2} et f({x1}) = {x1 * x1}",
                 f"({x2 * x2} − {x1 * x1}) / ({x2} − {par(x1)}) = {x2 * x2 - x1 * x1} / {x2 - x1} = {x1 + x2}")
    ex = Exercise(f"f(x) = x². Calcule le taux de variation moyen de f entre {x1} et {x2}, "
                    f"c'est-à-dire (f({x2}) − f({x1})) / ({x2} − {par(x1)}).", Number(x1 + x2),
                    "C'est la pente de la droite qui relie les deux points de la courbe (une « corde »). "
                    "Calcule les deux images, puis le quotient.",
                    curves(lambda t: t ** 2, lambda t: (x1 + x2) * (t - x1) + x1 * x1, xlim=(-7, 7), ylim=(-5, 40),
                           points=[(x1, x1 * x1), (x2, x2 * x2)]))
    ex.solution = _sol
    return ex


def l2_perp(r):
    a = r.choice([F(n) for n in (-4, -3, -2, 2, 3, 4)] + [F(1, 2), F(-1, 3), F(2, 3)])
    _sol = lines(f"Produit des pentes = −1 : {fmt(a)} × m = −1", f"m = −1 / {par(a)} = {fmt(-1 / a)}")
    ex = Exercise(f"Quelle est la pente d'une droite perpendiculaire à une droite de pente {a} ?", Number(-1 / a),
                    "Deux droites (non verticales) sont perpendiculaires si le produit de leurs pentes vaut −1.",
                    curves(lambda t: float(a) * t, xlim=(-5, 5), ylim=(-5, 5), equal=True))
    ex.solution = _sol
    return ex


def l2_word(r):
    fixed, per = r.randint(5, 30), r.randint(2, 9)
    n = r.randint(3, 20)
    _sol = lines(f"Coût pour x séances : {fixed} + {per}x", f"On résout {fixed} + {per}x = {fixed + per * n}", steps_lin(per, fixed, fixed + per * n))
    ex = Exercise(f"Un abonnement coûte {fixed} € fixes plus {per} € par séance. "
                    f"Combien de séances pour un total de {fixed + per * n} € ?", Number(n),
                    "Écris le coût comme une fonction affine du nombre de séances x, puis résous coût(x) = total.",
                    curves(lambda t: fixed + per * t, xlim=(0, 25)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

def l3_vertex(r):
    h, k = r.randint(-5, 5), r.randint(-6, 6)
    _sol = lines(f"(x {signed(-h)})² est minimal (= 0) quand x = {h}.", f"Alors f({h}) = 0 {signed(k)} = {k}",
                 f"Sommet : ({h}, {k})")
    ex = Exercise(f"f(x) = (x {signed(-h)})² {signed(k)}. Donne les coordonnées du sommet (x, y).",
                    Numbers([h, k], ordered=True),
                    "(x − h)² est minimal (= 0) quand x = h. La forme (x − h)² + k est x² décalée de h "
                    "horizontalement et de k verticalement. Attention au signe dans la parenthèse.",
                    curves(lambda t: (t - h) ** 2 + k, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l3_read_parabola(r):
    h, k = r.randint(-4, 4), r.randint(-5, 5)
    _sol = lines(f"Le sommet de la courbe orange est en ({h} ; {k}).",
                 f"Décalage de {h} horizontalement : on remplace x par (x {signed(-h)}).", f"Décalage de {k} verticalement : on ajoute {k}.",
                 f"f(x) = (x {signed(-h)})² {signed(k)}")
    ex = Exercise("La courbe est la parabole y = x² décalée. Donne son expression.",
                    Expr(f"(x - ({h}))**2 + ({k})"),
                    "Lis les coordonnées (h, k) du sommet : la courbe est alors y = (x − h)² + k.",
                    curves(lambda t: t ** 2, lambda t: (t - h) ** 2 + k, xlim=(-7, 7), ylim=(-7, 15),
                           labels=["x² (référence)", "?"]))
    ex.solution = _sol
    return ex


def l3_inverse_fn(r):
    a, b = r.choice([(1, 2), (2, 3), (1, 4), (3, 4), (2, 5), (1, 3)])
    s = r.choice([1, -1])
    _sol = lines(f"f({fmt(F(s * a, b))}) = 1 / ({fmt(F(s * a, b))})", f"Diviser par {fmt(F(s * a, b))} = multiplier par son inverse {fmt(F(s * b, a))}",
                 f"Résultat : {fmt(F(s * b, a))}")
    ex = Exercise(f"f(x) = 1/x. Calcule f({s * a}/{b}).".replace("-", "−"), Number(F(s * b, a)),
                    "Diviser par une fraction, c'est multiplier par son inverse : 1 / (a/b) = b/a.",
                    curves(lambda t: 1 / t, xlim=(-4, 4), ylim=(-6, 6)))
    ex.solution = _sol
    return ex


def l3_sqrt(r):
    k = r.randint(0, 12)
    _sol = lines(f"On cherche le nombre positif dont le carré vaut {k * k} : {k}² = {k * k}", f"Donc √{k * k} = {k}")
    ex = Exercise(f"f(x) = √x. Calcule f({k * k}).", Number(k),
                    "√A est le nombre POSITIF dont le carré vaut A.", curves(lambda t: np.sqrt(t), xlim=(0, 150)))
    ex.solution = _sol
    return ex


def l3_domain_sqrt(r):
    a = r.randint(-9, 9)
    _sol = lines(f"Il faut x {signed(-a)} ≥ 0", f"donc x ≥ {a}", f"La plus petite valeur est {a}.")
    ex = Exercise(f"f(x) = √(x {signed(-a)}). Quelle est la plus petite valeur de x pour laquelle f(x) existe ?",
                    Number(a), "Une racine carrée n'existe que pour un nombre ≥ 0. Résous (x ...) ≥ 0.",
                    curves(lambda t: np.sqrt(t - a), xlim=(a - 4, a + 12)))
    ex.solution = _sol
    return ex


def l3_forbidden(r):
    a = r.randint(-9, 9)
    _sol = lines(f"Le dénominateur s'annule quand x {signed(-a)} = 0", f"donc x = {a} est interdite.")
    ex = Exercise(f"f(x) = 1/(x {signed(-a)}). Quelle valeur de x est interdite ?", Number(a),
                    "On ne divise jamais par 0 : cherche la valeur qui annule le dénominateur.",
                    curves(lambda t: 1 / (t - a), xlim=(a - 6, a + 6), ylim=(-6, 6)))
    ex.solution = _sol
    return ex


def l3_shift(r):
    h = nz(r, -6, 6)
    _sol = lines(f"g(x) = f(x {signed(-h)}) : g prend en x la valeur que f prend en x {signed(-h)}.",
                 f"Pour retrouver la même valeur, il faut un x plus grand de {fmt(h)} : la courbe est décalée de {h} "
                 f"({'vers la droite' if h > 0 else 'vers la gauche'}).")
    ex = Exercise(f"g(x) = f(x {signed(-h)}). La courbe de g est celle de f décalée horizontalement. "
                    "De combien (positif = vers la droite, négatif = vers la gauche) ?", Number(h),
                    "Contre-intuitif : f(x − 3) est décalée de +3 (vers la droite). Pour retrouver une même "
                    "valeur de f, il faut un x plus grand de 3.",
                    curves(lambda t: np.exp(-t ** 2), lambda t: np.exp(-(t - h) ** 2), xlim=(-8, 8),
                           labels=["f", "g"]))
    ex.solution = _sol
    return ex


def l3_abs_min(r):
    a, b = r.randint(-6, 6), r.randint(-6, 6)
    _sol = lines(f"|x {signed(-a)}| ≥ 0, et il vaut 0 pour x = {a}.", f"Le minimum de f est donc 0 {signed(b)} = {b}")
    ex = Exercise(f"f(x) = |x {signed(-a)}| {signed(b)}. Quelle est la valeur minimale de f ?", Number(b),
                    "Une valeur absolue est toujours ≥ 0, et elle vaut 0 quand son contenu est nul.",
                    curves(lambda t: np.abs(t - a) + b, xlim=(a - 6, a + 6)))
    ex.solution = _sol
    return ex


def l3_parity(r):
    kind = r.choice(["paire", "impaire", "ni"])
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    expr = {"paire": poly({4: a, 2: b, 0: r.randint(-5, 5)}),
            "impaire": poly({3: a, 1: b}),
            "ni": poly({2: a, 1: b, 0: nz(r, -5, 5)})}[kind]
    _why = {"paire": "seulement des puissances paires de x : f(−x) = f(x)",
            "impaire": "seulement des puissances impaires de x : f(−x) = −f(x)",
            "ni": "des puissances paires ET impaires (ou une constante avec une puissance impaire) : ni f(−x) = f(x), ni f(−x) = −f(x)"}
    _sol = lines("On calcule f(−x) en remplaçant x par −x : (−x)ⁿ = xⁿ si n est pair, −xⁿ si n est impair.",
                 f"f(x) = {expr} contient {_why[kind]}.", f"Donc f est {'ni paire ni impaire' if kind == 'ni' else kind}.")
    ex = Exercise(f"f(x) = {expr}. f est-elle paire, impaire ou ni l'une ni l'autre ? (paire / impaire / ni)",
                    Choice(kind, ["paire", "impaire", "ni"]),
                    "Calcule f(−x). Si f(−x) = f(x) : paire (symétrique par rapport à l'axe vertical). "
                    "Si f(−x) = −f(x) : impaire (symétrique par rapport à l'origine).", None)
    ex.solution = _sol
    return ex


def l3_image_poly(r):
    a, b, c, k = nz(r, -3, 3), r.randint(-6, 6), r.randint(-9, 9), r.randint(-4, 4)
    _sol = lines(f"f({k}) = {a} × {par(k)}² + {par(b)} × {par(k)} + {par(c)}",
                 f"= {fmt(a * k * k)} {signed(b * k)} {signed(c)} = {fmt(a * k * k + b * k + c)}")
    ex = Exercise(f"f(x) = {poly({2: a, 1: b, 0: c})}. Calcule f({k}).", Number(a * k * k + b * k + c),
                    "Remplace chaque x par " + par(k) + " et calcule. Commence par le carré.",
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(-6, 6)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 4

def l4_compose_num(r):
    a, b, c, k = nz(r, -4, 4), r.randint(-5, 5), r.randint(-5, 5), r.randint(-4, 4)
    _g = k * k + c
    _sol = lines(f"g({k}) = {par(k)}² {signed(c)} = {_g}", f"f(g({k})) = f({_g}) = {a} × {par(_g)} {signed(b)} = {fmt(a * _g + b)}")
    ex = Exercise(f"f(x) = {lin(a, b)} et g(x) = {poly({2: 1, 0: c})}. Calcule f(g({k})).",
                    Number(a * (k * k + c) + b),
                    "On part de l'intérieur : calcule d'abord g(" + str(k) + "), puis applique f au résultat.",
                    curves(lambda t: a * t + b, lambda t: t ** 2 + c, xlim=(-6, 6), labels=["f", "g"]))
    ex.solution = _sol
    return ex


def l4_compose_expr(r):
    a, b, c = nz(r, -4, 4), r.randint(-5, 5), nz(r, -5, 5)
    if r.random() < 0.5:
        label, target = "f(g(x))", f"({a})*(x**2 + ({c})) + ({b})"
    else:
        label, target = "g(f(x))", f"(({a})*x + ({b}))**2 + ({c})"
    if label == "f(g(x))":
        _sol = lines(f"f(g(x)) : on remplace x dans f par g(x) = {poly({2: 1, 0: c})}",
                     f"f(g(x)) = {coef(a)}({poly({2: 1, 0: c})}) {signed(b)}", f"= {poly({2: a, 0: a * c + b})}")
    else:
        _sol = lines(f"g(f(x)) : on remplace x dans g par f(x) = {lin(a, b)}",
                     f"g(f(x)) = ({lin(a, b)})² {signed(c)}", f"= {poly({2: a * a, 1: 2 * a * b, 0: b * b + c})}")
    ex = Exercise(f"f(x) = {lin(a, b)} et g(x) = {poly({2: 1, 0: c})}. Donne l'expression de {label}.",
                    Expr(target),
                    "Remplace chaque x de la fonction EXTÉRIEURE par l'expression entière de la fonction INTÉRIEURE "
                    "(entre parenthèses). Tu n'es pas obligé de développer.",
                    curves(lambda t: a * t + b, lambda t: t ** 2 + c, xlim=(-6, 6), labels=["f", "g"]))
    ex.solution = _sol
    return ex


def l4_inverse_expr(r):
    a, b = r.choice([2, 3, 4, 5, -2, -3]), r.randint(-8, 8)
    _sol = lines(f"y = {lin(a, b)}", f"y {signed(-b)} = {a}x", f"x = (y {signed(-b)}) / {par(a)}",
                 f"On renomme y en x : f⁻¹(x) = (x {signed(-b)}) / {par(a)}")
    ex = Exercise(f"f(x) = {lin(a, b)}. Donne l'expression de la fonction réciproque f⁻¹(x).",
                    Expr(f"(x - ({b})) / ({a})"),
                    "Écris y = " + lin(a, b) + ", puis isole x en fonction de y. Enfin, renomme y en x.\n"
                    "(La courbe de f⁻¹ est la symétrique de celle de f par rapport à la droite y = x.)",
                    curves(lambda t: a * t + b, lambda t: t, xlim=(-8, 8), ylim=(-8, 8), labels=["f", "y = x"],
                           equal=True))
    ex.solution = _sol
    return ex


def l4_inverse_num(r):
    a, b, x = nz(r, -5, 5), r.randint(-8, 8), r.randint(-6, 6)
    _sol = lines(f"f⁻¹({fmt(a * x + b)}) est le x tel que {lin(a, b)} = {fmt(a * x + b)}", steps_lin(a, b, a * x + b))
    ex = Exercise(f"f(x) = {lin(a, b)}. Calcule f⁻¹({a * x + b}).", Number(x),
                    "f⁻¹(k) est le nombre x tel que f(x) = k. Résous donc f(x) = k.",
                    curves(lambda t: a * t + b, lambda t: a * x + b + 0 * t, xlim=(-8, 8), labels=["f", "y = k"]))
    ex.solution = _sol
    return ex


def l4_expand(r):
    a, b = nz(r, -7, 7), nz(r, -7, 7)
    _sol = lines(f"(x {signed(a)})(x {signed(b)}) = x·x + x·{par(b)} + {par(a)}·x + {par(a)}·{par(b)}",
                 f"= x² {signed(b)}x {signed(a)}x {signed(a * b)}", f"= {poly({2: 1, 1: a + b, 0: a * b})}")
    ex = Exercise(f"Développe (x {signed(a)})(x {signed(b)}).", Expr(f"(x + {a})*(x + {b})"),
                    "Double distributivité : chaque terme de la 1re parenthèse multiplie chaque terme de la 2e "
                    "(4 produits), puis regroupe les termes semblables.", None)
    ex.solution = _sol
    return ex


def l4_cubic_roots(r):
    p, q, s = r.sample(range(-4, 5), 3)
    c2, c1, c0 = -(p + q + s), p * q + p * s + q * s, -p * q * s
    _P = lambda t: t ** 3 + c2 * t ** 2 + c1 * t + c0
    _sol = lines(f"On teste des diviseurs du terme constant {c0} : P(t) = 0 ?",
                 f"P({p}) = {_P(p)}, P({q}) = {_P(q)}, P({s}) = {_P(s)}",
                 f"Donc P(x) = (x {signed(-p)})(x {signed(-q)})(x {signed(-s)}) et les racines sont {p}, {q}, {s}.")
    ex = Exercise(f"P(x) = {poly({3: 1, 2: c2, 1: c1, 0: c0})}. Trouve ses 3 racines (indice : elles sont entières).",
                    Numbers([p, q, s]),
                    "Teste de petites valeurs entières (diviseurs du terme constant) : si P(a) = 0, a est racine.\n"
                    "Tu peux ensuite factoriser par (x − a) pour obtenir un polynôme de degré 2.",
                    curves(lambda t: (t - p) * (t - q) * (t - s), xlim=(-6, 6), ylim=(-40, 40)))
    ex.solution = _sol
    return ex


def l4_shift_expr(r):
    h, k = nz(r, -5, 5), nz(r, -5, 5)
    dh = "droite" if h > 0 else "gauche"
    dv = "haut" if k > 0 else "bas"
    _sol = lines(f"Décalage horizontal de {h} : x² devient (x {signed(-h)})²",
                 f"Décalage vertical de {k} : on ajoute {k}", f"g(x) = (x {signed(-h)})² {signed(k)}")
    ex = Exercise(f"On décale la courbe de f(x) = x² de {abs(h)} vers la {dh} et de {abs(k)} vers le {dv}. "
                    "Donne l'expression de la nouvelle fonction g.", Expr(f"(x - ({h}))**2 + ({k})"),
                    "Décalage horizontal de h : on remplace x par (x − h). Décalage vertical de k : on ajoute k.",
                    curves(lambda t: t ** 2, xlim=(-8, 8), ylim=(-8, 20)))
    ex.solution = _sol
    return ex


def l4_ff(r):
    a, b, k = nz(r, -3, 3), r.randint(-5, 5), r.randint(-4, 4)
    _f1 = a * k + b
    _sol = lines(f"f({k}) = {a} × {par(k)} {signed(b)} = {_f1}", f"f(f({k})) = f({_f1}) = {a} × {par(_f1)} {signed(b)} = {a * _f1 + b}")
    ex = Exercise(f"f(x) = {lin(a, b)}. Calcule f(f({k})).", Number(a * (a * k + b) + b),
                    "Calcule f(" + str(k) + ") d'abord, puis applique f une seconde fois au résultat.",
                    curves(lambda t: a * t + b, xlim=(-10, 10)))
    ex.solution = _sol
    return ex


def l4_count_sol(r):
    a, h, k = r.choice([1, -1]), r.randint(-3, 3), r.randint(-4, 4)
    m = k + r.choice([-2, -1, 0, 1, 2]) * a
    n = 1 if m == k else (2 if (m - k) * a > 0 else 0)
    sign = "" if a > 0 else "−"
    _sol = lines(f"Sommet de la parabole : ({h} ; {k}), ouverte vers le {'haut' if a > 0 else 'bas'}.",
                 f"La droite y = {m} est " + ("au niveau du sommet : 1 point de contact." if m == k else
                                              ("du côté de l'ouverture : elle coupe la parabole en 2 points." if n == 2 else
                                               "de l'autre côté du sommet : aucun point commun.")),
                 f"Nombre de solutions : {n}")
    ex = Exercise(f"f(x) = {sign}(x {signed(-h)})² {signed(k)}. Combien de solutions a l'équation f(x) = {m} ?",
                    Number(n),
                    "Trace mentalement la droite horizontale y = " + str(m) + " : combien de fois coupe-t-elle la "
                    "parabole ? Repère le sommet et le sens d'ouverture.",
                    curves(lambda t: a * (t - h) ** 2 + k, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l4_remainder(r):
    a = r.randint(-3, 3)
    c3, c2, c1, c0 = nz(r, -3, 3), r.randint(-5, 5), r.randint(-5, 5), r.randint(-9, 9)
    val = c3 * a ** 3 + c2 * a ** 2 + c1 * a + c0
    _sol = lines(f"Théorème du reste : le reste vaut P({a}).",
                 f"P({a}) = {c3} × {par(a)}³ + {par(c2)} × {par(a)}² + {par(c1)} × {par(a)} + {par(c0)}",
                 f"= {c3 * a ** 3} {signed(c2 * a * a)} {signed(c1 * a)} {signed(c0)} = {val}")
    ex = Exercise(f"Quel est le reste de la division de P(x) = {poly({3: c3, 2: c2, 1: c1, 0: c0})} "
                    f"par (x {signed(-a)}) ?", Number(val),
                    "Théorème du reste : le reste de la division de P(x) par (x − a) est P(a). "
                    "Inutile de poser la division.",
                    curves(lambda t: c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0, xlim=(-4, 4), ylim=(-40, 40)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 5

def l5_lim_ratio(r):
    a, c = nz(r, -6, 6), nz(r, -6, 6)
    b, d = r.randint(-9, 9), r.randint(-9, 9)
    _sol = lines(f"On factorise x : ({lin(a, b)}) / ({lin(c, d)}) = ({a} + {par(b)}/x) / ({c} + {par(d)}/x)",
                 "Quand x → +∞, les termes en 1/x tendent vers 0.", f"Limite : {a} / {par(c)} = {fmt(F(a, c))}")
    ex = Exercise(f"Calcule la limite quand x → +∞ de ({lin(a, b)}) / ({lin(c, d)}).", Number(F(a, c)),
                    "Quand x est énorme, les constantes deviennent négligeables. Factorise x en haut et en bas "
                    "(ou ne garde que les termes de plus haut degré).",
                    curves(lambda t: (a * t + b) / (c * t + d), lambda t: a / c + 0 * t, xlim=(-40, 40), ylim=(-10, 10)))
    ex.solution = _sol
    return ex


def l5_vertical_asym(r):
    a, b, c = nz(r, -4, 4), r.randint(-6, 6), r.randint(-6, 6)
    if a * c + b == 0:
        b += 1
    _sol = lines(f"Le dénominateur x {signed(-c)} s'annule en x = {c}.",
                 f"Le numérateur y vaut {a} × {par(c)} {signed(b)} = {a * c + b} ≠ 0 : la fonction tend vers ±∞.",
                 f"Asymptote verticale : x = {c}")
    ex = Exercise(f"f(x) = ({lin(a, b)}) / (x {signed(-c)}). Pour quelle valeur de x la courbe a-t-elle une "
                    "asymptote verticale ?", Number(c),
                    "Asymptote verticale = la fonction « explose » vers ±∞ : là où le dénominateur s'annule "
                    "(et pas le numérateur).",
                    curves(lambda t: (a * t + b) / (t - c), xlim=(c - 8, c + 8), ylim=(-15, 15)))
    ex.solution = _sol
    return ex


def l5_horizontal_asym(r):
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    p, q = r.randint(-6, 6), r.randint(1, 9)
    _sol = lines("Même degré (2) en haut et en bas : on ne garde que les termes de plus haut degré.",
                 f"{a}x² / ({b}x²) = {fmt(F(a, b))}", f"Asymptote horizontale : y = {fmt(F(a, b))}")
    ex = Exercise(f"f(x) = ({poly({2: a, 1: p})}) / ({poly({2: b, 0: q})}). "
                    "Quelle est l'asymptote horizontale (la limite quand x → +∞) ?", Number(F(a, b)),
                    "Même degré en haut et en bas : la limite est le quotient des coefficients des termes de plus "
                    "haut degré.", curves(lambda t: (a * t ** 2 + p * t) / (b * t ** 2 + q), xlim=(-30, 30), ylim=(-8, 8)))
    ex.solution = _sol
    return ex


def l5_lim_factor(r):
    a = nz(r, -7, 7)
    _sol = lines(f"x² − {a * a} = (x {signed(-a)})(x {signed(a)})", f"On simplifie par (x {signed(-a)}) : il reste x {signed(a)}",
                 f"Quand x → {a} : {a} {signed(a)} = {2 * a}")
    ex = Exercise(f"Calcule la limite quand x → {a} de (x² − {a * a}) / (x {signed(-a)}).", Number(2 * a),
                    "En remplaçant directement, on obtient 0/0 (forme indéterminée). Factorise le numérateur "
                    "(A² − B² = (A − B)(A + B)), simplifie, puis remplace.",
                    curves(lambda t: (t ** 2 - a * a) / (t - a), xlim=(a - 5, a + 5)))
    ex.solution = _sol
    return ex


def l5_lim_derivative(r):
    a = nz(r, -6, 6)
    _sol = lines(f"({par(a)} + h)² = {a * a} {signed(2 * a)}h + h²",
                 f"On retire {par(a)}² = {a * a} : il reste {term(2 * a, 'h')} + h²", f"On divise par h : {2 * a} + h",
                 f"Quand h → 0 : {2 * a}")
    ex = Exercise(f"Calcule la limite quand h → 0 de (({par(a)} + h)² − {par(a)}²) / h.", Number(2 * a),
                    "Développe (a + h)², simplifie le numérateur, factorise par h et simplifie avant de faire h → 0.\n"
                    "(C'est exactement la définition de la dérivée de x² en a !)",
                    curves(lambda t: t ** 2, xlim=(a - 3, a + 3), points=[(a, a * a)]))
    ex.solution = _sol
    return ex


def l5_continuity(r):
    a, b, c = nz(r, -3, 3), r.randint(-5, 5), r.randint(-3, 3)
    k = a * c + b - c * c
    _sol = lines(f"À gauche de {c} : f({c}) → {a} × {par(c)} {signed(b)} = {a * c + b}",
                 f"À droite : f({c}) = {par(c)}² + k = {c * c} + k", f"Raccord : {c * c} + k = {a * c + b} ⇒ k = {k}")
    ex = Exercise(f"f(x) = {lin(a, b)} si x < {c}, et f(x) = x² + k si x ≥ {c}.\n"
                    "Quelle valeur de k rend f continue (sans « saut ») ?", Number(k),
                    "Les deux morceaux doivent se rejoindre au point de raccord : égalise leurs valeurs en x = "
                    + str(c) + " et résous en k.",
                    curves(lambda t: np.where(t < c, a * t + b, np.nan), lambda t: np.where(t >= c, t ** 2, np.nan),
                           xlim=(c - 4, c + 4), labels=[f"{lin(a, b)} (x < {c})", f"x² + k avec k = 0 (x ≥ {c})"]))
    ex.solution = _sol
    return ex


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
    _cmp = ("haut > bas : la fraction explose, signe donné par " + f"{a}/{par(b)}" if da > db
            else ("haut < bas : la fraction tend vers 0" if da < db else f"degrés égaux : quotient {a}/{par(b)} = {fmt(F(a, b))}"))
    _sol = lines(f"Degré du numérateur : {da}, du dénominateur : {db}.", f"Comparaison : {_cmp}.")
    ex = Exercise(f"Calcule la limite quand x → +∞ de ({poly({da: a, 0: p0})}) / "
                    f"({poly({db: b, 0: q0})}).\n(Tu peux répondre inf ou -inf.)", ans,
                    "Compare les degrés : degré du haut > degré du bas → ±∞ (signe à étudier) ; "
                    "degré du haut < degré du bas → 0 ; égaux → quotient des coefficients dominants.",
                    curves(lambda t: (a * t ** da + p0) / (b * t ** db + q0), xlim=(2, 30)))
    ex.solution = _sol
    return ex


def l5_mean_rate_h(r):
    a = r.randint(-4, 4)
    h = r.choice([0.1, 0.5, 1, 2])
    _sol = lines(f"f({fmt(a + h)}) = {fmt((a + h) ** 2)} et f({a}) = {a * a}",
                 f"({fmt((a + h) ** 2)} − {a * a}) / {h} = {fmt((a + h) ** 2 - a * a)} / {h} = {fmt(2 * a + h)}",
                 f"(Formule générale : 2a + h = {2 * a} + {h}. Quand h → 0, on obtient la dérivée 2a = {2 * a}.)")
    ex = Exercise(f"f(x) = x². Calcule (f({par(a)} + {h}) − f({par(a)})) / {h}.", Number(2 * a + h),
                    "Calcule les deux images puis le quotient. Puis observe : qu'arriverait-il si h devenait "
                    "de plus en plus petit ?",
                    curves(lambda t: t ** 2, lambda t: (2 * a + h) * (t - a) + a * a, xlim=(a - 3, a + 3),
                           points=[(a, a * a), (a + h, (a + h) ** 2)]))
    ex.solution = _sol
    return ex


def l5_bump_max(r):
    a, b = r.randint(1, 9), r.randint(-4, 4)
    _sol = lines(f"(x {signed(-b)})² ≥ 0, donc 1 + (x {signed(-b)})² ≥ 1, avec égalité pour x = {b}.",
                 f"Le dénominateur est minimal (= 1) en x = {b} : f({b}) = {a} / 1 = {a}")
    ex = Exercise(f"f(x) = {a} / (1 + (x {signed(-b)})²). Quelle est la valeur maximale de f ?", Number(a),
                    "Le dénominateur est le plus PETIT possible quand le carré vaut 0. Une fraction à numérateur "
                    "positif est d'autant plus grande que son dénominateur est petit.",
                    curves(lambda t: a / (1 + (t - b) ** 2), xlim=(b - 6, b + 6)))
    ex.solution = _sol
    return ex


def l5_cubic_roots(r):
    a = r.randint(1, 6)
    _sol = lines(f"x³ − {a * a}x = x(x² − {a * a}) = x(x − {a})(x + {a})", f"Produit nul : x = 0, x = {a} ou x = −{a}")
    ex = Exercise(f"Trouve toutes les racines de f(x) = x³ − {a * a}x.", Numbers([0, a, -a]),
                    "Factorise par x, puis reconnais une différence de carrés. Produit nul.",
                    curves(lambda t: t ** 3 - a * a * t, xlim=(-a - 2, a + 2), ylim=(-2 * a ** 3 - 5, 2 * a ** 3 + 5)))
    ex.solution = _sol
    return ex


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
