"""Équations : du premier degré aux systèmes."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Number, Numbers
from ..core import Exercise, curves, lin, nz, number_line, poly, rectangle, signed, sterm, term

DESCRIPTIONS = {
    1: "Premier degré simple (x + a = b, ax = b...)",
    2: "Premier degré : x des deux côtés, parenthèses, fractions",
    3: "Équations produit, carrés, valeur absolue, inéquations",
    4: "Second degré : factorisation, discriminant, formule",
    5: "Systèmes, équations rationnelles et racines",
}

M_ISOLATE = ("Le but est d'avoir x seul d'un côté.\n"
             "• Ce qui est ajouté à x : on le soustrait des DEUX côtés (et inversement).\n"
             "• Ce qui multiplie x : on divise les DEUX côtés par ce nombre (et inversement).\n"
             "Vérifie en remplaçant x par ta valeur dans l'équation de départ.")


def sides(fl, fr, xlim=(-10, 10), ylim=None):
    """Chaque membre de l'équation devient une courbe : la solution est l'abscisse de l'intersection."""
    return curves(fl, fr, xlim=xlim, ylim=ylim, labels=["membre de gauche", "membre de droite"])


# ================================================================ NIVEAU 1

def l1_add(r):
    x, a = r.randint(-12, 12), r.randint(1, 15)
    return Exercise(f"Résous : x + {a} = {x + a}", Number(x), M_ISOLATE,
                    sides(lambda t: t + a, lambda t: (x + a) + 0 * t, (x - 8, x + 8)))


def l1_sub(r):
    x, a = r.randint(-12, 12), r.randint(1, 15)
    return Exercise(f"Résous : x − {a} = {x - a}", Number(x), M_ISOLATE,
                    sides(lambda t: t - a, lambda t: (x - a) + 0 * t, (x - 8, x + 8)))


def l1_mul(r):
    x, a = r.randint(-10, 10), r.randint(2, 9)
    return Exercise(f"Résous : {a}x = {a * x}", Number(x), M_ISOLATE,
                    sides(lambda t: a * t, lambda t: a * x + 0 * t, (x - 6, x + 6)))


def l1_div(r):
    b, a = r.randint(-9, 9), r.randint(2, 8)
    return Exercise(f"Résous : x / {a} = {b}", Number(a * b),
                    "Diviser par un nombre se « défait » en multipliant les deux côtés par ce nombre.",
                    sides(lambda t: t / a, lambda t: b + 0 * t, (a * b - 15, a * b + 15)))


def l1_affine(r):
    x, a, b = r.randint(-8, 8), r.randint(2, 7), nz(r, -12, 12)
    return Exercise(f"Résous : {lin(a, b)} = {a * x + b}", Number(x),
                    M_ISOLATE + "\nCommence par le terme constant, puis le coefficient de x.",
                    sides(lambda t: a * t + b, lambda t: a * x + b + 0 * t, (x - 6, x + 6)))


def l1_minus_x(r):
    x, a = r.randint(-10, 10), r.randint(1, 15)
    return Exercise(f"Résous : {a} − x = {a - x}", Number(x),
                    "« −x » : ajoute x des deux côtés pour le rendre positif, puis isole-le.",
                    sides(lambda t: a - t, lambda t: (a - x) + 0 * t, (x - 8, x + 8)))


def l1_affine_neg(r):
    x, a, b = r.randint(-8, 8), r.randint(2, 7), r.randint(1, 12)
    return Exercise(f"Résous : {a}x − {b} = {a * x - b}", Number(x), M_ISOLATE,
                    sides(lambda t: a * t - b, lambda t: a * x - b + 0 * t, (x - 6, x + 6)))


def l1_word(r):
    x, a, b = r.randint(2, 15), r.randint(2, 6), r.randint(1, 20)
    return Exercise(f"Je pense à un nombre. Je le multiplie par {a}, puis j'ajoute {b}. "
                    f"J'obtiens {a * x + b}. Quel est ce nombre ?", Number(x),
                    "Appelle le nombre x et écris la phrase sous forme d'équation, puis isole x.",
                    number_line(0, 16, [(0, "?")]))


def l1_two_consts(r):
    x, a, b, c = r.randint(-10, 10), r.randint(1, 9), r.randint(5, 20), r.randint(1, 9)
    return Exercise(f"Résous : x + {a} = {b} − {c} + {x + a - b + c}".replace("+ -", "− "), Number(x),
                    "Simplifie d'abord le membre de droite (calcule-le), puis isole x.",
                    sides(lambda t: t + a, lambda t: (x + a) + 0 * t, (x - 8, x + 8)))


def l1_negcoef(r):
    x, a = r.randint(-9, 9), r.randint(2, 9)
    return Exercise(f"Résous : −{a}x = {-a * x}", Number(x),
                    "Divise les deux côtés par le coefficient de x, signe compris. "
                    "Rappel : négatif ÷ négatif = positif.",
                    sides(lambda t: -a * t, lambda t: -a * x + 0 * t, (x - 6, x + 6)))


# ================================================================ NIVEAU 2

M_BOTH = ("Regroupe tous les termes en x d'un côté et les nombres de l'autre "
          "(en faisant la même opération des deux côtés), puis divise par le coefficient de x.")


def l2_both_sides(r):
    while True:
        a, c = r.randint(-7, 9), r.randint(-7, 9)
        if a != c:
            break
    b, d = r.randint(-12, 12), r.randint(-12, 12)
    x = F(d - b, a - c)
    return Exercise(f"Résous : {lin(a, b)} = {lin(c, d)}", Number(x), M_BOTH + "\nLa réponse peut être une fraction.",
                    sides(lambda t: a * t + b, lambda t: c * t + d, (float(x) - 6, float(x) + 6)))


def l2_paren(r):
    x, a, b = r.randint(-8, 8), r.randint(2, 6), nz(r, -9, 9)
    return Exercise(f"Résous : {a}(x {signed(b)}) = {a * (x + b)}", Number(x),
                    "Deux façons : développe la parenthèse (distributivité : a(x+b) = ax + ab), "
                    "ou divise d'abord les deux côtés par le nombre devant la parenthèse.",
                    sides(lambda t: a * (t + b), lambda t: a * (x + b) + 0 * t, (x - 6, x + 6)))


def l2_paren_both(r):
    while True:
        a, c = r.randint(2, 7), r.randint(2, 7)
        if a != c:
            break
    b, d = nz(r, -8, 8), nz(r, -8, 8)
    x = F(c * d - a * b, a - c)
    return Exercise(f"Résous : {a}(x {signed(b)}) = {c}(x {signed(d)})", Number(x),
                    "Développe les deux parenthèses, puis : " + M_BOTH,
                    sides(lambda t: a * (t + b), lambda t: c * (t + d), (float(x) - 6, float(x) + 6)))


def l2_frac_plus(r):
    a, b, y = r.randint(2, 6), nz(r, -9, 9), r.randint(-6, 6)
    c = y + b
    return Exercise(f"Résous : x/{a} {signed(b)} = {c}", Number(a * y),
                    "Isole d'abord x/" + str(a) + " (enlève la constante), puis multiplie par " + str(a) + ".",
                    sides(lambda t: t / a + b, lambda t: c + 0 * t, (a * y - 15, a * y + 15)))


def l2_frac_num(r):
    a, b, c = nz(r, -9, 9), r.randint(2, 6), r.randint(-6, 6)
    x = b * c - a
    return Exercise(f"Résous : (x {signed(a)}) / {b} = {c}", Number(x),
                    "Multiplie les deux côtés par le dénominateur pour faire disparaître la fraction, puis isole x.",
                    sides(lambda t: (t + a) / b, lambda t: c + 0 * t, (x - 15, x + 15)))


def l2_frac_result(r):
    a, b = r.randint(2, 9), nz(r, -10, 10)
    while True:
        c = r.randint(-15, 15)
        if (c - b) % a:
            break
    x = F(c - b, a)
    return Exercise(f"Résous : {lin(a, b)} = {c}\n(la solution n'est pas entière : donne une fraction)",
                    Number(x), M_ISOLATE, sides(lambda t: a * t + b, lambda t: c + 0 * t, (-6, 6)))


def l2_mixed(r):
    a, b, c, d = r.randint(1, 9), r.randint(2, 6), r.randint(1, 6), r.randint(-10, 10)
    x = F(a - d, b + c)
    return Exercise(f"Résous : {a} − {b}x = {lin(c, d)}", Number(x), M_BOTH,
                    sides(lambda t: a - b * t, lambda t: c * t + d, (float(x) - 6, float(x) + 6)))


def l2_two_fracs(r):
    while True:
        a, b = r.sample([2, 3, 4, 5, 6], 2)
        c = nz(r, -6, 6)
        break
    x = F(c * a * b, b - a)
    return Exercise(f"Résous : x/{a} = x/{b} {signed(c)}", Number(x),
                    "Multiplie tout par un multiple commun des dénominateurs (par exemple leur produit) "
                    "pour supprimer les fractions, puis : " + M_BOTH,
                    sides(lambda t: t / a, lambda t: t / b + c, (float(x) - 20, float(x) + 20)))


def l2_rect(r):
    w, k = r.randint(2, 15), r.randint(1, 8)
    P = 2 * (w + w + k)
    return Exercise(f"Un rectangle a une longueur qui dépasse sa largeur de {k} cm. "
                    f"Son périmètre vaut {P} cm. Quelle est sa largeur (en cm) ?", Number(w),
                    "Appelle la largeur x. Exprime la longueur en fonction de x. "
                    "Périmètre = 2 × (longueur + largeur). Écris l'équation et résous.",
                    rectangle(f"longueur = x + {k}", "largeur = x"))


def l2_ages(r):
    b, d = r.randint(5, 30), r.randint(2, 25)
    return Exercise(f"Alice a {d} ans de plus que Bob. La somme de leurs âges vaut {2 * b + d}. Quel âge a Bob ?",
                    Number(b),
                    "Appelle l'âge de Bob x. L'âge d'Alice s'écrit en fonction de x. La somme donne l'équation.",
                    number_line(0, 10))


# ================================================================ NIVEAU 3

def l3_product(r):
    a, b = r.sample(range(-9, 10), 2)
    return Exercise(f"Résous : (x {signed(-a)})(x {signed(-b)}) = 0", Numbers([a, b]),
                    "Un produit est nul si et seulement si l'un des facteurs est nul.\n"
                    "Écris « premier facteur = 0 » OU « second facteur = 0 » et résous chacune.",
                    curves(lambda t: (t - a) * (t - b), xlim=(min(a, b) - 3, max(a, b) + 3), ylim=(-25, 25)))


def l3_square(r):
    k = r.randint(1, 12)
    return Exercise(f"Résous : x² = {k * k}", Numbers([k, -k]),
                    "Attention : un carré « perd » le signe. Si x² = A (A > 0), il y a DEUX solutions : √A et −√A.",
                    curves(lambda t: t ** 2, lambda t: k * k + 0 * t, xlim=(-k - 3, k + 3),
                           labels=["x²", str(k * k)]))


def l3_abs(r):
    a, b = r.randint(-8, 8), r.randint(1, 9)
    return Exercise(f"Résous : |x {signed(-a)}| = {b}", Numbers([a + b, a - b]),
                    "|A| = b signifie que A vaut b OU −b (la distance à 0 vaut b).\n"
                    "Écris les deux équations sans valeur absolue et résous chacune.",
                    curves(lambda t: np.abs(t - a), lambda t: b + 0 * t, xlim=(a - b - 4, a + b + 4),
                           labels=["|x − ...|", str(b)]))


def l3_diff_squares(r):
    k = r.randint(2, 11)
    return Exercise(f"Résous : x² − {k * k} = 0", Numbers([k, -k]),
                    "Identité remarquable : A² − B² = (A − B)(A + B). Puis : produit nul.",
                    curves(lambda t: t ** 2 - k * k, xlim=(-k - 3, k + 3)))


def l3_factor_x(r):
    a, b = r.randint(1, 6), nz(r, -12, 12)
    return Exercise(f"Résous : x({lin(a, b)}) = 0", Numbers([0, F(-b, a)]),
                    "C'est un produit de deux facteurs (x et la parenthèse). Produit nul : l'un des deux vaut 0.",
                    curves(lambda t: t * (a * t + b), xlim=(-8, 8), ylim=(-30, 30)))


def l3_ineq_min(r):
    a, b = r.randint(2, 7), r.randint(-10, 10)
    c = r.randint(-15, 25)
    # ax + b > c  <=>  x > (c-b)/a  -> plus petit entier strictement supérieur
    sol = math.floor(F(c - b, a)) + 1
    return Exercise(f"Résous l'inéquation {lin(a, b)} > {c}.\n"
                    "Donne la PLUS PETITE valeur ENTIÈRE de x qui la vérifie.", Number(sol),
                    "Une inéquation se résout comme une équation : mêmes opérations des deux côtés.\n"
                    "Seule règle en plus : multiplier ou diviser par un nombre NÉGATIF inverse le sens (> devient <).\n"
                    "Trouve la borne, puis cherche le plus petit entier qui respecte le sens.",
                    sides(lambda t: a * t + b, lambda t: c + 0 * t, (-10, 10)))


def l3_ineq_max(r):
    a = r.randint(-6, -2)
    b, c = r.randint(-10, 10), r.randint(-15, 15)
    # ax + b >= c  with a<0 <=> x <= (c-b)/a
    sol = math.floor(F(c - b, a))
    return Exercise(f"Résous l'inéquation {lin(a, b)} ≥ {c}.\n"
                    "Donne la PLUS GRANDE valeur ENTIÈRE de x qui la vérifie.", Number(sol),
                    "Isole x. Ici le coefficient de x est négatif : en divisant par lui, le sens de "
                    "l'inégalité s'inverse (≥ devient ≤). Puis cherche le plus grand entier possible.",
                    sides(lambda t: a * t + b, lambda t: c + 0 * t, (-10, 10)))


def l3_sqrt_sol(r):
    k = r.choice([2, 3, 5, 6, 7, 8, 10, 11, 12, 13])
    return Exercise(f"Résous : x² = {k}\n(écris par ex. sqrt({k}) et −sqrt({k}), ou des valeurs décimales précises)",
                    Numbers([math.sqrt(k), -math.sqrt(k)]),
                    "Même idée que x² = 9, mais la racine n'est pas entière. Deux solutions : ±√" + str(k) + ".",
                    curves(lambda t: t ** 2, lambda t: k + 0 * t, xlim=(-5, 5), labels=["x²", str(k)]))


def l3_shifted_square(r):
    a, b = r.randint(-6, 6), r.randint(1, 7)
    return Exercise(f"Résous : (x {signed(-a)})² = {b * b}", Numbers([a + b, a - b]),
                    "Comme pour x² = A : la parenthèse vaut +√A ou −√A. Écris les deux cas et isole x.",
                    curves(lambda t: (t - a) ** 2, lambda t: b * b + 0 * t, xlim=(a - b - 3, a + b + 3)))


def l3_abs_lin(r):
    a, b, c = r.randint(2, 5), nz(r, -8, 8), r.randint(1, 10)
    return Exercise(f"Résous : |{lin(a, b)}| = {c}", Numbers([F(c - b, a), F(-c - b, a)]),
                    "|A| = c donne deux équations : A = c et A = −c. Résous chacune (fractions possibles).",
                    curves(lambda t: np.abs(a * t + b), lambda t: c + 0 * t, xlim=(-10, 10), ylim=(-2, 20)))


# ================================================================ NIVEAU 4

M_QUAD = ("Pour ax² + bx + c = 0 :\n"
          "1. Calcule le discriminant Δ = b² − 4ac.\n"
          "2. Si Δ > 0 : deux solutions x = (−b − √Δ) / (2a) et x = (−b + √Δ) / (2a).\n"
          "   Si Δ = 0 : une solution x = −b / (2a).   Si Δ < 0 : aucune solution réelle.\n"
          "Astuce si a = 1 : cherche deux nombres de somme −b et de produit c.")


def _quad_draw(a, b, c, x1=None, x2=None):
    xs = [v for v in (x1, x2) if v is not None]
    cx = -b / (2 * a)
    lo, hi = (min(xs + [cx]) - 3, max(xs + [cx]) + 3) if xs else (cx - 5, cx + 5)
    return curves(lambda t: a * t ** 2 + b * t + c, xlim=(lo, hi))


def l4_monic(r):
    p, q = r.randint(-8, 8), r.randint(-8, 8)
    b, c = -(p + q), p * q
    return Exercise(f"Résous : {poly({2: 1, 1: b, 0: c})} = 0", Numbers(sorted({p, q})), M_QUAD,
                    _quad_draw(1, b, c, p, q))


def l4_general(r):
    a = r.randint(2, 4)
    p, q = r.randint(-5, 5), r.randint(-5, 5)
    b, c = -a * (p + q), a * p * q
    return Exercise(f"Résous : {poly({2: a, 1: b, 0: c})} = 0", Numbers(sorted({p, q})),
                    M_QUAD + "\nAstuce : tu peux d'abord diviser toute l'équation par " + str(a) + ".",
                    _quad_draw(a, b, c, p, q))


def l4_delta(r):
    a, b, c = nz(r, -5, 5), r.randint(-10, 10), r.randint(-10, 10)
    return Exercise(f"Calcule le discriminant Δ de {poly({2: a, 1: b, 0: c})}.", Number(b * b - 4 * a * c),
                    "Δ = b² − 4ac. Repère bien a, b, c avec leurs signes. Attention : b² est toujours positif.",
                    _quad_draw(a, b, c))


def l4_count(r):
    a, b, c = nz(r, -4, 4), r.randint(-8, 8), r.randint(-8, 8)
    d = b * b - 4 * a * c
    n = 2 if d > 0 else (1 if d == 0 else 0)
    return Exercise(f"Combien de solutions réelles a l'équation {poly({2: a, 1: b, 0: c})} = 0 ?",
                    Number(n),
                    "Calcule Δ = b² − 4ac et regarde son signe : Δ > 0 → 2, Δ = 0 → 1, Δ < 0 → 0.\n"
                    "Sur le graphe : c'est le nombre de fois où la parabole coupe l'axe horizontal.",
                    None)


def l4_irrational(r):
    while True:
        b, c = r.randint(-6, 6), r.randint(-8, 4)
        d = b * b - 4 * c
        if d > 0 and math.isqrt(d) ** 2 != d:
            break
    s = math.sqrt(d)
    return Exercise(f"Résous : {poly({2: 1, 1: b, 0: c})} = 0\n(les solutions contiennent une racine carrée)",
                    Numbers([(-b - s) / 2, (-b + s) / 2]),
                    M_QUAD + "\nTu peux écrire la réponse avec sqrt(...), par ex. (−3 + sqrt(5))/2.",
                    _quad_draw(1, b, c, (-b - s) / 2, (-b + s) / 2))


def l4_none(r):
    h, k = r.randint(-4, 4), r.randint(1, 6)
    b, c = -2 * h, h * h + k
    return Exercise(f"Résous : {poly({2: 1, 1: b, 0: c})} = 0\n(écris « aucune » s'il n'y a pas de solution réelle)",
                    Numbers([]), M_QUAD, _quad_draw(1, b, c))


def l4_double(r):
    h = r.randint(-7, 7)
    return Exercise(f"Résous : {poly({2: 1, 1: -2 * h, 0: h * h})} = 0", Numbers([h]),
                    M_QUAD + "\nRemarque : regarde si tu reconnais une identité remarquable (a − b)² = a² − 2ab + b².",
                    _quad_draw(1, -2 * h, h * h, h))


def l4_vertex(r):
    a = nz(r, -3, 3)
    h = r.randint(-6, 6)
    b, c = -2 * a * h, r.randint(-8, 8)
    return Exercise(f"Quelle est l'abscisse du sommet de la parabole y = {poly({2: a, 1: b, 0: c})} ?",
                    Number(h),
                    "Le sommet est à x = −b / (2a). Une parabole est symétrique : son sommet est au milieu "
                    "de ses deux racines, quand elles existent.",
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(h - 5, h + 5)))


def l4_sum_roots(r):
    a = r.randint(1, 5)
    b, c = r.randint(-15, 15), r.randint(-20, -1)
    return Exercise(f"Sans calculer les solutions, donne leur SOMME pour {poly({2: a, 1: b, 0: c})} = 0.",
                    Number(F(-b, a)),
                    "Pour ax² + bx + c = 0 : somme des solutions = −b/a, produit = c/a.\n"
                    "(Tu peux le retrouver en additionnant les deux formules avec ±√Δ : les √Δ s'annulent.)",
                    _quad_draw(a, b, c))


def l4_area(r):
    w, k = r.randint(2, 12), r.randint(1, 6)
    A = w * (w + k)
    return Exercise(f"Un rectangle a une longueur qui dépasse sa largeur de {k} m et une aire de {A} m². "
                    "Quelle est sa largeur ?", Number(w),
                    "Largeur = x, longueur = x + " + str(k) + ". Aire = largeur × longueur donne une équation "
                    "du second degré. Garde la solution qui a du sens (une longueur est positive).",
                    rectangle(f"x + {k}", "x"))


# ================================================================ NIVEAU 5

def _lines(a1, b1, a2, b2, x0):
    a1, b1, a2, b2, x0 = map(float, (a1, b1, a2, b2, x0))
    return curves(lambda t: a1 * t + b1, lambda t: a2 * t + b2, xlim=(x0 - 6, x0 + 6),
                  labels=["droite 1", "droite 2"])


def l5_subst(r):
    x, y = r.randint(-6, 6), r.randint(-6, 6)
    a = nz(r, -4, 4)
    b = y - a * x
    c, d = nz(r, -5, 5), nz(r, -5, 5)
    e = c * x + d * y
    return Exercise(f"Résous le système :\n  y = {lin(a, b)}\n  {term(c, 'x')} {sterm(d, 'y')} = {e}",
                    Numbers([x, y], ordered=True),
                    "Substitution : remplace y dans la deuxième équation par l'expression de la première.\n"
                    "Tu obtiens une équation en x seul. Résous-la, puis calcule y.",
                    _lines(a, b, F(-c, d), F(e, d), x))


def l5_general(r):
    while True:
        x, y = r.randint(-6, 6), r.randint(-6, 6)
        a, b, c, d = (nz(r, -5, 5) for _ in range(4))
        if a * d - b * c != 0:
            break
    e, f = a * x + b * y, c * x + d * y
    return Exercise(f"Résous le système :\n  {term(a, 'x')} {sterm(b, 'y')} = {e}\n  {term(c, 'x')} {sterm(d, 'y')} = {f}",
                    Numbers([x, y], ordered=True),
                    "Combinaison : multiplie les équations pour que les coefficients de y (ou de x) soient opposés, "
                    "puis additionne-les : une inconnue disparaît. Résous, puis remplace pour trouver l'autre.\n"
                    "Réponse au format : x, y",
                    _lines(F(-a, b), F(e, b), F(-c, d), F(f, d), x))


def l5_fraction(r):
    while True:
        a, b, c, d = (nz(r, -5, 5) for _ in range(4))
        det = a * d - b * c
        if det not in (0, 1, -1):
            break
    e, f = r.randint(-9, 9), r.randint(-9, 9)
    x, y = F(e * d - b * f, det), F(a * f - c * e, det)
    return Exercise(f"Résous le système (solutions éventuellement fractionnaires) :\n  {term(a, 'x')} {sterm(b, 'y')} = {e}\n  {term(c, 'x')} {sterm(d, 'y')} = {f}",
                    Numbers([x, y], ordered=True),
                    "Même méthode (substitution ou combinaison). Garde les fractions exactes jusqu'au bout.\n"
                    "Réponse au format : x, y",
                    _lines(F(-a, b), F(e, b), F(-c, d), F(f, d), float(x)))


def l5_word(r):
    p1, p2 = r.randint(1, 6), r.randint(2, 9)
    n1, n2, m1, m2 = r.randint(1, 5), r.randint(1, 5), r.randint(1, 5), r.randint(1, 5)
    if n1 * m2 == n2 * m1:
        m2 += 1
    t1, t2 = n1 * p1 + n2 * p2, m1 * p1 + m2 * p2
    return Exercise(f"{n1} cafés et {n2} croissants coûtent {t1} €. {m1} cafés et {m2} croissants coûtent {t2} €.\n"
                    "Quel est le prix d'un café, puis d'un croissant ?",
                    Numbers([p1, p2], ordered=True),
                    "Appelle x le prix d'un café et y celui d'un croissant. Chaque phrase donne une équation. "
                    "Résous le système. Réponse : café, croissant",
                    None)


def l5_triangular(r):
    x, y, z = (r.randint(-5, 5) for _ in range(3))
    a, b, _c = nz(r, -3, 3), nz(r, -3, 3), nz(r, -3, 3)
    return Exercise(f"Résous le système à 3 inconnues :\n  x + y + z = {x + y + z}\n"
                    f"  y {sterm(a, 'z')} = {y + a * z}\n  {term(b, 'z')} = {b * z}",
                    Numbers([x, y, z], ordered=True),
                    "Le système est « en escalier » : commence par la dernière équation (une seule inconnue), "
                    "remonte en remplaçant à chaque ligne. Réponse : x, y, z",
                    None)


def l5_intersection(r):
    x = r.randint(-5, 5)
    a1, a2 = r.sample(range(-4, 5), 2)
    b1 = r.randint(-6, 6)
    y = a1 * x + b1
    b2 = y - a2 * x
    return Exercise(f"Donne les coordonnées du point d'intersection des droites\n"
                    f"  y = {lin(a1, b1)}   et   y = {lin(a2, b2)}", Numbers([x, y], ordered=True),
                    "Au point d'intersection, les deux y sont égaux : écris « expression 1 = expression 2 », "
                    "résous en x, puis calcule y. Réponse : x, y",
                    _lines(a1, b1, a2, b2, x))


def l5_biquad(r):
    p, q = r.sample(range(1, 6), 2)
    s, pr = p * p + q * q, p * p * q * q
    return Exercise(f"Résous : x⁴ − {s}x² + {pr} = 0", Numbers([p, -p, q, -q]),
                    "Pose X = x². L'équation devient X² − " + str(s) + "X + " + str(pr) + " = 0 (second degré). "
                    "Trouve les valeurs de X, puis reviens à x avec x² = X (attention aux ±).",
                    curves(lambda t: t ** 4 - s * t ** 2 + pr, xlim=(-max(p, q) - 1.5, max(p, q) + 1.5), ylim=(-pr - 5, pr + 20)))


def l5_rational(r):
    while True:
        a, b, c = r.randint(-6, 6), r.randint(-6, 6), r.choice([-3, -2, 2, 3, 4])
        if a != b:
            break
    x = F(c * b - a, 1 - c)
    return Exercise(f"Résous : (x {signed(a)}) / (x {signed(b)}) = {c}", Number(x),
                    "Multiplie les deux côtés par le dénominateur (en notant que x ne peut pas l'annuler), "
                    "développe, puis résous l'équation du premier degré. Vérifie que la solution n'annule pas le dénominateur.",
                    curves(lambda t: (t + a) / (t + b), lambda t: c + 0 * t, xlim=(-15, 15), ylim=(-10, 10)))


def l5_sqrt(r):
    a, b = r.randint(-8, 8), r.randint(1, 7)
    return Exercise(f"Résous : √(x {signed(a)}) = {b}", Number(b * b - a),
                    "Élève les deux côtés au carré pour faire disparaître la racine, puis isole x. "
                    "Vérifie ensuite dans l'équation de départ (une racine est toujours ≥ 0).",
                    curves(lambda t: np.sqrt(t + a), lambda t: b + 0 * t, xlim=(-a - 1, b * b - a + 8)))


def l5_parallel(r):
    a = nz(r, -4, 4)
    b1, b2 = r.sample(range(-8, 9), 2)
    k = r.randint(2, 3)
    return Exercise(f"Résous le système :\n  y = {lin(a, b1)}\n  {k}y = {lin(k * a, k * b2)}\n"
                    "(écris « aucune » s'il n'y a pas de solution)", Numbers([]),
                    "Mets les deux équations sous la forme y = ... et compare les pentes et les ordonnées à l'origine. "
                    "Deux droites parallèles distinctes ne se coupent jamais.",
                    _lines(a, b1, a, b2, 0))


LEVELS = {
    1: [l1_add, l1_sub, l1_mul, l1_div, l1_affine, l1_minus_x, l1_affine_neg, l1_word, l1_two_consts, l1_negcoef],
    2: [l2_both_sides, l2_paren, l2_paren_both, l2_frac_plus, l2_frac_num, l2_frac_result, l2_mixed,
        l2_two_fracs, l2_rect, l2_ages],
    3: [l3_product, l3_square, l3_abs, l3_diff_squares, l3_factor_x, l3_ineq_min, l3_ineq_max, l3_sqrt_sol,
        l3_shifted_square, l3_abs_lin],
    4: [l4_monic, l4_general, l4_delta, l4_count, l4_irrational, l4_none, l4_double, l4_vertex, l4_sum_roots, l4_area],
    5: [l5_subst, l5_general, l5_fraction, l5_word, l5_triangular, l5_intersection, l5_biquad, l5_rational,
        l5_sqrt, l5_parallel],
}
