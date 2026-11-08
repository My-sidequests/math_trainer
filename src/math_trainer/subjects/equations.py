"""Équations : du premier degré aux systèmes."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Number, Numbers
from ..core import Exercise, curves, fmt, lin, nz, number_line, par, poly, rectangle, signed, sterm, term

from ..core import fmt, lines, par, steps_lin, term  # noqa: E402,F811

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
    _sol = lines(f"x + {a} = {x + a}", steps_lin(1, a, x + a), f"Vérification : {fmt(x)} + {a} = {x + a} ✔")
    ex = Exercise(f"Résous : x + {a} = {x + a}", Number(x), M_ISOLATE,
                    sides(lambda t: t + a, lambda t: (x + a) + 0 * t, (x - 8, x + 8)))
    ex.solution = _sol
    return ex


def l1_sub(r):
    x, a = r.randint(-12, 12), r.randint(1, 15)
    _sol = lines(f"x − {a} = {x - a}", steps_lin(1, -a, x - a), f"Vérification : {fmt(x)} − {a} = {fmt(x - a)} ✔")
    ex = Exercise(f"Résous : x − {a} = {x - a}", Number(x), M_ISOLATE,
                    sides(lambda t: t - a, lambda t: (x - a) + 0 * t, (x - 8, x + 8)))
    ex.solution = _sol
    return ex


def l1_mul(r):
    x, a = r.randint(-10, 10), r.randint(2, 9)
    _sol = lines(f"{a}x = {fmt(a * x)}", steps_lin(a, 0, a * x), f"Vérification : {a} × {par(x)} = {fmt(a * x)} ✔")
    ex = Exercise(f"Résous : {a}x = {a * x}", Number(x), M_ISOLATE,
                    sides(lambda t: a * t, lambda t: a * x + 0 * t, (x - 6, x + 6)))
    ex.solution = _sol
    return ex


def l1_div(r):
    b, a = r.randint(-9, 9), r.randint(2, 8)
    _sol = lines(f"x / {a} = {fmt(b)}", f"On multiplie les deux côtés par {a} : x = {fmt(b)} × {a} = {fmt(a * b)}",
                 f"Vérification : {fmt(a * b)} / {a} = {fmt(b)} ✔")
    ex = Exercise(f"Résous : x / {a} = {b}", Number(a * b),
                    "Diviser par un nombre se « défait » en multipliant les deux côtés par ce nombre.",
                    sides(lambda t: t / a, lambda t: b + 0 * t, (a * b - 15, a * b + 15)))
    ex.solution = _sol
    return ex


def l1_affine(r):
    x, a, b = r.randint(-8, 8), r.randint(2, 7), nz(r, -12, 12)
    _sol = lines(f"{lin(a, b)} = {fmt(a * x + b)}", steps_lin(a, b, a * x + b),
                 f"Vérification : {a} × {par(x)} {signed(b)} = {fmt(a * x + b)} ✔")
    ex = Exercise(f"Résous : {lin(a, b)} = {a * x + b}", Number(x),
                    M_ISOLATE + "\nCommence par le terme constant, puis le coefficient de x.",
                    sides(lambda t: a * t + b, lambda t: a * x + b + 0 * t, (x - 6, x + 6)))
    ex.solution = _sol
    return ex


def l1_minus_x(r):
    x, a = r.randint(-10, 10), r.randint(1, 15)
    _sol = lines(f"{a} − x = {fmt(a - x)}", f"On ajoute x des deux côtés : {a} = {fmt(a - x)} + x",
                 f"On soustrait {fmt(a - x)} des deux côtés : x = {a} − {par(a - x)} = {fmt(x)}",
                 f"Vérification : {a} − {par(x)} = {fmt(a - x)} ✔")
    ex = Exercise(f"Résous : {a} − x = {a - x}", Number(x),
                    "« −x » : ajoute x des deux côtés pour le rendre positif, puis isole-le.",
                    sides(lambda t: a - t, lambda t: (a - x) + 0 * t, (x - 8, x + 8)))
    ex.solution = _sol
    return ex


def l1_affine_neg(r):
    x, a, b = r.randint(-8, 8), r.randint(2, 7), r.randint(1, 12)
    _sol = lines(f"{a}x − {b} = {fmt(a * x - b)}", steps_lin(a, -b, a * x - b),
                 f"Vérification : {a} × {par(x)} − {b} = {fmt(a * x - b)} ✔")
    ex = Exercise(f"Résous : {a}x − {b} = {a * x - b}", Number(x), M_ISOLATE,
                    sides(lambda t: a * t - b, lambda t: a * x - b + 0 * t, (x - 6, x + 6)))
    ex.solution = _sol
    return ex


def l1_word(r):
    x, a, b = r.randint(2, 15), r.randint(2, 6), r.randint(1, 20)
    _sol = lines(f"On appelle le nombre x. La phrase s'écrit : {a}x + {b} = {a * x + b}", steps_lin(a, b, a * x + b),
                 f"Vérification : {x} × {a} = {a * x}, puis + {b} = {a * x + b} ✔")
    ex = Exercise(f"Je pense à un nombre. Je le multiplie par {a}, puis j'ajoute {b}. "
                    f"J'obtiens {a * x + b}. Quel est ce nombre ?", Number(x),
                    "Appelle le nombre x et écris la phrase sous forme d'équation, puis isole x.",
                    number_line(0, 16, [(0, "?")]))
    ex.solution = _sol
    return ex


def l1_two_consts(r):
    x, a, b, c = r.randint(-10, 10), r.randint(1, 9), r.randint(5, 20), r.randint(1, 9)
    _k = x + a - b + c
    _sol = lines(f"On calcule le membre de droite : {b} − {c} {signed(_k)} = {fmt(x + a)}",
                 f"L'équation devient x + {a} = {fmt(x + a)}", steps_lin(1, a, x + a),
                 f"Vérification : {fmt(x)} + {a} = {fmt(x + a)} ✔")
    ex = Exercise(f"Résous : x + {a} = {b} − {c} + {x + a - b + c}".replace("+ -", "− "), Number(x),
                    "Simplifie d'abord le membre de droite (calcule-le), puis isole x.",
                    sides(lambda t: t + a, lambda t: (x + a) + 0 * t, (x - 8, x + 8)))
    ex.solution = _sol
    return ex


def l1_negcoef(r):
    x, a = r.randint(-9, 9), r.randint(2, 9)
    _sol = lines(f"−{a}x = {fmt(-a * x)}", f"On divise les deux côtés par (−{a}) : x = {fmt(-a * x)} / (−{a}) = {fmt(x)}",
                 f"Vérification : −{a} × {par(x)} = {fmt(-a * x)} ✔")
    ex = Exercise(f"Résous : −{a}x = {-a * x}", Number(x),
                    "Divise les deux côtés par le coefficient de x, signe compris. "
                    "Rappel : négatif ÷ négatif = positif.",
                    sides(lambda t: -a * t, lambda t: -a * x + 0 * t, (x - 6, x + 6)))
    ex.solution = _sol
    return ex


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
    _sol = lines(f"{lin(a, b)} = {lin(c, d)}",
                 f"On regroupe les x à gauche (on retire {term(c, 'x') if c else '0'} des deux côtés) et les nombres à droite "
                 f"(on retire {fmt(b)}) : {term(a, 'x')} − {par(c)}x = {fmt(d)} − {par(b)}",
                 f"Donc {term(a - c, 'x')} = {fmt(d - b)}",
                 f"On divise par {par(a - c)} : x = {fmt(d - b)} / {par(a - c)} = {fmt(x)}",
                 f"Vérification : les deux membres valent {fmt(a * x + b)} ✔")
    ex = Exercise(f"Résous : {lin(a, b)} = {lin(c, d)}", Number(x), M_BOTH + "\nLa réponse peut être une fraction.",
                    sides(lambda t: a * t + b, lambda t: c * t + d, (float(x) - 6, float(x) + 6)))
    ex.solution = _sol
    return ex


def l2_paren(r):
    x, a, b = r.randint(-8, 8), r.randint(2, 6), nz(r, -9, 9)
    _sol = lines(f"{a}(x {signed(b)}) = {a * (x + b)}",
                 f"On divise les deux côtés par {a} : x {signed(b)} = {fmt(a * (x + b))} / {a} = {fmt(x + b)}",
                 steps_lin(1, b, x + b), f"Vérification : {a} × ({fmt(x)} {signed(b)}) = {a} × {par(x + b)} = {a * (x + b)} ✔")
    ex = Exercise(f"Résous : {a}(x {signed(b)}) = {a * (x + b)}", Number(x),
                    "Deux façons : développe la parenthèse (distributivité : a(x+b) = ax + ab), "
                    "ou divise d'abord les deux côtés par le nombre devant la parenthèse.",
                    sides(lambda t: a * (t + b), lambda t: a * (x + b) + 0 * t, (x - 6, x + 6)))
    ex.solution = _sol
    return ex


def l2_paren_both(r):
    while True:
        a, c = r.randint(2, 7), r.randint(2, 7)
        if a != c:
            break
    b, d = nz(r, -8, 8), nz(r, -8, 8)
    x = F(c * d - a * b, a - c)
    _sol = lines(f"On développe : {lin(a, a * b)} = {lin(c, c * d)}",
                 f"On regroupe : {term(a, 'x')} − {term(c, 'x')} = {fmt(c * d)} − {par(a * b)}",
                 f"Donc {term(a - c, 'x')} = {fmt(c * d - a * b)}",
                 f"On divise par {par(a - c)} : x = {fmt(x)}",
                 f"Vérification : les deux membres valent {fmt(a * (x + b))} ✔")
    ex = Exercise(f"Résous : {a}(x {signed(b)}) = {c}(x {signed(d)})", Number(x),
                    "Développe les deux parenthèses, puis : " + M_BOTH,
                    sides(lambda t: a * (t + b), lambda t: c * (t + d), (float(x) - 6, float(x) + 6)))
    ex.solution = _sol
    return ex


def l2_frac_plus(r):
    a, b, y = r.randint(2, 6), nz(r, -9, 9), r.randint(-6, 6)
    c = y + b
    _sol = lines(f"x/{a} {signed(b)} = {c}",
                 f"On {'soustrait' if b > 0 else 'ajoute'} {abs(b)} des deux côtés : x/{a} = {fmt(c - b)}",
                 f"On multiplie par {a} : x = {fmt(c - b)} × {a} = {fmt(a * y)}",
                 f"Vérification : {fmt(a * y)}/{a} {signed(b)} = {fmt(y)} {signed(b)} = {c} ✔")
    ex = Exercise(f"Résous : x/{a} {signed(b)} = {c}", Number(a * y),
                    "Isole d'abord x/" + str(a) + " (enlève la constante), puis multiplie par " + str(a) + ".",
                    sides(lambda t: t / a + b, lambda t: c + 0 * t, (a * y - 15, a * y + 15)))
    ex.solution = _sol
    return ex


def l2_frac_num(r):
    a, b, c = nz(r, -9, 9), r.randint(2, 6), r.randint(-6, 6)
    x = b * c - a
    _sol = lines(f"(x {signed(a)}) / {b} = {c}", f"On multiplie les deux côtés par {b} : x {signed(a)} = {fmt(b * c)}",
                 steps_lin(1, a, b * c), f"Vérification : ({fmt(x)} {signed(a)}) / {b} = {fmt(b * c)} / {b} = {c} ✔")
    ex = Exercise(f"Résous : (x {signed(a)}) / {b} = {c}", Number(x),
                    "Multiplie les deux côtés par le dénominateur pour faire disparaître la fraction, puis isole x.",
                    sides(lambda t: (t + a) / b, lambda t: c + 0 * t, (x - 15, x + 15)))
    ex.solution = _sol
    return ex


def l2_frac_result(r):
    a, b = r.randint(2, 9), nz(r, -10, 10)
    while True:
        c = r.randint(-15, 15)
        if (c - b) % a:
            break
    x = F(c - b, a)
    _sol = lines(f"{lin(a, b)} = {c}", steps_lin(a, b, c), "La fraction est irréductible : c'est la réponse exacte.")
    ex = Exercise(f"Résous : {lin(a, b)} = {c}\n(la solution n'est pas entière : donne une fraction)",
                    Number(x), M_ISOLATE, sides(lambda t: a * t + b, lambda t: c + 0 * t, (-6, 6)))
    ex.solution = _sol
    return ex


def l2_mixed(r):
    a, b, c, d = r.randint(1, 9), r.randint(2, 6), r.randint(1, 6), r.randint(-10, 10)
    x = F(a - d, b + c)
    _sol = lines(f"{a} − {b}x = {lin(c, d)}",
                 f"On ajoute {b}x des deux côtés : {a} = {term(b + c, 'x')} {signed(d)}",
                 f"On {'soustrait' if d > 0 else 'ajoute'} {abs(d)} des deux côtés : {fmt(a - d)} = {term(b + c, 'x')}" if d else
                 f"Donc {fmt(a)} = {term(b + c, 'x')}",
                 f"On divise par {b + c} : x = {fmt(a - d)} / {b + c} = {fmt(x)}",
                 f"Vérification : les deux membres valent {fmt(a - b * x)} ✔")
    ex = Exercise(f"Résous : {a} − {b}x = {lin(c, d)}", Number(x), M_BOTH,
                    sides(lambda t: a - b * t, lambda t: c * t + d, (float(x) - 6, float(x) + 6)))
    ex.solution = _sol
    return ex


def l2_two_fracs(r):
    a, b = r.sample([2, 3, 4, 5, 6], 2)
    c = nz(r, -6, 6)
    x = F(c * a * b, b - a)
    m = math.lcm(a, b)
    p, q, k = m // a, m // b, c * m
    left, right = x / a, x / b + c
    xs = fmt(x) if x.denominator == 1 and x >= 0 else f"({fmt(x)})"  # x entre parenthèses s'il est négatif ou fractionnaire
    step4 = (f"4. Le coefficient de x vaut 1 : x est déjà isolé, x = {fmt(x)}" if p - q == 1 else
             f"4. On divise par {par(p - q)} : x = {fmt(k)} / {par(p - q)} = {fmt(x)}")
    method = (
        "Objectif : faire disparaître les fractions, puis isoler x. (Démonstration complète sur cet exercice.)\n"
        f"1. Le plus petit nombre divisible par {a} et par {b} est {m}.\n"
        f"2. On multiplie CHAQUE terme des deux côtés par {m}, y compris le nombre seul ({fmt(c)}) :\n"
        f"     x/{a} × {m} = {term(p, 'x')}      x/{b} × {m} = {term(q, 'x')}      {par(c)} × {m} = {fmt(k)}\n"
        f"   L'équation devient : {term(p, 'x')} = {term(q, 'x')} {signed(k)}\n"
        f"3. On soustrait {term(q, 'x')} des deux côtés : {term(p, 'x')} − {term(q, 'x')} = {fmt(k)}, donc {term(p - q, 'x')} = {fmt(k)}\n"
        f"{step4}\n"
        f"5. Vérification dans l'équation de départ :\n"
        f"     à gauche  x/{a} = {xs}/{a} = {fmt(left)}\n"
        f"     à droite  x/{b} {signed(c)} = {xs}/{b} {signed(c)} = {fmt(right)}\n"
        f"   Les deux côtés valent {fmt(left)} : x = {fmt(x)} est bien la solution."
    )
    return Exercise(f"Résous : x/{a} = x/{b} {signed(c)}", Number(x), method,
                    sides(lambda t: t / a, lambda t: t / b + c, (float(x) - 20, float(x) + 20)))


def l2_rect(r):
    w, k = r.randint(2, 15), r.randint(1, 8)
    P = 2 * (w + w + k)
    _sol = lines(f"Largeur = x, longueur = x + {k}.", f"Périmètre : 2 × (x + x + {k}) = {P}, soit 4x + {2 * k} = {P}",
                 steps_lin(4, 2 * k, P), f"Vérification : largeur {w}, longueur {w + k}, périmètre 2 × ({w} + {w + k}) = {P} ✔")
    ex = Exercise(f"Un rectangle a une longueur qui dépasse sa largeur de {k} cm. "
                    f"Son périmètre vaut {P} cm. Quelle est sa largeur (en cm) ?", Number(w),
                    "Appelle la largeur x. Exprime la longueur en fonction de x. "
                    "Périmètre = 2 × (longueur + largeur). Écris l'équation et résous.",
                    rectangle(f"longueur = x + {k}", "largeur = x"))
    ex.solution = _sol
    return ex


def l2_ages(r):
    b, d = r.randint(5, 30), r.randint(2, 25)
    _sol = lines(f"Âge de Bob = x, âge d'Alice = x + {d}.", f"Somme : x + x + {d} = {2 * b + d}, soit 2x + {d} = {2 * b + d}",
                 steps_lin(2, d, 2 * b + d), f"Vérification : Bob {b} ans, Alice {b + d} ans, total {2 * b + d} ✔")
    ex = Exercise(f"Alice a {d} ans de plus que Bob. La somme de leurs âges vaut {2 * b + d}. Quel âge a Bob ?",
                    Number(b),
                    "Appelle l'âge de Bob x. L'âge d'Alice s'écrit en fonction de x. La somme donne l'équation.",
                    number_line(0, 10))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

def l3_product(r):
    a, b = r.sample(range(-9, 10), 2)
    _sol = lines(f"(x {signed(-a)})(x {signed(-b)}) = 0 : un produit est nul si l'un des facteurs est nul.",
                 f"x {signed(-a)} = 0  ⇒  x = {fmt(a)}", f"x {signed(-b)} = 0  ⇒  x = {fmt(b)}")
    ex = Exercise(f"Résous : (x {signed(-a)})(x {signed(-b)}) = 0", Numbers([a, b]),
                    "Un produit est nul si et seulement si l'un des facteurs est nul.\n"
                    "Écris « premier facteur = 0 » OU « second facteur = 0 » et résous chacune.",
                    curves(lambda t: (t - a) * (t - b), xlim=(min(a, b) - 3, max(a, b) + 3), ylim=(-25, 25)))
    ex.solution = _sol
    return ex


def l3_square(r):
    k = r.randint(1, 12)
    _sol = lines(f"x² = {k * k}. On cherche les nombres dont le carré vaut {k * k}.",
                 f"{k}² = {k * k} et (−{k})² = {k * k} : deux solutions, x = {k} ou x = −{k}")
    ex = Exercise(f"Résous : x² = {k * k}", Numbers([k, -k]),
                    "Attention : un carré « perd » le signe. Si x² = A (A > 0), il y a DEUX solutions : √A et −√A.",
                    curves(lambda t: t ** 2, lambda t: k * k + 0 * t, xlim=(-k - 3, k + 3),
                           labels=["x²", str(k * k)]))
    ex.solution = _sol
    return ex


def l3_abs(r):
    a, b = r.randint(-8, 8), r.randint(1, 9)
    _sol = lines(f"|x {signed(-a)}| = {b} signifie x {signed(-a)} = {b} OU x {signed(-a)} = −{b}",
                 f"Premier cas : x = {b} {signed(a)} = {fmt(a + b)}", f"Second cas : x = −{b} {signed(a)} = {fmt(a - b)}")
    ex = Exercise(f"Résous : |x {signed(-a)}| = {b}", Numbers([a + b, a - b]),
                    "|A| = b signifie que A vaut b OU −b (la distance à 0 vaut b).\n"
                    "Écris les deux équations sans valeur absolue et résous chacune.",
                    curves(lambda t: np.abs(t - a), lambda t: b + 0 * t, xlim=(a - b - 4, a + b + 4),
                           labels=["|x − ...|", str(b)]))
    ex.solution = _sol
    return ex


def l3_diff_squares(r):
    k = r.randint(2, 11)
    _sol = lines(f"x² − {k * k} = x² − {k}² = (x − {k})(x + {k})", "Produit nul : l'un des facteurs est nul.",
                 f"x − {k} = 0 ⇒ x = {k}   ou   x + {k} = 0 ⇒ x = −{k}")
    ex = Exercise(f"Résous : x² − {k * k} = 0", Numbers([k, -k]),
                    "Identité remarquable : A² − B² = (A − B)(A + B). Puis : produit nul.",
                    curves(lambda t: t ** 2 - k * k, xlim=(-k - 3, k + 3)))
    ex.solution = _sol
    return ex


def l3_factor_x(r):
    a, b = r.randint(1, 6), nz(r, -12, 12)
    _sol = lines(f"x·({lin(a, b)}) = 0 : produit nul.", "Premier facteur : x = 0",
                 [f"Second facteur : {lin(a, b)} = 0"] + steps_lin(a, b, 0))
    ex = Exercise(f"Résous : x({lin(a, b)}) = 0", Numbers([0, F(-b, a)]),
                    "C'est un produit de deux facteurs (x et la parenthèse). Produit nul : l'un des deux vaut 0.",
                    curves(lambda t: t * (a * t + b), xlim=(-8, 8), ylim=(-30, 30)))
    ex.solution = _sol
    return ex


def l3_ineq_min(r):
    a, b = r.randint(2, 7), r.randint(-10, 10)
    c = r.randint(-15, 25)
    # ax + b > c  <=>  x > (c-b)/a  -> plus petit entier strictement supérieur
    sol = math.floor(F(c - b, a)) + 1
    _bound = F(c - b, a)
    _sol = lines(f"{lin(a, b)} > {c}",
                 f"On {'soustrait' if b > 0 else 'ajoute'} {abs(b)} : {a}x > {fmt(c - b)}" if b else f"{a}x > {c}",
                 f"On divise par {a} (positif, le sens ne change pas) : x > {fmt(_bound)}",
                 f"Le plus petit entier strictement supérieur à {fmt(_bound)} est {sol}.")
    ex = Exercise(f"Résous l'inéquation {lin(a, b)} > {c}.\n"
                    "Donne la PLUS PETITE valeur ENTIÈRE de x qui la vérifie.", Number(sol),
                    "Une inéquation se résout comme une équation : mêmes opérations des deux côtés.\n"
                    "Seule règle en plus : multiplier ou diviser par un nombre NÉGATIF inverse le sens (> devient <).\n"
                    "Trouve la borne, puis cherche le plus petit entier qui respecte le sens.",
                    sides(lambda t: a * t + b, lambda t: c + 0 * t, (-10, 10)))
    ex.solution = _sol
    return ex


def l3_ineq_max(r):
    a = r.randint(-6, -2)
    b, c = r.randint(-10, 10), r.randint(-15, 15)
    # ax + b >= c  with a<0 <=> x <= (c-b)/a
    sol = math.floor(F(c - b, a))
    _bound = F(c - b, a)
    _sol = lines(f"{lin(a, b)} ≥ {c}",
                 f"On {'soustrait' if b > 0 else 'ajoute'} {abs(b)} : {term(a, 'x')} ≥ {fmt(c - b)}" if b else f"{term(a, 'x')} ≥ {c}",
                 f"On divise par {par(a)} (NÉGATIF : le sens s'inverse) : x ≤ {fmt(_bound)}",
                 f"Le plus grand entier inférieur ou égal à {fmt(_bound)} est {sol}.")
    ex = Exercise(f"Résous l'inéquation {lin(a, b)} ≥ {c}.\n"
                    "Donne la PLUS GRANDE valeur ENTIÈRE de x qui la vérifie.", Number(sol),
                    "Isole x. Ici le coefficient de x est négatif : en divisant par lui, le sens de "
                    "l'inégalité s'inverse (≥ devient ≤). Puis cherche le plus grand entier possible.",
                    sides(lambda t: a * t + b, lambda t: c + 0 * t, (-10, 10)))
    ex.solution = _sol
    return ex


def l3_sqrt_sol(r):
    k = r.choice([2, 3, 5, 6, 7, 8, 10, 11, 12, 13])
    _sol = lines(f"x² = {k} : les solutions sont √{k} et −√{k}.", f"√{k} ≈ {math.sqrt(k):.4f}, donc x ≈ ±{math.sqrt(k):.4f}")
    ex = Exercise(f"Résous : x² = {k}\n(écris par ex. sqrt({k}) et −sqrt({k}), ou des valeurs décimales précises)",
                    Numbers([math.sqrt(k), -math.sqrt(k)]),
                    "Même idée que x² = 9, mais la racine n'est pas entière. Deux solutions : ±√" + str(k) + ".",
                    curves(lambda t: t ** 2, lambda t: k + 0 * t, xlim=(-5, 5), labels=["x²", str(k)]))
    ex.solution = _sol
    return ex


def l3_shifted_square(r):
    a, b = r.randint(-6, 6), r.randint(1, 7)
    _sol = lines(f"(x {signed(-a)})² = {b * b} ⇒ x {signed(-a)} = {b} ou x {signed(-a)} = −{b}",
                 f"Premier cas : x = {b} {signed(a)} = {fmt(a + b)}", f"Second cas : x = −{b} {signed(a)} = {fmt(a - b)}")
    ex = Exercise(f"Résous : (x {signed(-a)})² = {b * b}", Numbers([a + b, a - b]),
                    "Comme pour x² = A : la parenthèse vaut +√A ou −√A. Écris les deux cas et isole x.",
                    curves(lambda t: (t - a) ** 2, lambda t: b * b + 0 * t, xlim=(a - b - 3, a + b + 3)))
    ex.solution = _sol
    return ex


def l3_abs_lin(r):
    a, b, c = r.randint(2, 5), nz(r, -8, 8), r.randint(1, 10)
    _sol = lines(f"|{lin(a, b)}| = {c} ⇒ {lin(a, b)} = {c} ou {lin(a, b)} = −{c}",
                 [f"Cas 1 : {lin(a, b)} = {c}"] + steps_lin(a, b, c), [f"Cas 2 : {lin(a, b)} = −{c}"] + steps_lin(a, b, -c))
    ex = Exercise(f"Résous : |{lin(a, b)}| = {c}", Numbers([F(c - b, a), F(-c - b, a)]),
                    "|A| = c donne deux équations : A = c et A = −c. Résous chacune (fractions possibles).",
                    curves(lambda t: np.abs(a * t + b), lambda t: c + 0 * t, xlim=(-10, 10), ylim=(-2, 20)))
    ex.solution = _sol
    return ex


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
    _d = b * b - 4 * c
    _sol = lines(f"x² {signed(b)}x {signed(c)} = 0 : a = 1, b = {fmt(b)}, c = {fmt(c)}",
                 f"Δ = b² − 4ac = {par(b)}² − 4 × {par(c)} = {fmt(_d)}",
                 f"x = (−b ± √Δ) / 2 = ({fmt(-b)} ± {math.isqrt(_d)}) / 2",
                 f"Solutions : {fmt(p)} et {fmt(q)}" if p != q else f"Δ = 0 : une seule solution {fmt(p)}",
                 f"Astuce de vérification : somme = {fmt(p + q)} = −b et produit = {fmt(p * q)} = c ✔")
    ex = Exercise(f"Résous : {poly({2: 1, 1: b, 0: c})} = 0", Numbers(sorted({p, q})), M_QUAD,
                    _quad_draw(1, b, c, p, q))
    ex.solution = _sol
    return ex


def l4_general(r):
    a = r.randint(2, 4)
    p, q = r.randint(-5, 5), r.randint(-5, 5)
    b, c = -a * (p + q), a * p * q
    _sol = lines(f"{poly({2: a, 1: b, 0: c})} = 0. On divise tout par {a} : {poly({2: 1, 1: b // a, 0: c // a})} = 0",
                 f"Δ = {par(b // a)}² − 4 × {par(c // a)} = {fmt((b // a) ** 2 - 4 * (c // a))}",
                 f"x = ({fmt(-(b // a))} ± √{fmt((b // a) ** 2 - 4 * (c // a))}) / 2",
                 f"Solutions : {fmt(min(p, q))} et {fmt(max(p, q))}" if p != q else f"Une seule solution : {fmt(p)}")
    ex = Exercise(f"Résous : {poly({2: a, 1: b, 0: c})} = 0", Numbers(sorted({p, q})),
                    M_QUAD + "\nAstuce : tu peux d'abord diviser toute l'équation par " + str(a) + ".",
                    _quad_draw(a, b, c, p, q))
    ex.solution = _sol
    return ex


def l4_delta(r):
    a, b, c = nz(r, -5, 5), r.randint(-10, 10), r.randint(-10, 10)
    _sol = lines(f"a = {fmt(a)}, b = {fmt(b)}, c = {fmt(c)}", f"Δ = b² − 4ac = {par(b)}² − 4 × {par(a)} × {par(c)}",
                 f"Δ = {b * b} − {par(4 * a * c)} = {b * b - 4 * a * c}")
    ex = Exercise(f"Calcule le discriminant Δ de {poly({2: a, 1: b, 0: c})}.", Number(b * b - 4 * a * c),
                    "Δ = b² − 4ac. Repère bien a, b, c avec leurs signes. Attention : b² est toujours positif.",
                    _quad_draw(a, b, c))
    ex.solution = _sol
    return ex


def l4_count(r):
    a, b, c = nz(r, -4, 4), r.randint(-8, 8), r.randint(-8, 8)
    d = b * b - 4 * a * c
    n = 2 if d > 0 else (1 if d == 0 else 0)
    _sol = lines(f"a = {fmt(a)}, b = {fmt(b)}, c = {fmt(c)}", f"Δ = {par(b)}² − 4 × {par(a)} × {par(c)} = {d}",
                 f"Δ {'> 0 → 2 solutions' if d > 0 else ('= 0 → 1 solution' if d == 0 else '< 0 → 0 solution')}")
    ex = Exercise(f"Combien de solutions réelles a l'équation {poly({2: a, 1: b, 0: c})} = 0 ?",
                    Number(n),
                    "Calcule Δ = b² − 4ac et regarde son signe : Δ > 0 → 2, Δ = 0 → 1, Δ < 0 → 0.\n"
                    "Sur le graphe : c'est le nombre de fois où la parabole coupe l'axe horizontal.",
                    None)
    ex.solution = _sol
    return ex


def l4_irrational(r):
    while True:
        b, c = r.randint(-6, 6), r.randint(-8, 4)
        d = b * b - 4 * c
        if d > 0 and math.isqrt(d) ** 2 != d:
            break
    s = math.sqrt(d)
    _sol = lines(f"a = 1, b = {fmt(b)}, c = {fmt(c)}", f"Δ = {par(b)}² − 4 × {par(c)} = {d} (pas un carré parfait)",
                 f"x = ({fmt(-b)} − √{d}) / 2 ≈ {(-b - s) / 2:.4f}   et   x = ({fmt(-b)} + √{d}) / 2 ≈ {(-b + s) / 2:.4f}")
    ex = Exercise(f"Résous : {poly({2: 1, 1: b, 0: c})} = 0\n(les solutions contiennent une racine carrée)",
                    Numbers([(-b - s) / 2, (-b + s) / 2]),
                    M_QUAD + "\nTu peux écrire la réponse avec sqrt(...), par ex. (−3 + sqrt(5))/2.",
                    _quad_draw(1, b, c, (-b - s) / 2, (-b + s) / 2))
    ex.solution = _sol
    return ex


def l4_none(r):
    h, k = r.randint(-4, 4), r.randint(1, 6)
    b, c = -2 * h, h * h + k
    _sol = lines(f"a = 1, b = {fmt(b)}, c = {fmt(c)}", f"Δ = {par(b)}² − 4 × {c} = {b * b - 4 * c} < 0",
                 "Δ négatif : la parabole ne coupe jamais l'axe, pas de solution réelle.")
    ex = Exercise(f"Résous : {poly({2: 1, 1: b, 0: c})} = 0\n(écris « aucune » s'il n'y a pas de solution réelle)",
                    Numbers([]), M_QUAD, _quad_draw(1, b, c))
    ex.solution = _sol
    return ex


def l4_double(r):
    h = r.randint(-7, 7)
    _sol = lines(f"On reconnaît (x {signed(-h)})² = x² {signed(-2 * h)}x + {h * h}",
                 f"(x {signed(-h)})² = 0 ⇒ x {signed(-h)} = 0 ⇒ x = {h}", "(Δ = 0 : une seule solution, dite « double ».)")
    ex = Exercise(f"Résous : {poly({2: 1, 1: -2 * h, 0: h * h})} = 0", Numbers([h]),
                    M_QUAD + "\nRemarque : regarde si tu reconnais une identité remarquable (a − b)² = a² − 2ab + b².",
                    _quad_draw(1, -2 * h, h * h, h))
    ex.solution = _sol
    return ex


def l4_vertex(r):
    a = nz(r, -3, 3)
    h = r.randint(-6, 6)
    b, c = -2 * a * h, r.randint(-8, 8)
    _sol = lines(f"a = {a}, b = {b}", f"x_sommet = −b / (2a) = {fmt(-b)} / {par(2 * a)} = {h}")
    ex = Exercise(f"Quelle est l'abscisse du sommet de la parabole y = {poly({2: a, 1: b, 0: c})} ?",
                    Number(h),
                    "Le sommet est à x = −b / (2a). Une parabole est symétrique : son sommet est au milieu "
                    "de ses deux racines, quand elles existent.",
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l4_sum_roots(r):
    a = r.randint(1, 5)
    b, c = r.randint(-15, 15), r.randint(-20, -1)
    _sol = lines(f"a = {a}, b = {b}, c = {c}", f"Somme des solutions = −b/a = {fmt(-b)}/{a} = {fmt(F(-b, a))}")
    ex = Exercise(f"Sans calculer les solutions, donne leur SOMME pour {poly({2: a, 1: b, 0: c})} = 0.",
                    Number(F(-b, a)),
                    "Pour ax² + bx + c = 0 : somme des solutions = −b/a, produit = c/a.\n"
                    "(Tu peux le retrouver en additionnant les deux formules avec ±√Δ : les √Δ s'annulent.)",
                    _quad_draw(a, b, c))
    ex.solution = _sol
    return ex


def l4_area(r):
    w, k = r.randint(2, 12), r.randint(1, 6)
    A = w * (w + k)
    _sol = lines(f"Largeur x, longueur x + {k} : x(x + {k}) = {A}", f"x² + {k}x − {A} = 0",
                 f"Δ = {k}² + 4 × {A} = {k * k + 4 * A} = {math.isqrt(k * k + 4 * A)}²",
                 f"x = (−{k} ± {math.isqrt(k * k + 4 * A)}) / 2 → {w} ou {-w - k}",
                 f"Une longueur est positive : x = {w}. Vérification : {w} × {w + k} = {A} ✔")
    ex = Exercise(f"Un rectangle a une longueur qui dépasse sa largeur de {k} m et une aire de {A} m². "
                    "Quelle est sa largeur ?", Number(w),
                    "Largeur = x, longueur = x + " + str(k) + ". Aire = largeur × longueur donne une équation "
                    "du second degré. Garde la solution qui a du sens (une longueur est positive).",
                    rectangle(f"x + {k}", "x"))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 5

def _lines(a1, b1, a2, b2, x0):
    a1, b1, a2, b2, x0 = map(float, (a1, b1, a2, b2, x0))
    return curves(lambda t: a1 * t + b1, lambda t: a2 * t + b2, xlim=(x0 - 6, x0 + 6),
                  labels=["droite 1", "droite 2"])


def l5_subst(r):
    x, y = r.randint(-6, 6), r.randint(-6, 6)
    a = nz(r, -4, 4)
    b = y - a * x
    while True:  # c + d·a ≠ 0, sinon les deux droites sont parallèles
        c, d = nz(r, -5, 5), nz(r, -5, 5)
        if c + d * a != 0:
            break
    e = c * x + d * y
    _sol = lines(f"On remplace y par {lin(a, b)} dans la 2e équation : {term(c, 'x')} + {d}({lin(a, b)}) = {e}",
                 f"On développe : {term(c + d * a, 'x')} {signed(d * b)} = {e}", steps_lin(c + d * a, d * b, e),
                 f"Puis y = {a} × {par(x)} {signed(b)} = {y}")
    ex = Exercise(f"Résous le système :\n  y = {lin(a, b)}\n  {term(c, 'x')} {sterm(d, 'y')} = {e}",
                    Numbers([x, y], ordered=True),
                    "Substitution : remplace y dans la deuxième équation par l'expression de la première.\n"
                    "Tu obtiens une équation en x seul. Résous-la, puis calcule y.",
                    _lines(a, b, F(-c, d), F(e, d), x))
    ex.solution = _sol
    return ex


def l5_general(r):
    while True:
        x, y = r.randint(-6, 6), r.randint(-6, 6)
        a, b, c, d = (nz(r, -5, 5) for _ in range(4))
        if a * d - b * c != 0:
            break
    e, f = a * x + b * y, c * x + d * y
    _det = a * d - b * c
    _sol = lines(f"On élimine y : on multiplie la 1re équation par {par(d)} et la 2e par {par(b)}, puis on soustrait.",
                 f"({a} × {par(d)} − {par(c)} × {par(b)})x = {par(e)} × {par(d)} − {par(f)} × {par(b)}  ⇒  "
                 f"{term(_det, 'x')} = {fmt(e * d - f * b)}",
                 f"x = {fmt(e * d - f * b)} / {par(_det)} = {fmt(x)}",
                 [f"On remplace x = {fmt(x)} dans la 1re équation : {a} × {par(x)} {sterm(b, 'y')} = {fmt(e)}"]
                 + steps_lin(b, a * x, e, "y"))
    ex = Exercise(f"Résous le système :\n  {term(a, 'x')} {sterm(b, 'y')} = {e}\n  {term(c, 'x')} {sterm(d, 'y')} = {f}",
                    Numbers([x, y], ordered=True),
                    "Combinaison : multiplie les équations pour que les coefficients de y (ou de x) soient opposés, "
                    "puis additionne-les : une inconnue disparaît. Résous, puis remplace pour trouver l'autre.\n"
                    "Réponse au format : x, y",
                    _lines(F(-a, b), F(e, b), F(-c, d), F(f, d), x))
    ex.solution = _sol
    return ex


def l5_fraction(r):
    while True:
        a, b, c, d = (nz(r, -5, 5) for _ in range(4))
        det = a * d - b * c
        if det not in (0, 1, -1):
            break
    e, f = r.randint(-9, 9), r.randint(-9, 9)
    x, y = F(e * d - b * f, det), F(a * f - c * e, det)
    _sol = lines(f"Déterminant : {a} × {par(d)} − {par(b)} × {par(c)} = {det}",
                 f"x = ({e} × {par(d)} − {par(b)} × {f}) / {par(det)} = {fmt(x)}",
                 f"y = ({a} × {f} − {par(c)} × {e}) / {par(det)} = {fmt(y)}",
                 "(C'est la règle de Cramer ; la combinaison des lignes donne le même résultat.)",
                 f"Vérification : {a} × {par(x)} + {par(b)} × {par(y)} = {fmt(a * x + b * y)} = {e} ✔")
    ex = Exercise(f"Résous le système (solutions éventuellement fractionnaires) :\n  {term(a, 'x')} {sterm(b, 'y')} = {e}\n  {term(c, 'x')} {sterm(d, 'y')} = {f}",
                    Numbers([x, y], ordered=True),
                    "Même méthode (substitution ou combinaison). Garde les fractions exactes jusqu'au bout.\n"
                    "Réponse au format : x, y",
                    _lines(F(-a, b), F(e, b), F(-c, d), F(f, d), float(x)))
    ex.solution = _sol
    return ex


def l5_word(r):
    p1, p2 = r.randint(1, 6), r.randint(2, 9)
    n1, n2, m1, m2 = r.randint(1, 5), r.randint(1, 5), r.randint(1, 5), r.randint(1, 5)
    if n1 * m2 == n2 * m1:
        m2 += 1
    t1, t2 = n1 * p1 + n2 * p2, m1 * p1 + m2 * p2
    _sol = lines("x = prix du café, y = prix du croissant.", f"{n1}x + {n2}y = {t1}   et   {m1}x + {m2}y = {t2}",
                 f"Déterminant : {n1} × {m2} − {n2} × {m1} = {n1 * m2 - n2 * m1}",
                 f"x = ({t1} × {m2} − {n2} × {t2}) / {n1 * m2 - n2 * m1} = {p1}",
                 f"y = ({n1} × {t2} − {m1} × {t1}) / {n1 * m2 - n2 * m1} = {p2}",
                 f"Vérification : {n1} × {p1} + {n2} × {p2} = {t1} ✔")
    ex = Exercise(f"{n1} cafés et {n2} croissants coûtent {t1} €. {m1} cafés et {m2} croissants coûtent {t2} €.\n"
                    "Quel est le prix d'un café, puis d'un croissant ?",
                    Numbers([p1, p2], ordered=True),
                    "Appelle x le prix d'un café et y celui d'un croissant. Chaque phrase donne une équation. "
                    "Résous le système. Réponse : café, croissant",
                    None)
    ex.solution = _sol
    return ex


def l5_triangular(r):
    x, y, z = (r.randint(-5, 5) for _ in range(3))
    a, b, _c = nz(r, -3, 3), nz(r, -3, 3), nz(r, -3, 3)
    _sol = lines(f"Dernière équation : {term(b, 'z')} = {b * z} ⇒ z = {z}",
                 f"2e équation : y {sterm(a, 'z')} = {y + a * z} ⇒ y = {y + a * z} − ({a} × {par(z)}) = {y}",
                 f"1re équation : x + y + z = {x + y + z} ⇒ x = {x + y + z} − {par(y)} − {par(z)} = {x}")
    ex = Exercise(f"Résous le système à 3 inconnues :\n  x + y + z = {x + y + z}\n"
                    f"  y {sterm(a, 'z')} = {y + a * z}\n  {term(b, 'z')} = {b * z}",
                    Numbers([x, y, z], ordered=True),
                    "Le système est « en escalier » : commence par la dernière équation (une seule inconnue), "
                    "remonte en remplaçant à chaque ligne. Réponse : x, y, z",
                    None)
    ex.solution = _sol
    return ex


def l5_intersection(r):
    x = r.randint(-5, 5)
    a1, a2 = r.sample(range(-4, 5), 2)
    b1 = r.randint(-6, 6)
    y = a1 * x + b1
    b2 = y - a2 * x
    _sol = lines(f"Au point d'intersection : {lin(a1, b1)} = {lin(a2, b2)}",
                 f"{term(a1 - a2, 'x')} = {b2 - b1}  ⇒  x = {b2 - b1} / {par(a1 - a2)} = {x}",
                 f"y = {a1} × {par(x)} {signed(b1)} = {y}")
    ex = Exercise(f"Donne les coordonnées du point d'intersection des droites\n"
                    f"  y = {lin(a1, b1)}   et   y = {lin(a2, b2)}", Numbers([x, y], ordered=True),
                    "Au point d'intersection, les deux y sont égaux : écris « expression 1 = expression 2 », "
                    "résous en x, puis calcule y. Réponse : x, y",
                    _lines(a1, b1, a2, b2, x))
    ex.solution = _sol
    return ex


def l5_biquad(r):
    p, q = r.sample(range(1, 6), 2)
    s, pr = p * p + q * q, p * p * q * q
    _sol = lines(f"On pose X = x² : X² − {s}X + {pr} = 0", f"Δ = {s}² − 4 × {pr} = {s * s - 4 * pr} = {abs(p * p - q * q)}²",
                 f"X = ({s} ± {abs(p * p - q * q)}) / 2 → X = {p * p} ou X = {q * q}",
                 f"x² = {p * p} ⇒ x = ±{p} ;  x² = {q * q} ⇒ x = ±{q}")
    ex = Exercise(f"Résous : x⁴ − {s}x² + {pr} = 0", Numbers([p, -p, q, -q]),
                    "Pose X = x². L'équation devient X² − " + str(s) + "X + " + str(pr) + " = 0 (second degré). "
                    "Trouve les valeurs de X, puis reviens à x avec x² = X (attention aux ±).",
                    curves(lambda t: t ** 4 - s * t ** 2 + pr, xlim=(-max(p, q) - 1.5, max(p, q) + 1.5), ylim=(-pr - 5, pr + 20)))
    ex.solution = _sol
    return ex


def l5_rational(r):
    while True:
        a, b, c = r.randint(-6, 6), r.randint(-6, 6), r.choice([-3, -2, 2, 3, 4])
        if a != b:
            break
    x = F(c * b - a, 1 - c)
    _sol = lines(f"x ≠ {fmt(-b)} (dénominateur). On multiplie par (x {signed(b)}) : x {signed(a)} = {c}(x {signed(b)})",
                 f"x {signed(a)} = {term(c, 'x')} {signed(c * b)}", f"x − {par(c)}x = {c * b} − {par(a)}  ⇒  {term(1 - c, 'x')} = {c * b - a}",
                 f"x = {c * b - a} / {par(1 - c)} = {fmt(x)} (différent de {fmt(-b)} ✔)")
    ex = Exercise(f"Résous : (x {signed(a)}) / (x {signed(b)}) = {c}", Number(x),
                    "Multiplie les deux côtés par le dénominateur (en notant que x ne peut pas l'annuler), "
                    "développe, puis résous l'équation du premier degré. Vérifie que la solution n'annule pas le dénominateur.",
                    curves(lambda t: (t + a) / (t + b), lambda t: c + 0 * t, xlim=(-15, 15), ylim=(-10, 10)))
    ex.solution = _sol
    return ex


def l5_sqrt(r):
    a, b = r.randint(-8, 8), r.randint(1, 7)
    _sol = lines(f"√(x {signed(a)}) = {b}", f"On élève au carré : x {signed(a)} = {b * b}", steps_lin(1, a, b * b),
                 f"Vérification : √({b * b - a} {signed(a)}) = √{b * b} = {b} ✔")
    ex = Exercise(f"Résous : √(x {signed(a)}) = {b}", Number(b * b - a),
                    "Élève les deux côtés au carré pour faire disparaître la racine, puis isole x. "
                    "Vérifie ensuite dans l'équation de départ (une racine est toujours ≥ 0).",
                    curves(lambda t: np.sqrt(t + a), lambda t: b + 0 * t, xlim=(-a - 1, b * b - a + 8)))
    ex.solution = _sol
    return ex


def l5_parallel(r):
    a = nz(r, -4, 4)
    b1, b2 = r.sample(range(-8, 9), 2)
    k = r.randint(2, 3)
    _sol = lines(f"La 2e équation divisée par {k} : y = {lin(a, b2)}",
                 f"Les deux droites ont la même pente {a} mais des ordonnées à l'origine différentes ({b1} et {b2}).",
                 "Elles sont parallèles et distinctes : aucun point commun, donc aucune solution.")
    ex = Exercise(f"Résous le système :\n  y = {lin(a, b1)}\n  {k}y = {lin(k * a, k * b2)}\n"
                    "(écris « aucune » s'il n'y a pas de solution)", Numbers([]),
                    "Mets les deux équations sous la forme y = ... et compare les pentes et les ordonnées à l'origine. "
                    "Deux droites parallèles distinctes ne se coupent jamais.",
                    _lines(a, b1, a, b2, 0))
    ex.solution = _sol
    return ex


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
