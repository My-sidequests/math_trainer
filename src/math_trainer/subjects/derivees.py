"""Dérivées : règles de base, fonctions usuelles, produit/quotient, chaîne simple, dérivée seconde."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Expr, Number, Numbers
from ..core import Exercise, sterm, coef, curves, lin, nz, poly, signed

DESCRIPTIONS = {
    1: "Dérivée d'un polynôme, pente de la tangente",
    2: "Fonctions usuelles : exp, ln, sin, cos, 1/x, √x",
    3: "Produit et quotient",
    4: "Fonctions composées simples, extremums",
    5: "Dérivée seconde, approximations, Newton",
}

TABLE = ("Table de base :\n"
         "  (c)' = 0      (x)' = 1      (xⁿ)' = n·xⁿ⁻¹\n"
         "  (eˣ)' = eˣ    (ln x)' = 1/x   (sin x)' = cos x   (cos x)' = −sin x\n"
         "  (1/x)' = −1/x²   (√x)' = 1/(2√x)\n"
         "Linéarité : (a·f + b·g)' = a·f' + b·g'.")
M_POLY = ("Dérive terme par terme avec (xⁿ)' = n·xⁿ⁻¹ : l'exposant descend devant et diminue de 1.\n"
          "Une constante seule disparaît (sa dérivée vaut 0), un terme ax devient a.")
M_VALUE = "Calcule d'abord l'expression de f'(x), puis remplace x par la valeur. f'(a) = pente de la tangente en a."
M_PROD = "Produit : (u·v)' = u'·v + u·v'.\n" + TABLE
M_QUOT = "Quotient : (u/v)' = (u'·v − u·v') / v².\n" + TABLE
M_CHAIN = ("Fonction composée f(g(x)) : on dérive l'extérieur en gardant l'intérieur, puis on multiplie par la "
           "dérivée de l'intérieur : (f(g(x)))' = f'(g(x))·g'(x).\n"
           "Exemples : (eᵘ)' = u'·eᵘ, (uⁿ)' = n·u'·uⁿ⁻¹, (ln u)' = u'/u, (sin u)' = u'·cos u, (√u)' = u'/(2√u).")


def tangent(f, df, a, xlim):
    return curves(f, lambda t: f(a) + df(a) * (t - a), xlim=xlim, labels=["f", "tangente"], points=[(a, f(a))])


# ================================================================ NIVEAU 1

def l1_affine(r):
    a, b = nz(r, -9, 9), r.randint(-9, 9)
    return Exercise(f"Dérive f(x) = {lin(a, b)}.", Expr(str(a)), M_POLY, curves(lambda t: a * t + b, xlim=(-5, 5)))


def l1_power(r):
    n = r.randint(2, 7)
    return Exercise(f"Dérive f(x) = x{'²³⁴⁵⁶⁷'[n - 2]}.", Expr(f"{n}*x**{n - 1}"), M_POLY,
                    curves(lambda t: t ** n, xlim=(-1.5, 1.5)))


def l1_mono(r):
    a, n = nz(r, -8, 8), r.randint(2, 5)
    return Exercise(f"Dérive f(x) = {poly({n: a})}.", Expr(f"{a * n}*x**{n - 1}"), M_POLY,
                    curves(lambda t: a * t ** n, xlim=(-2, 2)))


def l1_quad(r):
    a, b, c = nz(r, -6, 6), r.randint(-9, 9), r.randint(-9, 9)
    return Exercise(f"Dérive f(x) = {poly({2: a, 1: b, 0: c})}.", Expr(f"{2 * a}*x + {b}"), M_POLY,
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(-5, 5)))


def l1_sq_value(r):
    k = r.randint(-6, 6)
    return Exercise(f"f(x) = x². Calcule f'({k}).", Number(2 * k), M_VALUE,
                    tangent(lambda t: t ** 2, lambda t: 2 * t, k, (k - 3, k + 3)))


def l1_slope(r):
    a, b, k = nz(r, -4, 4), r.randint(-6, 6), r.randint(-4, 4)
    return Exercise(f"Quelle est la pente de la tangente à f(x) = {poly({2: a, 1: b})} au point x = {k} ?",
                    Number(2 * a * k + b), M_VALUE,
                    tangent(lambda t: a * t ** 2 + b * t, lambda t: 2 * a * t + b, k, (k - 3, k + 3)))


def l1_cube_value(r):
    k = r.randint(-4, 4)
    return Exercise(f"f(x) = x³. Calcule f'({k}).", Number(3 * k * k), M_VALUE,
                    tangent(lambda t: t ** 3, lambda t: 3 * t ** 2, k, (k - 2, k + 2)))


def l1_monotony(r):
    a, b, k = nz(r, -4, 4), r.randint(-8, 8), r.randint(-4, 4)
    d = 2 * a * k + b
    if d == 0:
        b += 1
        d = 2 * a * k + b
    return Exercise(f"f(x) = {poly({2: a, 1: b})}. En x = {k}, f est-elle croissante ou décroissante ?",
                    Choice("croissante" if d > 0 else "decroissante", ["croissante", "décroissante"]),
                    "Le signe de f'(x) donne le sens de variation : f' > 0 → croissante, f' < 0 → décroissante. "
                    "Calcule f'(" + str(k) + ").", None)


def l1_poly(r):
    a, b, c = nz(r, -5, 5), nz(r, -6, 6), r.randint(-9, 9)
    return Exercise(f"Dérive f(x) = {poly({3: a, 2: b, 1: c, 0: r.randint(-9, 9)})}.",
                    Expr(f"{3 * a}*x**2 + {2 * b}*x + {c}"), M_POLY, None)


def l1_zero(r):
    h = r.randint(-6, 6)
    a = r.choice([1, 2, 3, -1])
    return Exercise(f"Pour quelle valeur de x la dérivée de f(x) = {poly({2: a, 1: -2 * a * h})} s'annule-t-elle ?",
                    Number(h), "Calcule f'(x), puis résous f'(x) = 0. C'est là que la tangente est horizontale.",
                    curves(lambda t: a * t ** 2 - 2 * a * h * t, xlim=(h - 5, h + 5)))


# ================================================================ NIVEAU 2

def l2_exp(r):
    a, b = nz(r, -5, 5), r.randint(-5, 5)
    return Exercise(f"Dérive f(x) = {coef(a)}eˣ {sterm(b, 'x')}.",
                    Expr(f"{a}*exp(x) + {b}"), TABLE, curves(lambda t: a * np.exp(t) + b * t, xlim=(-3, 2)))


def l2_ln(r):
    a, b = nz(r, -6, 6), r.randint(-5, 5)
    return Exercise(f"Dérive f(x) = {coef(a)}ln(x) {sterm(b, 'x')}  (x > 0).", Expr(f"{a}/x + {b}", (0.2, 4)), TABLE,
                    curves(lambda t: a * np.log(t) + b * t, xlim=(0.05, 6)))


def l2_trig(r):
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    return Exercise(f"Dérive f(x) = {a}·sin(x) {signed(b)}·cos(x).", Expr(f"{a}*cos(x) - {b}*sin(x)"),
                    TABLE + "\nAttention au signe : (cos x)' = −sin x.",
                    curves(lambda t: a * np.sin(t) + b * np.cos(t), xlim=(-7, 7)))


def l2_inv(r):
    a = nz(r, -9, 9)
    return Exercise(f"Dérive f(x) = {a}/x.", Expr(f"-{a}/x**2", (0.3, 3)),
                    TABLE + "\nTu peux aussi écrire 1/x = x⁻¹ et appliquer (xⁿ)' = n·xⁿ⁻¹.",
                    curves(lambda t: a / t, xlim=(-5, 5), ylim=(-10, 10)))


def l2_sqrt_value(r):
    k = r.randint(1, 8)
    return Exercise(f"f(x) = √x. Calcule f'({k * k}).", Number(F(1, 2 * k)), TABLE,
                    tangent(np.sqrt, lambda t: 1 / (2 * np.sqrt(t)), k * k, (0.01, k * k * 2 + 2)))


def l2_neg_power(r):
    n = r.randint(1, 4)
    return Exercise(f"Dérive f(x) = x^(−{n}).", Expr(f"-{n}*x**(-{n + 1})", (0.3, 3)),
                    "(xⁿ)' = n·xⁿ⁻¹ marche aussi pour n négatif : n = −" + str(n) + ".", None)


def l2_frac_power(r):
    p, q = r.choice([(3, 2), (5, 2), (1, 3), (4, 3), (2, 3)])
    return Exercise(f"Dérive f(x) = x^({p}/{q})  (x > 0).", Expr(f"({p}/{q})*x**({p}/{q} - 1)", (0.2, 3)),
                    "(xⁿ)' = n·xⁿ⁻¹ marche pour tout exposant réel. Ici n = " + f"{p}/{q}. Calcule n − 1.", None)


def l2_value0(r):
    a, b = nz(r, -6, 6), r.randint(-6, 6)
    return Exercise(f"f(x) = {coef(a)}eˣ {sterm(b, 'x²')}. Calcule f'(0).", Number(a), M_VALUE + "\nRappel : e⁰ = 1.",
                    tangent(lambda t: a * np.exp(t) + b * t ** 2, lambda t: a * np.exp(t) + 2 * b * t, 0, (-2, 2)))


def l2_tangent_line(r):
    k = nz(r, -4, 4)
    return Exercise(f"Donne l'équation de la tangente à f(x) = x² au point x = {k} (forme y = a*x + b).",
                    Expr(f"{2 * k}*x - {k * k}"),
                    "Tangente en a : y = f'(a)·(x − a) + f(a). Calcule f(a) et f'(a), puis développe.",
                    curves(lambda t: t ** 2, xlim=(k - 4, k + 4), points=[(k, k * k)]))


def l2_value1(r):
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    return Exercise(f"f(x) = {a}eˣ {signed(b)}·ln(x). Calcule f'(1) (valeur exacte, tu peux utiliser e).",
                    Number(a * math.e + b), M_VALUE + "\nRappel : e¹ = e, et 1/1 = 1.", None)


# ================================================================ NIVEAU 3

def l3_prod_poly(r):
    a, b = nz(r, -6, 6), nz(r, -6, 6)
    return Exercise(f"Dérive f(x) = (x {signed(a)})(x {signed(b)}).", Expr(f"2*x + {a + b}"),
                    M_PROD + "\n(Ou développe d'abord, puis dérive : les deux méthodes donnent le même résultat.)", None)


def l3_xexp(r):
    a = nz(r, -4, 4)
    return Exercise(f"Dérive f(x) = {coef(a)}x·eˣ.", Expr(f"{a}*(exp(x) + x*exp(x))"), M_PROD,
                    curves(lambda t: a * t * np.exp(t), xlim=(-5, 2)))


def l3_x2ln(r):
    return Exercise("Dérive f(x) = x²·ln(x)  (x > 0).", Expr("2*x*log(x) + x", (0.2, 4)), M_PROD,
                    curves(lambda t: t ** 2 * np.log(t), xlim=(0.01, 3)))


def l3_xsin(r):
    fn = r.choice(["sin", "cos"])
    target = "sin(x) + x*cos(x)" if fn == "sin" else "cos(x) - x*sin(x)"
    return Exercise(f"Dérive f(x) = x·{fn}(x).", Expr(target), M_PROD,
                    curves(lambda t: t * (np.sin(t) if fn == "sin" else np.cos(t)), xlim=(-8, 8)))


def l3_quot_lin(r):
    a, b = r.sample(range(-6, 7), 2)
    return Exercise(f"Dérive f(x) = (x {signed(a)}) / (x {signed(b)}).", Expr(f"({b} - {a}) / (x + {b})**2"), M_QUOT,
                    curves(lambda t: (t + a) / (t + b), xlim=(-b - 6, -b + 6), ylim=(-10, 10)))


def l3_exp_over_x(r):
    return Exercise("Dérive f(x) = eˣ / x.", Expr("exp(x)*(x - 1)/x**2", (0.3, 3)), M_QUOT,
                    curves(lambda t: np.exp(t) / t, xlim=(0.1, 3), ylim=(0, 12)))


def l3_value_prod(r):
    k = r.randint(-2, 2)
    return Exercise(f"f(x) = x²·eˣ. Calcule f'({k}) (exact avec e, ou décimal précis).",
                    Number((2 * k + k * k) * math.exp(k)), M_PROD, None)


def l3_lnx_over_x(r):
    k = r.choice([1, 2, 3, 4])
    return Exercise(f"f(x) = ln(x) / x. Calcule f'({k}) (exact ou décimal précis).", Number((1 - math.log(k)) / k ** 2),
                    M_QUOT, curves(lambda t: np.log(t) / t, xlim=(0.3, 8)))


def l3_inv_quad(r):
    a = r.randint(1, 9)
    return Exercise(f"Dérive f(x) = 1 / (x² + {a}).", Expr(f"-2*x/(x**2 + {a})**2"),
                    M_QUOT + "\nIci le numérateur est constant : u' = 0.",
                    curves(lambda t: 1 / (t ** 2 + a), xlim=(-5, 5)))


def l3_x3exp(r):
    return Exercise("Dérive f(x) = x³·eˣ.", Expr("3*x**2*exp(x) + x**3*exp(x)"), M_PROD,
                    curves(lambda t: t ** 3 * np.exp(t), xlim=(-5, 1.5)))


# ================================================================ NIVEAU 4

def l4_power_lin(r):
    a, b, n = nz(r, -4, 4), nz(r, -5, 5), r.randint(2, 5)
    return Exercise(f"Dérive f(x) = ({lin(a, b)})^{n}.", Expr(f"{n}*{a}*({a}*x + {b})**{n - 1}"), M_CHAIN, None)


def l4_exp_ax(r):
    a, b = nz(r, -5, 5), r.randint(-4, 4)
    return Exercise(f"Dérive f(x) = e^({lin(a, b)}).", Expr(f"{a}*exp({a}*x + {b})"), M_CHAIN,
                    curves(lambda t: np.exp(a * t + b), xlim=(-2, 2), ylim=(-1, 20)))


def l4_sin_ax(r):
    a, b = nz(r, -5, 5), r.randint(-3, 3)
    fn = r.choice(["sin", "cos"])
    target = f"{a}*cos({a}*x + {b})" if fn == "sin" else f"-{a}*sin({a}*x + {b})"
    return Exercise(f"Dérive f(x) = {fn}({lin(a, b)}).", Expr(target), M_CHAIN, None)


def l4_ln_ax(r):
    a, b = r.randint(1, 6), r.randint(1, 6)
    return Exercise(f"Dérive f(x) = ln({lin(a, b)}).", Expr(f"{a}/({a}*x + {b})", (0.1, 4)), M_CHAIN, None)


def l4_sqrt_ax(r):
    a, b = r.randint(1, 6), r.randint(1, 9)
    return Exercise(f"Dérive f(x) = √({lin(a, b)}).", Expr(f"{a}/(2*sqrt({a}*x + {b}))", (0.1, 4)), M_CHAIN, None)


def l4_extremum_x(r):
    a, h = nz(r, -3, 3), r.randint(-6, 6)
    b, c = -2 * a * h, r.randint(-9, 9)
    return Exercise(f"f(x) = {poly({2: a, 1: b, 0: c})}. En quel x f atteint-elle son extremum ?", Number(h),
                    "Un extremum (min ou max) se trouve là où f'(x) = 0 (tangente horizontale). Résous f'(x) = 0.",
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(h - 5, h + 5)))


def l4_critical_cubic(r):
    a = r.randint(1, 5)
    return Exercise(f"Trouve les points critiques (f'(x) = 0) de f(x) = x³ − {3 * a * a}x.", Numbers([a, -a]),
                    "Dérive, puis résous f'(x) = 0 (équation du second degré très simple).",
                    curves(lambda t: t ** 3 - 3 * a * a * t, xlim=(-2.2 * a, 2.2 * a)))


def l4_nature(r):
    a = nz(r, -4, 4)
    b, c = r.randint(-8, 8), r.randint(-9, 9)
    return Exercise(f"f(x) = {poly({2: a, 1: b, 0: c})} a un seul extremum. Est-ce un minimum ou un maximum ?",
                    Choice("minimum" if a > 0 else "maximum", ["minimum", "maximum"]),
                    "Regarde comment f' change de signe autour du point critique : − puis + → minimum ; + puis − → "
                    "maximum. (Ou le signe de f'' : f'' > 0 → minimum.)", None)


def l4_max_value(r):
    h, k = r.randint(-5, 5), r.randint(-9, 9)
    return Exercise(f"Quelle est la valeur maximale de f(x) = {poly({2: -1, 1: 2 * h, 0: k - h * h})} ?", Number(k),
                    "1. Trouve x tel que f'(x) = 0.  2. Calcule f en ce x : c'est la valeur maximale.",
                    curves(lambda t: -(t - h) ** 2 + k, xlim=(h - 5, h + 5)))


def l4_exp_x2(r):
    a = nz(r, -3, 3)
    return Exercise(f"Dérive f(x) = e^({poly({2: a})}).", Expr(f"{2 * a}*x*exp({a}*x**2)"), M_CHAIN,
                    curves(lambda t: np.exp(a * t ** 2), xlim=(-2, 2), ylim=(-1, 10)))


# ================================================================ NIVEAU 5

def l5_second(r):
    a, b, c = nz(r, -4, 4), r.randint(-6, 6), r.randint(-9, 9)
    return Exercise(f"Calcule la dérivée seconde f''(x) de f(x) = {poly({4: a, 3: b, 1: c})}.",
                    Expr(f"{12 * a}*x**2 + {6 * b}*x"), "La dérivée seconde est la dérivée de la dérivée : dérive deux fois.",
                    None)


def l5_second_value(r):
    k = r.randint(-2, 2)
    return Exercise(f"f(x) = eˣ + x³. Calcule f''({k}) (exact avec e, ou décimal précis).",
                    Number(math.exp(k) + 6 * k), "Dérive deux fois, puis remplace x.", None)


def l5_inflection(r):
    b = r.choice([-9, -6, -3, 3, 6, 9])
    c = r.randint(-5, 5)
    return Exercise(f"f(x) = {poly({3: 1, 2: b, 1: c})}. En quel x se trouve le point d'inflexion ?", Number(-b / 3),
                    "Point d'inflexion : la courbure change de sens, f''(x) = 0 (et change de signe). Dérive deux fois.",
                    curves(lambda t: t ** 3 + b * t ** 2 + c * t, xlim=(-b / 3 - 6, -b / 3 + 6), ylim=(-150, 150)))


def l5_linear_approx(r):
    a = r.randint(2, 9)
    d = r.choice([0.1, 0.2, 0.5, -0.3, 1])
    return Exercise(f"Approximation linéaire : estime √({a * a} + {d}) avec f(a) + f'(a)·δ où f(x) = √x, a = {a * a}, "
                    f"δ = {d}.", Number(a + d / (2 * a), 1e-4),
                    "Près de a, une fonction ressemble à sa tangente : f(a + δ) ≈ f(a) + f'(a)·δ. "
                    "Calcule f(a), f'(a) = 1/(2√a), puis la formule.",
                    tangent(np.sqrt, lambda t: 1 / (2 * np.sqrt(t)), a * a, (0.1, 2 * a * a)))


def l5_numeric(r):
    k, h = r.randint(1, 3), r.choice([0.1, 0.5, 1])
    val = ((k + h) ** 3 - (k - h) ** 3) / (2 * h)
    return Exercise(f"Dérivée numérique : calcule (f({k} + {h}) − f({k} − {h})) / (2·{h}) pour f(x) = x³.\n"
                    f"Compare avec la vraie valeur f'({k}).", Number(val, 1e-4),
                    "Calcule les deux images, soustrais, divise. C'est la « différence centrée » utilisée pour "
                    "vérifier un gradient calculé à la main (gradient checking).",
                    tangent(lambda t: t ** 3, lambda t: 3 * t ** 2, k, (k - 2, k + 2)))


def l5_newton(r):
    a = r.choice([2, 3, 5, 7, 10])
    x0 = r.choice([1, 2, 3])
    x1 = x0 - (x0 * x0 - a) / (2 * x0)
    return Exercise(f"Méthode de Newton pour résoudre x² − {a} = 0 : fais UNE itération x₁ = x₀ − f(x₀)/f'(x₀) avec x₀ = {x0}.",
                    Number(x1), "f(x) = x² − " + str(a) + ", f'(x) = 2x. Remplace x₀ dans la formule. "
                    "(Géométriquement : on suit la tangente jusqu'à l'axe horizontal.)",
                    tangent(lambda t: t ** 2 - a, lambda t: 2 * t, x0, (0, 4)))


def l5_velocity(r):
    a, b, t = nz(r, -5, 5), r.randint(-9, 9), r.randint(0, 6)
    return Exercise(f"La position d'un objet est s(t) = {poly({2: a, 1: b}, 't')} (mètres). Vitesse à t = {t} s ?",
                    Number(2 * a * t + b), "La vitesse est la dérivée de la position : v(t) = s'(t).",
                    curves(lambda u: a * u ** 2 + b * u, xlim=(0, 8)))


def l5_accel(r):
    a, b, c = nz(r, -3, 3), r.randint(-6, 6), r.randint(-9, 9)
    t = r.randint(0, 5)
    return Exercise(f"s(t) = {poly({3: a, 2: b, 1: c}, 't')}. Accélération à t = {t} ?", Number(6 * a * t + 2 * b),
                    "Accélération = dérivée de la vitesse = dérivée seconde de la position.", None)


def l5_ax(r):
    a = r.choice([2, 3, 10])
    return Exercise(f"f(x) = {a}ˣ. Calcule f'(0) (exact avec ln, ou décimal précis).", Number(math.log(a)),
                    "aˣ = e^(x·ln a). Dérive avec la règle de la chaîne : (aˣ)' = ln(a)·aˣ.",
                    tangent(lambda t: float(a) ** t, lambda t: math.log(a) * float(a) ** t, 0, (-2, 2)))


def l5_convex(r):
    kind = r.choice(["convexe", "concave"])
    expr = {"convexe": r.choice(["eˣ", "x² + 3x", "x⁴", "−ln(x)"]),
            "concave": r.choice(["ln(x)", "√x", "−x²", "−eˣ"])}[kind]
    return Exercise(f"Sur son domaine, f(x) = {expr} est-elle convexe ou concave ?",
                    Choice(kind, ["convexe", "concave"]),
                    "Calcule f''(x). f'' ≥ 0 partout → convexe (forme de bol, ∪) ; f'' ≤ 0 → concave (∩).\n"
                    "Les fonctions de coût convexes ont un seul minimum : c'est ce qu'on aime en optimisation.", None)


LEVELS = {
    1: [l1_affine, l1_power, l1_mono, l1_quad, l1_sq_value, l1_slope, l1_cube_value, l1_monotony, l1_poly, l1_zero],
    2: [l2_exp, l2_ln, l2_trig, l2_inv, l2_sqrt_value, l2_neg_power, l2_frac_power, l2_value0, l2_tangent_line,
        l2_value1],
    3: [l3_prod_poly, l3_xexp, l3_x2ln, l3_xsin, l3_quot_lin, l3_exp_over_x, l3_value_prod, l3_lnx_over_x,
        l3_inv_quad, l3_x3exp],
    4: [l4_power_lin, l4_exp_ax, l4_sin_ax, l4_ln_ax, l4_sqrt_ax, l4_extremum_x, l4_critical_cubic, l4_nature,
        l4_max_value, l4_exp_x2],
    5: [l5_second, l5_second_value, l5_inflection, l5_linear_approx, l5_numeric, l5_newton, l5_velocity, l5_accel,
        l5_ax, l5_convex],
}
