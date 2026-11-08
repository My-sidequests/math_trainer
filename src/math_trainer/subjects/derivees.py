"""Dérivées : règles de base, fonctions usuelles, produit/quotient, chaîne simple, dérivée seconde."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Expr, Number, Numbers
from ..core import Exercise, sterm, coef, curves, lin, nz, poly, signed

from ..core import fmt, lines, par, term  # noqa: E402,F811

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
    _sol = lines(f"(ax + b)' = a : le terme {term(a, 'x')} donne {a}, la constante {b} donne 0.", f"f'(x) = {a}")
    ex = Exercise(f"Dérive f(x) = {lin(a, b)}.", Expr(str(a)), M_POLY, curves(lambda t: a * t + b, xlim=(-5, 5)))
    ex.solution = _sol
    return ex


def l1_power(r):
    n = r.randint(2, 7)
    _sol = lines(f"(xⁿ)' = n·xⁿ⁻¹ avec n = {n} : l'exposant descend devant et diminue de 1.", f"f'(x) = {poly({n - 1: n})}")
    ex = Exercise(f"Dérive f(x) = x{'²³⁴⁵⁶⁷'[n - 2]}.", Expr(f"{n}*x**{n - 1}"), M_POLY,
                    curves(lambda t: t ** n, xlim=(-1.5, 1.5)))
    ex.solution = _sol
    return ex


def l1_mono(r):
    a, n = nz(r, -8, 8), r.randint(2, 5)
    _sol = lines(f"(a·xⁿ)' = a·n·xⁿ⁻¹ avec a = {a}, n = {n}", f"f'(x) = {a} × {n}·x^{n - 1} = {poly({n - 1: a * n})}")
    ex = Exercise(f"Dérive f(x) = {poly({n: a})}.", Expr(f"{a * n}*x**{n - 1}"), M_POLY,
                    curves(lambda t: a * t ** n, xlim=(-2, 2)))
    ex.solution = _sol
    return ex


def l1_quad(r):
    a, b, c = nz(r, -6, 6), r.randint(-9, 9), r.randint(-9, 9)
    _sol = lines(f"({term(a, 'x²')})' = {2 * a}x ;  ({term(b, 'x') if b else '0'})' = {b} ;  ({c})' = 0", f"f'(x) = {lin(2 * a, b)}")
    ex = Exercise(f"Dérive f(x) = {poly({2: a, 1: b, 0: c})}.", Expr(f"{2 * a}*x + {b}"), M_POLY,
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(-5, 5)))
    ex.solution = _sol
    return ex


def l1_sq_value(r):
    k = r.randint(-6, 6)
    _sol = lines("f'(x) = 2x", f"f'({k}) = 2 × {par(k)} = {2 * k}")
    ex = Exercise(f"f(x) = x². Calcule f'({k}).", Number(2 * k), M_VALUE,
                    tangent(lambda t: t ** 2, lambda t: 2 * t, k, (k - 3, k + 3)))
    ex.solution = _sol
    return ex


def l1_slope(r):
    a, b, k = nz(r, -4, 4), r.randint(-6, 6), r.randint(-4, 4)
    _sol = lines(f"f'(x) = {lin(2 * a, b)}", f"f'({k}) = {2 * a} × {par(k)} {signed(b)} = {2 * a * k + b}",
                 f"La pente de la tangente en x = {k} vaut {2 * a * k + b}.")
    ex = Exercise(f"Quelle est la pente de la tangente à f(x) = {poly({2: a, 1: b})} au point x = {k} ?",
                    Number(2 * a * k + b), M_VALUE,
                    tangent(lambda t: a * t ** 2 + b * t, lambda t: 2 * a * t + b, k, (k - 3, k + 3)))
    ex.solution = _sol
    return ex


def l1_cube_value(r):
    k = r.randint(-4, 4)
    _sol = lines("f'(x) = 3x²", f"f'({k}) = 3 × {par(k)}² = 3 × {k * k} = {3 * k * k}")
    ex = Exercise(f"f(x) = x³. Calcule f'({k}).", Number(3 * k * k), M_VALUE,
                    tangent(lambda t: t ** 3, lambda t: 3 * t ** 2, k, (k - 2, k + 2)))
    ex.solution = _sol
    return ex


def l1_monotony(r):
    a, b, k = nz(r, -4, 4), r.randint(-8, 8), r.randint(-4, 4)
    d = 2 * a * k + b
    if d == 0:
        b += 1
        d = 2 * a * k + b
    _sol = lines(f"f'(x) = {lin(2 * a, b)}", f"f'({k}) = {2 * a} × {par(k)} {signed(b)} = {d}",
                 f"f'({k}) {'> 0 : f est croissante' if d > 0 else '< 0 : f est décroissante'} en ce point.")
    ex = Exercise(f"f(x) = {poly({2: a, 1: b})}. En x = {k}, f est-elle croissante ou décroissante ?",
                    Choice("croissante" if d > 0 else "decroissante", ["croissante", "décroissante"]),
                    "Le signe de f'(x) donne le sens de variation : f' > 0 → croissante, f' < 0 → décroissante. "
                    "Calcule f'(" + str(k) + ").", None)
    ex.solution = _sol
    return ex


def l1_poly(r):
    a, b, c = nz(r, -5, 5), nz(r, -6, 6), r.randint(-9, 9)
    _sol = lines(f"On dérive terme par terme : ({term(a, 'x³')})' = {term(3 * a, 'x²')}, ({term(b, 'x²')})' = {term(2 * b, 'x')}, "
                 f"({term(c, 'x') if c else '0'})' = {c}, constante → 0", f"f'(x) = {poly({2: 3 * a, 1: 2 * b, 0: c})}")
    ex = Exercise(f"Dérive f(x) = {poly({3: a, 2: b, 1: c, 0: r.randint(-9, 9)})}.",
                    Expr(f"{3 * a}*x**2 + {2 * b}*x + {c}"), M_POLY, None)
    ex.solution = _sol
    return ex


def l1_zero(r):
    h = r.randint(-6, 6)
    a = r.choice([1, 2, 3, -1])
    _sol = lines(f"f'(x) = {lin(2 * a, -2 * a * h)}", f"f'(x) = 0 ⇔ {term(2 * a, 'x')} = {2 * a * h} ⇔ x = {h}")
    ex = Exercise(f"Pour quelle valeur de x la dérivée de f(x) = {poly({2: a, 1: -2 * a * h})} s'annule-t-elle ?",
                    Number(h), "Calcule f'(x), puis résous f'(x) = 0. C'est là que la tangente est horizontale.",
                    curves(lambda t: a * t ** 2 - 2 * a * h * t, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 2

def l2_exp(r):
    a, b = nz(r, -5, 5), r.randint(-5, 5)
    _sol = lines("(eˣ)' = eˣ et (x)' = 1", f"f'(x) = {coef(a)}eˣ {signed(b)}")
    ex = Exercise(f"Dérive f(x) = {coef(a)}eˣ {sterm(b, 'x')}.",
                    Expr(f"{a}*exp(x) + {b}"), TABLE, curves(lambda t: a * np.exp(t) + b * t, xlim=(-3, 2)))
    ex.solution = _sol
    return ex


def l2_ln(r):
    a, b = nz(r, -6, 6), r.randint(-5, 5)
    _sol = lines("(ln x)' = 1/x et (x)' = 1", f"f'(x) = {a}/x {signed(b)}")
    ex = Exercise(f"Dérive f(x) = {coef(a)}ln(x) {sterm(b, 'x')}  (x > 0).", Expr(f"{a}/x + {b}", (0.2, 4)), TABLE,
                    curves(lambda t: a * np.log(t) + b * t, xlim=(0.05, 6)))
    ex.solution = _sol
    return ex


def l2_trig(r):
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    _sol = lines("(sin x)' = cos x ;  (cos x)' = −sin x", f"f'(x) = {a}·cos(x) + {par(b)}·(−sin(x)) = {a}·cos(x) {signed(-b)}·sin(x)")
    ex = Exercise(f"Dérive f(x) = {a}·sin(x) {signed(b)}·cos(x).", Expr(f"{a}*cos(x) - {b}*sin(x)"),
                    TABLE + "\nAttention au signe : (cos x)' = −sin x.",
                    curves(lambda t: a * np.sin(t) + b * np.cos(t), xlim=(-7, 7)))
    ex.solution = _sol
    return ex


def l2_inv(r):
    a = nz(r, -9, 9)
    _sol = lines(f"{a}/x = {a}·x⁻¹", f"({a}·x⁻¹)' = {a} × (−1)·x⁻² = {fmt(-a)}/x²")
    ex = Exercise(f"Dérive f(x) = {a}/x.", Expr(f"-{a}/x**2", (0.3, 3)),
                    TABLE + "\nTu peux aussi écrire 1/x = x⁻¹ et appliquer (xⁿ)' = n·xⁿ⁻¹.",
                    curves(lambda t: a / t, xlim=(-5, 5), ylim=(-10, 10)))
    ex.solution = _sol
    return ex


def l2_sqrt_value(r):
    k = r.randint(1, 8)
    _sol = lines("(√x)' = 1 / (2√x)", f"f'({k * k}) = 1 / (2 × √{k * k}) = 1 / (2 × {k}) = 1/{2 * k}")
    ex = Exercise(f"f(x) = √x. Calcule f'({k * k}).", Number(F(1, 2 * k)), TABLE,
                    tangent(np.sqrt, lambda t: 1 / (2 * np.sqrt(t)), k * k, (0.01, k * k * 2 + 2)))
    ex.solution = _sol
    return ex


def l2_neg_power(r):
    n = r.randint(1, 4)
    _sol = lines(f"(xⁿ)' = n·xⁿ⁻¹ avec n = −{n}", f"f'(x) = −{n}·x^(−{n} − 1) = −{n}·x^(−{n + 1}) = −{n}/x^{n + 1}")
    ex = Exercise(f"Dérive f(x) = x^(−{n}).", Expr(f"-{n}*x**(-{n + 1})", (0.3, 3)),
                    "(xⁿ)' = n·xⁿ⁻¹ marche aussi pour n négatif : n = −" + str(n) + ".", None)
    ex.solution = _sol
    return ex


def l2_frac_power(r):
    p, q = r.choice([(3, 2), (5, 2), (1, 3), (4, 3), (2, 3)])
    _e = F(p, q) - 1
    _sol = lines(f"(xⁿ)' = n·xⁿ⁻¹ avec n = {p}/{q}", f"n − 1 = {p}/{q} − 1 = {fmt(_e)}", f"f'(x) = ({p}/{q})·x^({fmt(_e)})")
    ex = Exercise(f"Dérive f(x) = x^({p}/{q})  (x > 0).", Expr(f"({p}/{q})*x**({p}/{q} - 1)", (0.2, 3)),
                    "(xⁿ)' = n·xⁿ⁻¹ marche pour tout exposant réel. Ici n = " + f"{p}/{q}. Calcule n − 1.", None)
    ex.solution = _sol
    return ex


def l2_value0(r):
    a, b = nz(r, -6, 6), r.randint(-6, 6)
    _sol = lines(f"f'(x) = {coef(a)}eˣ {signed(2 * b)}x", f"f'(0) = {a} × e⁰ + {par(2 * b)} × 0 = {a} × 1 = {a}")
    ex = Exercise(f"f(x) = {coef(a)}eˣ {sterm(b, 'x²')}. Calcule f'(0).", Number(a), M_VALUE + "\nRappel : e⁰ = 1.",
                    tangent(lambda t: a * np.exp(t) + b * t ** 2, lambda t: a * np.exp(t) + 2 * b * t, 0, (-2, 2)))
    ex.solution = _sol
    return ex


def l2_tangent_line(r):
    k = nz(r, -4, 4)
    _sol = lines(f"f({k}) = {k * k} et f'(x) = 2x donc f'({k}) = {2 * k}", f"Tangente : y = f'(a)(x − a) + f(a) = {2 * k}(x − {par(k)}) + {k * k}",
                 f"y = {term(2 * k, 'x')} {signed(-2 * k * k)} + {k * k} = {lin(2 * k, -k * k)}")
    ex = Exercise(f"Donne l'équation de la tangente à f(x) = x² au point x = {k} (forme y = a*x + b).",
                    Expr(f"{2 * k}*x - {k * k}"),
                    "Tangente en a : y = f'(a)·(x − a) + f(a). Calcule f(a) et f'(a), puis développe.",
                    curves(lambda t: t ** 2, xlim=(k - 4, k + 4), points=[(k, k * k)]))
    ex.solution = _sol
    return ex


def l2_value1(r):
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    _sol = lines(f"f'(x) = {coef(a)}eˣ + {b}/x", f"f'(1) = {a}e + {b}/1 = {a}e {signed(b)} ≈ {a * math.e + b:.4f}")
    ex = Exercise(f"f(x) = {a}eˣ {signed(b)}·ln(x). Calcule f'(1) (valeur exacte, tu peux utiliser e).",
                    Number(a * math.e + b), M_VALUE + "\nRappel : e¹ = e, et 1/1 = 1.", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

def l3_prod_poly(r):
    a, b = nz(r, -6, 6), nz(r, -6, 6)
    _sol = lines(f"u = x {signed(a)}, v = x {signed(b)} ; u' = 1, v' = 1", f"(uv)' = u'v + uv' = (x {signed(b)}) + (x {signed(a)})",
                 f"f'(x) = {lin(2, a + b)}")
    ex = Exercise(f"Dérive f(x) = (x {signed(a)})(x {signed(b)}).", Expr(f"2*x + {a + b}"),
                    M_PROD + "\n(Ou développe d'abord, puis dérive : les deux méthodes donnent le même résultat.)", None)
    ex.solution = _sol
    return ex


def l3_xexp(r):
    a = nz(r, -4, 4)
    _sol = lines(f"u = {term(a, 'x')}, v = eˣ ; u' = {a}, v' = eˣ", f"(uv)' = {a}·eˣ + {term(a, 'x')}·eˣ", f"f'(x) = {coef(a)}eˣ(1 + x)")
    ex = Exercise(f"Dérive f(x) = {coef(a)}x·eˣ.", Expr(f"{a}*(exp(x) + x*exp(x))"), M_PROD,
                    curves(lambda t: a * t * np.exp(t), xlim=(-5, 2)))
    ex.solution = _sol
    return ex


def l3_x2ln(r):
    _sol = lines("u = x², v = ln x ; u' = 2x, v' = 1/x", "(uv)' = 2x·ln x + x²·(1/x)", "f'(x) = 2x·ln x + x")
    ex = Exercise("Dérive f(x) = x²·ln(x)  (x > 0).", Expr("2*x*log(x) + x", (0.2, 4)), M_PROD,
                    curves(lambda t: t ** 2 * np.log(t), xlim=(0.01, 3)))
    ex.solution = _sol
    return ex


def l3_xsin(r):
    fn = r.choice(["sin", "cos"])
    target = "sin(x) + x*cos(x)" if fn == "sin" else "cos(x) - x*sin(x)"
    _sol = (lines("u = x, v = sin x ; u' = 1, v' = cos x", "(uv)' = 1·sin x + x·cos x", "f'(x) = sin x + x·cos x") if fn == "sin"
            else lines("u = x, v = cos x ; u' = 1, v' = −sin x", "(uv)' = 1·cos x + x·(−sin x)", "f'(x) = cos x − x·sin x"))
    ex = Exercise(f"Dérive f(x) = x·{fn}(x).", Expr(target), M_PROD,
                    curves(lambda t: t * (np.sin(t) if fn == "sin" else np.cos(t)), xlim=(-8, 8)))
    ex.solution = _sol
    return ex


def l3_quot_lin(r):
    a, b = r.sample(range(-6, 7), 2)
    _sol = lines(f"u = x {signed(a)}, v = x {signed(b)} ; u' = 1, v' = 1", f"(u/v)' = (u'v − uv') / v² = [(x {signed(b)}) − (x {signed(a)})] / (x {signed(b)})²",
                 f"f'(x) = {b - a} / (x {signed(b)})²")
    ex = Exercise(f"Dérive f(x) = (x {signed(a)}) / (x {signed(b)}).", Expr(f"({b} - {a}) / (x + {b})**2"), M_QUOT,
                    curves(lambda t: (t + a) / (t + b), xlim=(-b - 6, -b + 6), ylim=(-10, 10)))
    ex.solution = _sol
    return ex


def l3_exp_over_x(r):
    _sol = lines("u = eˣ, v = x ; u' = eˣ, v' = 1", "(u/v)' = (eˣ·x − eˣ·1) / x²", "f'(x) = eˣ(x − 1) / x²")
    ex = Exercise("Dérive f(x) = eˣ / x.", Expr("exp(x)*(x - 1)/x**2", (0.3, 3)), M_QUOT,
                    curves(lambda t: np.exp(t) / t, xlim=(0.1, 3), ylim=(0, 12)))
    ex.solution = _sol
    return ex


def l3_value_prod(r):
    k = r.randint(-2, 2)
    _sol = lines("f'(x) = 2x·eˣ + x²·eˣ = eˣ(x² + 2x)", f"f'({k}) = e^{k} × ({k * k} {signed(2 * k)}) = {k * k + 2 * k}·e^{k} ≈ {(k * k + 2 * k) * math.exp(k):.4f}")
    ex = Exercise(f"f(x) = x²·eˣ. Calcule f'({k}) (exact avec e, ou décimal précis).",
                    Number((2 * k + k * k) * math.exp(k)), M_PROD, None)
    ex.solution = _sol
    return ex


def l3_lnx_over_x(r):
    k = r.choice([1, 2, 3, 4])
    _sol = lines("u = ln x, v = x : (u/v)' = ((1/x)·x − ln x·1) / x² = (1 − ln x) / x²",
                 f"f'({k}) = (1 − ln {k}) / {k * k} ≈ (1 − {math.log(k):.4f}) / {k * k} ≈ {(1 - math.log(k)) / k ** 2:.4f}")
    ex = Exercise(f"f(x) = ln(x) / x. Calcule f'({k}) (exact ou décimal précis).", Number((1 - math.log(k)) / k ** 2),
                    M_QUOT, curves(lambda t: np.log(t) / t, xlim=(0.3, 8)))
    ex.solution = _sol
    return ex


def l3_inv_quad(r):
    a = r.randint(1, 9)
    _sol = lines(f"u = 1, v = x² + {a} ; u' = 0, v' = 2x", f"(u/v)' = (0 × (x² + {a}) − 1 × 2x) / (x² + {a})²", f"f'(x) = −2x / (x² + {a})²")
    ex = Exercise(f"Dérive f(x) = 1 / (x² + {a}).", Expr(f"-2*x/(x**2 + {a})**2"),
                    M_QUOT + "\nIci le numérateur est constant : u' = 0.",
                    curves(lambda t: 1 / (t ** 2 + a), xlim=(-5, 5)))
    ex.solution = _sol
    return ex


def l3_x3exp(r):
    _sol = lines("u = x³, v = eˣ ; u' = 3x², v' = eˣ", "(uv)' = 3x²·eˣ + x³·eˣ", "f'(x) = eˣ(x³ + 3x²)")
    ex = Exercise("Dérive f(x) = x³·eˣ.", Expr("3*x**2*exp(x) + x**3*exp(x)"), M_PROD,
                    curves(lambda t: t ** 3 * np.exp(t), xlim=(-5, 1.5)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 4

def l4_power_lin(r):
    a, b, n = nz(r, -4, 4), nz(r, -5, 5), r.randint(2, 5)
    _sol = lines(f"u = {lin(a, b)}, u' = {a} ; (uⁿ)' = n·u'·uⁿ⁻¹ avec n = {n}", f"f'(x) = {n} × {par(a)} × ({lin(a, b)})^{n - 1} = {n * a}({lin(a, b)})^{n - 1}")
    ex = Exercise(f"Dérive f(x) = ({lin(a, b)})^{n}.", Expr(f"{n}*{a}*({a}*x + {b})**{n - 1}"), M_CHAIN, None)
    ex.solution = _sol
    return ex


def l4_exp_ax(r):
    a, b = nz(r, -5, 5), r.randint(-4, 4)
    _sol = lines(f"u = {lin(a, b)}, u' = {a} ; (eᵘ)' = u'·eᵘ", f"f'(x) = {coef(a)}e^({lin(a, b)})")
    ex = Exercise(f"Dérive f(x) = e^({lin(a, b)}).", Expr(f"{a}*exp({a}*x + {b})"), M_CHAIN,
                    curves(lambda t: np.exp(a * t + b), xlim=(-2, 2), ylim=(-1, 20)))
    ex.solution = _sol
    return ex


def l4_sin_ax(r):
    a, b = nz(r, -5, 5), r.randint(-3, 3)
    fn = r.choice(["sin", "cos"])
    target = f"{a}*cos({a}*x + {b})" if fn == "sin" else f"-{a}*sin({a}*x + {b})"
    _sol = (lines(f"u = {lin(a, b)}, u' = {a} ; (sin u)' = u'·cos u", f"f'(x) = {a}·cos({lin(a, b)})") if fn == "sin"
            else lines(f"u = {lin(a, b)}, u' = {a} ; (cos u)' = −u'·sin u", f"f'(x) = {fmt(-a)}·sin({lin(a, b)})"))
    ex = Exercise(f"Dérive f(x) = {fn}({lin(a, b)}).", Expr(target), M_CHAIN, None)
    ex.solution = _sol
    return ex


def l4_ln_ax(r):
    a, b = r.randint(1, 6), r.randint(1, 6)
    _sol = lines(f"u = {lin(a, b)}, u' = {a} ; (ln u)' = u'/u", f"f'(x) = {a} / ({lin(a, b)})")
    ex = Exercise(f"Dérive f(x) = ln({lin(a, b)}).", Expr(f"{a}/({a}*x + {b})", (0.1, 4)), M_CHAIN, None)
    ex.solution = _sol
    return ex


def l4_sqrt_ax(r):
    a, b = r.randint(1, 6), r.randint(1, 9)
    _sol = lines(f"u = {lin(a, b)}, u' = {a} ; (√u)' = u' / (2√u)", f"f'(x) = {a} / (2√({lin(a, b)}))" + (f" = {fmt(F(a, 2))}/√({lin(a, b)})" if a % 2 == 0 else ""))
    ex = Exercise(f"Dérive f(x) = √({lin(a, b)}).", Expr(f"{a}/(2*sqrt({a}*x + {b}))", (0.1, 4)), M_CHAIN, None)
    ex.solution = _sol
    return ex


def l4_extremum_x(r):
    a, h = nz(r, -3, 3), r.randint(-6, 6)
    b, c = -2 * a * h, r.randint(-9, 9)
    _sol = lines(f"f'(x) = {lin(2 * a, b)}", f"f'(x) = 0 ⇔ {term(2 * a, 'x')} = {fmt(-b)} ⇔ x = {fmt(-b)} / {par(2 * a)} = {h}")
    ex = Exercise(f"f(x) = {poly({2: a, 1: b, 0: c})}. En quel x f atteint-elle son extremum ?", Number(h),
                    "Un extremum (min ou max) se trouve là où f'(x) = 0 (tangente horizontale). Résous f'(x) = 0.",
                    curves(lambda t: a * t ** 2 + b * t + c, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l4_critical_cubic(r):
    a = r.randint(1, 5)
    _sol = lines(f"f'(x) = 3x² − {3 * a * a}", f"3x² = {3 * a * a} ⇔ x² = {a * a} ⇔ x = {a} ou x = −{a}")
    ex = Exercise(f"Trouve les points critiques (f'(x) = 0) de f(x) = x³ − {3 * a * a}x.", Numbers([a, -a]),
                    "Dérive, puis résous f'(x) = 0 (équation du second degré très simple).",
                    curves(lambda t: t ** 3 - 3 * a * a * t, xlim=(-2.2 * a, 2.2 * a)))
    ex.solution = _sol
    return ex


def l4_nature(r):
    a = nz(r, -4, 4)
    b, c = r.randint(-8, 8), r.randint(-9, 9)
    _sol = lines(f"f'(x) = {lin(2 * a, b)} et f''(x) = {2 * a}",
                 f"f'' {'> 0 : la courbe est un bol (∪), c est un minimum' if a > 0 else '< 0 : la courbe est un dôme (∩), c est un maximum'}.".replace("c est", "c'est"))
    ex = Exercise(f"f(x) = {poly({2: a, 1: b, 0: c})} a un seul extremum. Est-ce un minimum ou un maximum ?",
                    Choice("minimum" if a > 0 else "maximum", ["minimum", "maximum"]),
                    "Regarde comment f' change de signe autour du point critique : − puis + → minimum ; + puis − → "
                    "maximum. (Ou le signe de f'' : f'' > 0 → minimum.)", None)
    ex.solution = _sol
    return ex


def l4_max_value(r):
    h, k = r.randint(-5, 5), r.randint(-9, 9)
    _sol = lines(f"f'(x) = −2x {signed(2 * h)}", f"f'(x) = 0 ⇔ x = {h}", f"f({h}) = −{par(h)}² + {2 * h} × {par(h)} {signed(k - h * h)} = {k}")
    ex = Exercise(f"Quelle est la valeur maximale de f(x) = {poly({2: -1, 1: 2 * h, 0: k - h * h})} ?", Number(k),
                    "1. Trouve x tel que f'(x) = 0.  2. Calcule f en ce x : c'est la valeur maximale.",
                    curves(lambda t: -(t - h) ** 2 + k, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l4_exp_x2(r):
    a = nz(r, -3, 3)
    _sol = lines(f"u = {poly({2: a})}, u' = {term(2 * a, 'x')} ; (eᵘ)' = u'·eᵘ", f"f'(x) = {term(2 * a, 'x')}·e^({poly({2: a})})")
    ex = Exercise(f"Dérive f(x) = e^({poly({2: a})}).", Expr(f"{2 * a}*x*exp({a}*x**2)"), M_CHAIN,
                    curves(lambda t: np.exp(a * t ** 2), xlim=(-2, 2), ylim=(-1, 10)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 5

def l5_second(r):
    a, b, c = nz(r, -4, 4), r.randint(-6, 6), r.randint(-9, 9)
    _sol = lines(f"f'(x) = {poly({3: 4 * a, 2: 3 * b, 0: c})}", f"f''(x) = {poly({2: 12 * a, 1: 6 * b})}")
    ex = Exercise(f"Calcule la dérivée seconde f''(x) de f(x) = {poly({4: a, 3: b, 1: c})}.",
                    Expr(f"{12 * a}*x**2 + {6 * b}*x"), "La dérivée seconde est la dérivée de la dérivée : dérive deux fois.",
                    None)
    ex.solution = _sol
    return ex


def l5_second_value(r):
    k = r.randint(-2, 2)
    _sol = lines("f'(x) = eˣ + 3x² ; f''(x) = eˣ + 6x", f"f''({k}) = e^{k} {signed(6 * k)} ≈ {math.exp(k) + 6 * k:.4f}")
    ex = Exercise(f"f(x) = eˣ + x³. Calcule f''({k}) (exact avec e, ou décimal précis).",
                    Number(math.exp(k) + 6 * k), "Dérive deux fois, puis remplace x.", None)
    ex.solution = _sol
    return ex


def l5_inflection(r):
    b = r.choice([-9, -6, -3, 3, 6, 9])
    c = r.randint(-5, 5)
    _sol = lines(f"f'(x) = 3x² {signed(2 * b)}x {signed(c)}", f"f''(x) = 6x {signed(2 * b)}", f"f''(x) = 0 ⇔ x = {fmt(-2 * b)} / 6 = {fmt(F(-b, 3))}")
    ex = Exercise(f"f(x) = {poly({3: 1, 2: b, 1: c})}. En quel x se trouve le point d'inflexion ?", Number(-b / 3),
                    "Point d'inflexion : la courbure change de sens, f''(x) = 0 (et change de signe). Dérive deux fois.",
                    curves(lambda t: t ** 3 + b * t ** 2 + c * t, xlim=(-b / 3 - 6, -b / 3 + 6), ylim=(-150, 150)))
    ex.solution = _sol
    return ex


def l5_linear_approx(r):
    a = r.randint(2, 9)
    d = r.choice([0.1, 0.2, 0.5, -0.3, 1])
    _sol = lines(f"f({a * a}) = √{a * a} = {a}", f"f'(x) = 1/(2√x) donc f'({a * a}) = 1/(2 × {a}) = 1/{2 * a}",
                 f"√({a * a} + {d}) ≈ {a} + {d} / {2 * a} ≈ {a + d / (2 * a):.5f}", f"(Vraie valeur : {math.sqrt(a * a + d):.5f})")
    ex = Exercise(f"Approximation linéaire : estime √({a * a} + {d}) avec f(a) + f'(a)·δ où f(x) = √x, a = {a * a}, "
                    f"δ = {d}.", Number(a + d / (2 * a), 1e-4),
                    "Près de a, une fonction ressemble à sa tangente : f(a + δ) ≈ f(a) + f'(a)·δ. "
                    "Calcule f(a), f'(a) = 1/(2√a), puis la formule.",
                    tangent(np.sqrt, lambda t: 1 / (2 * np.sqrt(t)), a * a, (0.1, 2 * a * a)))
    ex.solution = _sol
    return ex


def l5_numeric(r):
    k, h = r.randint(1, 3), r.choice([0.1, 0.5, 1])
    val = ((k + h) ** 3 - (k - h) ** 3) / (2 * h)
    _sol = lines(f"f({fmt(k + h)}) = {fmt((k + h) ** 3)} et f({fmt(k - h)}) = {fmt((k - h) ** 3)}",
                 f"({fmt((k + h) ** 3)} − {par((k - h) ** 3)}) / {fmt(2 * h)} = {fmt(val)}", f"Vraie dérivée : f'({k}) = 3 × {k}² = {3 * k * k}")
    ex = Exercise(f"Dérivée numérique : calcule (f({k} + {h}) − f({k} − {h})) / (2·{h}) pour f(x) = x³.\n"
                    f"Compare avec la vraie valeur f'({k}).", Number(val, 1e-4),
                    "Calcule les deux images, soustrais, divise. C'est la « différence centrée » utilisée pour "
                    "vérifier un gradient calculé à la main (gradient checking).",
                    tangent(lambda t: t ** 3, lambda t: 3 * t ** 2, k, (k - 2, k + 2)))
    ex.solution = _sol
    return ex


def l5_newton(r):
    a = r.choice([2, 3, 5, 7, 10])
    x0 = r.choice([1, 2, 3])
    x1 = x0 - (x0 * x0 - a) / (2 * x0)
    _sol = lines(f"f({x0}) = {x0}² − {a} = {x0 * x0 - a} ; f'({x0}) = 2 × {x0} = {2 * x0}",
                 f"x₁ = {x0} − ({x0 * x0 - a}) / {2 * x0} = {fmt(F(x0 * x0 + a, 2 * x0))} ≈ {x1:.5f}", f"(√{a} ≈ {math.sqrt(a):.5f})")
    ex = Exercise(f"Méthode de Newton pour résoudre x² − {a} = 0 : fais UNE itération x₁ = x₀ − f(x₀)/f'(x₀) avec x₀ = {x0}.",
                    Number(x1), "f(x) = x² − " + str(a) + ", f'(x) = 2x. Remplace x₀ dans la formule. "
                    "(Géométriquement : on suit la tangente jusqu'à l'axe horizontal.)",
                    tangent(lambda t: t ** 2 - a, lambda t: 2 * t, x0, (0, 4)))
    ex.solution = _sol
    return ex


def l5_velocity(r):
    a, b, t = nz(r, -5, 5), r.randint(-9, 9), r.randint(0, 6)
    _sol = lines(f"v(t) = s'(t) = {lin(2 * a, b, 't')}", f"v({t}) = {2 * a} × {t} {signed(b)} = {2 * a * t + b} m/s")
    ex = Exercise(f"La position d'un objet est s(t) = {poly({2: a, 1: b}, 't')} (mètres). Vitesse à t = {t} s ?",
                    Number(2 * a * t + b), "La vitesse est la dérivée de la position : v(t) = s'(t).",
                    curves(lambda u: a * u ** 2 + b * u, xlim=(0, 8)))
    ex.solution = _sol
    return ex


def l5_accel(r):
    a, b, c = nz(r, -3, 3), r.randint(-6, 6), r.randint(-9, 9)
    t = r.randint(0, 5)
    _sol = lines(f"v(t) = {poly({2: 3 * a, 1: 2 * b, 0: c}, 't')}", f"a(t) = v'(t) = {lin(6 * a, 2 * b, 't')}",
                 f"a({t}) = {6 * a} × {t} {signed(2 * b)} = {6 * a * t + 2 * b}")
    ex = Exercise(f"s(t) = {poly({3: a, 2: b, 1: c}, 't')}. Accélération à t = {t} ?", Number(6 * a * t + 2 * b),
                    "Accélération = dérivée de la vitesse = dérivée seconde de la position.", None)
    ex.solution = _sol
    return ex


def l5_ax(r):
    a = r.choice([2, 3, 10])
    _sol = lines(f"{a}ˣ = e^(x·ln {a}), donc ({a}ˣ)' = ln({a})·{a}ˣ", f"f'(0) = ln({a}) × {a}⁰ = ln {a} ≈ {math.log(a):.4f}")
    ex = Exercise(f"f(x) = {a}ˣ. Calcule f'(0) (exact avec ln, ou décimal précis).", Number(math.log(a)),
                    "aˣ = e^(x·ln a). Dérive avec la règle de la chaîne : (aˣ)' = ln(a)·aˣ.",
                    tangent(lambda t: float(a) ** t, lambda t: math.log(a) * float(a) ** t, 0, (-2, 2)))
    ex.solution = _sol
    return ex


def l5_convex(r):
    kind = r.choice(["convexe", "concave"])
    expr = {"convexe": r.choice(["eˣ", "x² + 3x", "x⁴", "−ln(x)"]),
            "concave": r.choice(["ln(x)", "√x", "−x²", "−eˣ"])}[kind]
    _d2 = {"eˣ": "eˣ > 0", "x² + 3x": "2 > 0", "x⁴": "12x² ≥ 0", "−ln(x)": "1/x² > 0",
           "ln(x)": "−1/x² < 0", "√x": "−1/(4x^(3/2)) < 0", "−x²": "−2 < 0", "−eˣ": "−eˣ < 0"}[expr]
    _sol = lines(f"f''(x) = {_d2}", f"Donc f est {kind}.")
    ex = Exercise(f"Sur son domaine, f(x) = {expr} est-elle convexe ou concave ?",
                    Choice(kind, ["convexe", "concave"]),
                    "Calcule f''(x). f'' ≥ 0 partout → convexe (forme de bol, ∪) ; f'' ≤ 0 → concave (∩).\n"
                    "Les fonctions de coût convexes ont un seul minimum : c'est ce qu'on aime en optimisation.", None)
    ex.solution = _sol
    return ex


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
