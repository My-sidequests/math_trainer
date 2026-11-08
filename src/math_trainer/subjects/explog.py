"""Exponentielle & logarithme : puissances, exp, ln, équations, fonctions de l'IA."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Number, Numbers
from ..core import BLUE, Exercise, curves, nz, num, par, sup

from ..core import fmt, lin, lines, par, steps_lin  # noqa: E402,F811

DESCRIPTIONS = {
    1: "Puissances et règles de calcul",
    2: "La fonction exponentielle, croissance et décroissance",
    3: "Logarithmes : définition et règles",
    4: "Équations exponentielles et logarithmiques, applications",
    5: "Vers l'IA : sigmoïde, softmax, log-loss, entropie croisée",
}

CALC = "\n(calculatrice ou Python autorisés ; 3 chiffres significatifs suffisent)"
M_POW = ("Règles des puissances (même base a) :\n"
         "  aᵐ × aⁿ = aᵐ⁺ⁿ      aᵐ / aⁿ = aᵐ⁻ⁿ      (aᵐ)ⁿ = aᵐˣⁿ\n"
         "  a⁰ = 1      a⁻ⁿ = 1 / aⁿ      a^(1/n) = racine n-ième de a      (ab)ⁿ = aⁿbⁿ")
M_LN = ("ln est la réciproque de exp : ln(eˣ) = x et e^(ln x) = x (x > 0).\n"
        "Règles : ln(ab) = ln a + ln b ;  ln(a/b) = ln a − ln b ;  ln(aⁿ) = n·ln a ;  ln 1 = 0 ;  ln e = 1.\n"
        "Plus généralement log_b(x) = « la puissance à laquelle élever b pour obtenir x » = ln x / ln b.")


def exp_curve(xlim=(-3, 3), ylim=None, extra=None, labels=None):
    fs = [np.exp] + ([extra] if extra else [])
    return curves(*fs, xlim=xlim, ylim=ylim, labels=labels)


# ================================================================ NIVEAU 1

def l1_power(r):
    a, n = r.randint(2, 5), r.randint(2, 4)
    if a == 5 and n == 4:
        n = 3
    s = r.choice([1, -1])
    _prod = " × ".join([par(s * a)] * n)
    _sol = lines(f"{par(s * a)}{sup(n)} = {_prod}", f"= {fmt((s * a) ** n)}" + (" (exposant pair : le signe disparaît)" if s < 0 and n % 2 == 0 else ""))
    ex = Exercise(f"Calcule {par(s * a)}{sup(n)}.", Number((s * a) ** n),
                    "aⁿ = a × a × ... × a (n fois). Un négatif à une puissance paire devient positif.",
                    curves(lambda t: t ** n, xlim=(-a - 1, a + 1)))
    ex.solution = _sol
    return ex


def _exp_rule(r, kind):
    a, m, n = r.choice([2, 3, 5, 10, 7]), r.randint(-4, 7), r.randint(-4, 7)
    if kind == "mul":
        txt, res = f"{a}{sup(m)} × {a}{sup(n)}", m + n
    elif kind == "div":
        txt, res = f"{a}{sup(m)} / {a}{sup(n)}", m - n
    else:
        txt, res = f"({a}{sup(m)}){sup(n)}", m * n
    _rule = {"mul": f"aᵐ × aⁿ = aᵐ⁺ⁿ : on ADDITIONNE les exposants : {m} + {par(n)} = {res}",
             "div": f"aᵐ / aⁿ = aᵐ⁻ⁿ : on SOUSTRAIT les exposants : {m} − {par(n)} = {res}",
             "pow": f"(aᵐ)ⁿ = aᵐˣⁿ : on MULTIPLIE les exposants : {m} × {par(n)} = {res}"}[kind]
    _sol = lines(_rule, f"{txt} = {a}{sup(res)}, donc k = {res}")
    ex = Exercise(f"Écris {txt} sous la forme {a}ᵏ. Que vaut k ?", Number(res), M_POW,
                    curves(lambda t: float(a) ** t, xlim=(-2, 3), labels=[f"y = {a}ˣ"]))
    ex.solution = _sol
    return ex


def l1_mul(r):
    return _exp_rule(r, "mul")


def l1_div(r):
    return _exp_rule(r, "div")


def l1_powpow(r):
    return _exp_rule(r, "pow")


def l1_neg_exp(r):
    a, n = r.randint(2, 6), r.randint(0, 3)
    _sol = lines("a⁻ⁿ = 1 / aⁿ", f"{a}{sup(-n)} = 1 / {a}{sup(n)} = 1 / {a ** n} = {fmt(F(1, a ** n))}" if n else f"{a}⁰ = 1 (toute puissance 0 vaut 1)")
    ex = Exercise(f"Calcule {a}{sup(-n)} (fraction).", Number(F(1, a ** n)), M_POW,
                    curves(lambda t: a ** t, xlim=(-4, 2), points=[(-n, a ** -n)]))
    ex.solution = _sol
    return ex


def l1_scientific(r):
    m, k = r.choice([1.5, 2.3, 3.2, 4.7, 6.02, 9.1]), r.randint(-4, 5)
    _sol = lines(f"× 10{sup(k)} : on décale la virgule de {abs(k)} rang(s) vers la {'droite' if k >= 0 else 'gauche'}.",
                 f"{m} × 10{sup(k)} = {m * 10 ** k:g}")
    ex = Exercise(f"Écris {m} × 10{sup(k)} en nombre décimal.", Number(m * 10 ** k, 1e-9),
                    "Multiplier par 10ᵏ décale la virgule de k rangs vers la droite (vers la gauche si k < 0).",
                    curves(lambda t: 10.0 ** t, xlim=(-2, 2), labels=["y = 10ˣ"]))
    ex.solution = _sol
    return ex


def l1_sqrt_pow(r):
    k = r.randint(2, 15)
    _sol = lines("a^(1/2) = √a", f"√{k * k} = {k} car {k}² = {k * k}")
    ex = Exercise(f"Calcule {k * k}^(1/2).", Number(k), M_POW + "\na^(1/2) = √a.",
                    curves(lambda t: np.sqrt(t), xlim=(0, 230)))
    ex.solution = _sol
    return ex


def l1_frac_pow(r):
    base, n, m = r.choice([(8, 3, 2), (27, 3, 2), (16, 4, 3), (32, 5, 2), (4, 2, 3), (64, 3, 2), (81, 4, 3)])
    _root = round(base ** (1 / n))
    _rn = {2: "carrée", 3: "cubique"}.get(n, f"{n}-ième")
    _sol = lines(f"{base}^({m}/{n}) = (racine {_rn} de {base})^{m}", f"Racine {_rn} de {base} : {_root}, car {_root}{sup(n)} = {base}",
                 f"{_root}{sup(m)} = {_root ** m}")
    ex = Exercise(f"Calcule {base}^({m}/{n}).", Number(round(base ** (m / n))),
                    M_POW + f"\na^(m/n) = (racine n-ième de a)^m : commence par la racine {n}-ième de {base}.",
                    curves(lambda t: np.power(t, m / n), xlim=(0, base * 1.2), labels=[f"y = x^({m}/{n})"]))
    ex.solution = _sol
    return ex


def l1_solve_pow(r):
    a, k = r.choice([2, 3, 5]), r.randint(-3, 6)
    _sol = lines(f"{num(F(a) ** k)} = {a}{sup(k)}", f"{a}ˣ = {a}{sup(k)} ⇒ x = {k} (mêmes bases : on identifie les exposants)")
    ex = Exercise(f"Résous {a}ˣ = {num(F(a) ** k)}.", Number(k),
                    "Écris le membre de droite comme une puissance de " + str(a) + ", puis identifie les exposants.",
                    curves(lambda t: float(a) ** t, lambda t: float(F(a) ** k) + 0 * t, xlim=(-4, 7),
                           ylim=(-1, float(F(a) ** k) * 1.5 + 1)))
    ex.solution = _sol
    return ex


def l1_product_pow(r):
    a, b, n = r.randint(2, 5), r.randint(2, 5), r.randint(2, 3)
    _sol = lines(f"({a} × {b}){sup(n)} = {a}{sup(n)} × {b}{sup(n)} = {a ** n} × {b ** n} = {(a * b) ** n}",
                 f"(Vérification directe : {a * b}{sup(n)} = {(a * b) ** n})")
    ex = Exercise(f"Calcule ({a} × {b}){sup(n)} en utilisant (ab)ⁿ = aⁿbⁿ.", Number((a * b) ** n), M_POW,
                    curves(lambda t: t ** n, xlim=(0, a * b + 1), labels=[f"y = x{sup(n)}"]))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 2

M_EXP = ("e ≈ 2.718 est une constante. exp(x) = eˣ obéit aux mêmes règles que les puissances :\n"
         "  eᵃ·eᵇ = eᵃ⁺ᵇ,  eᵃ/eᵇ = eᵃ⁻ᵇ,  e⁰ = 1,  eˣ > 0 pour tout x.\n"
         "eˣ = eʸ ⇔ x = y (exp est strictement croissante).")


def l2_exp_mul(r):
    a, b = r.randint(-5, 9), r.randint(-5, 9)
    _sol = lines("eᵃ × eᵇ = eᵃ⁺ᵇ", f"e{sup(a)} × e{sup(b)} = e^({a} + {par(b)}) = e{sup(a + b)}, donc k = {a + b}")
    ex = Exercise(f"Écris e{sup(a)} × e{sup(b)} sous la forme eᵏ. Que vaut k ?", Number(a + b), M_EXP, exp_curve())
    ex.solution = _sol
    return ex


def l2_exp_div(r):
    a, b = r.randint(-5, 9), r.randint(-5, 9)
    _sol = lines("eᵃ / eᵇ = eᵃ⁻ᵇ", f"e{sup(a)} / e{sup(b)} = e^({a} − {par(b)}) = e{sup(a - b)}, donc k = {a - b}")
    ex = Exercise(f"Écris e{sup(a)} / e{sup(b)} sous la forme eᵏ. Que vaut k ?", Number(a - b), M_EXP, exp_curve())
    ex.solution = _sol
    return ex


def l2_exp_value(r):
    x = r.choice([-2, -1, -0.5, 0.5, 1, 1.5, 2, 3])
    _sol = lines(f"e ≈ 2,71828. e^({fmt(x)}) se calcule à la calculatrice (touche eˣ) ou en Python : math.exp({x})",
                 f"e^({fmt(x)}) ≈ {math.exp(x):.4f}")
    ex = Exercise(f"Calcule e^({num(x)})." + CALC, Number(math.exp(x), 5e-3),
                    "Calculatrice (touche eˣ) ou Python : math.exp(...).",
                    curves(np.exp, xlim=(-3, 3.5), points=[(x, math.exp(x))]))
    ex.solution = _sol
    return ex


def l2_exp_eq(r):
    a, b, c = nz(r, -4, 4), r.randint(-6, 6), r.randint(-6, 6)
    _sol = lines("e^A = e^B ⇔ A = B : on identifie les exposants.", [f"{lin(a, b)} = {c}"] + steps_lin(a, b, c))
    ex = Exercise(f"Résous e^({a}x {'+' if b >= 0 else '−'} {abs(b)}) = e^({c}).".replace("1x", "x"),
                    Number(F(c - b, a)), M_EXP + "\nIdentifie les exposants, puis résous l'équation du premier degré.",
                    curves(lambda t: np.exp(a * t + b), lambda t: math.exp(c) + 0 * t, xlim=(-6, 6),
                           ylim=(-1, max(5, math.exp(c) * 2)), labels=["membre de gauche", "membre de droite"]))
    ex.solution = _sol
    return ex


def l2_doubling(r):
    p0, t = r.choice([100, 50, 200, 1000]), r.randint(2, 8)
    _sol = lines(f"Après t heures : P(t) = {p0} × 2ᵗ", f"P({t}) = {p0} × 2{sup(t)} = {p0} × {2 ** t} = {p0 * 2 ** t}")
    ex = Exercise(f"Une population de {p0} bactéries double toutes les heures. Combien y en a-t-il après {t} heures ?",
                    Number(p0 * 2 ** t),
                    "Doubler t fois = multiplier par 2 × 2 × ... × 2 = 2ᵗ. Population(t) = P₀ × 2ᵗ.",
                    curves(lambda x: p0 * 2 ** x, xlim=(0, t + 1)))
    ex.solution = _sol
    return ex


def l2_interest(r):
    c, rate, n = r.choice([1000, 2000, 5000]), r.choice([2, 3, 4, 5, 8]), r.randint(3, 15)
    _sol = lines(f"Chaque année : × (1 + {rate}/100) = × {1 + rate / 100:g}", f"Après {n} ans : {c} × {1 + rate / 100:g}{sup(n)}",
                 f"= {c} × {(1 + rate / 100) ** n:.4f} ≈ {c * (1 + rate / 100) ** n:.2f} €")
    ex = Exercise(f"On place {c} € à {rate} % par an (intérêts composés). Capital après {n} ans ?" + CALC,
                    Number(c * (1 + rate / 100) ** n, 1e-3),
                    "Augmenter de r % = multiplier par (1 + r/100). Répété n fois : C × (1 + r/100)ⁿ.",
                    curves(lambda x: c * (1 + rate / 100) ** x, xlim=(0, n + 2)))
    ex.solution = _sol
    return ex


def l2_exp_ln(r):
    a = r.choice([2, 3, 5, 7, 0.5, 10])
    _sol = lines("exp et ln sont réciproques : e^(ln a) = a pour tout a > 0.", f"e^(ln {fmt(a)}) = {fmt(a)}")
    ex = Exercise(f"Simplifie e^(ln {num(a)}).", Number(a), M_LN,
                    curves(np.exp, np.log, lambda t: t, xlim=(-3, 4), ylim=(-3, 4), labels=["exp", "ln", "y = x"],
                           equal=True))
    ex.solution = _sol
    return ex


def l2_halflife(r):
    n0, T = r.choice([800, 1000, 1600, 6400]), r.choice([5, 8, 10, 30])
    k = r.randint(1, 5)
    _sol = lines(f"{k * T} ans = {k} demi-vies de {T} ans.", f"À chaque demi-vie la quantité est divisée par 2 : {n0} / 2{sup(k)} = {n0} / {2 ** k}",
                 f"= {fmt(n0 / 2 ** k)} g")
    ex = Exercise(f"Une substance a une demi-vie de {T} ans. On part de {n0} g. Combien en reste-t-il après {k * T} ans ?",
                    Number(n0 / 2 ** k),
                    "À chaque demi-vie, la quantité est divisée par 2. Compte combien de demi-vies se sont écoulées : "
                    "N = N₀ × (1/2)^(t/T).", curves(lambda x: n0 * 0.5 ** (x / T), xlim=(0, (k + 1) * T)))
    ex.solution = _sol
    return ex


def l2_exp_one(r):
    a, b = nz(r, -5, 5), r.randint(-9, 9)
    _sol = lines("1 = e⁰, donc l'exposant doit valoir 0.", [f"{lin(a, b)} = 0"] + steps_lin(a, b, 0))
    ex = Exercise(f"Résous e^({a}x {'+' if b >= 0 else '−'} {abs(b)}) = 1.".replace("(1x", "(x"), Number(F(-b, a)),
                    "1 = e⁰. Donc l'exposant doit valoir 0.",
                    curves(lambda t: np.exp(a * t + b), lambda t: 1 + 0 * t, xlim=(-8, 8), ylim=(-1, 5)))
    ex.solution = _sol
    return ex


def l2_exp_eq2(r):
    k = r.randint(-6, 6)
    _sol = lines("On identifie les exposants : 2x = x " + (f"+ {k}" if k >= 0 else f"− {-k}"), f"On soustrait x : x = {k}")
    ex = Exercise(f"Résous e^(2x) = e^(x {'+' if k >= 0 else '−'} {abs(k)}).", Number(k),
                    M_EXP + "\nIdentifie les exposants.",
                    curves(lambda t: np.exp(2 * t), lambda t: np.exp(t + k), xlim=(k - 3, k + 2), labels=["e^(2x)", "e^(x + k)"]))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

def l3_log10(r):
    n = r.randint(-3, 6)
    _sol = lines(f"{num(F(10) ** n)} = 10{sup(n)}", f"log₁₀(10{sup(n)}) = {n}")
    ex = Exercise(f"Calcule log₁₀({num(F(10) ** n)}).", Number(n), M_LN,
                    curves(np.log10, xlim=(0.001, 20)))
    ex.solution = _sol
    return ex


def l3_ln_e(r):
    k = r.randint(-6, 9)
    _sol = lines("ln est la réciproque de exp : ln(eᵏ) = k", f"ln(e{sup(k)}) = {k}")
    ex = Exercise(f"Calcule ln(e{sup(k)}).", Number(k), M_LN, curves(np.log, xlim=(0.01, 10), ylim=(-5, 3)))
    ex.solution = _sol
    return ex


def l3_log2(r):
    n = r.randint(-3, 10)
    _sol = lines(f"{num(F(2) ** n)} = 2{sup(n)}", f"log₂(2{sup(n)}) = {n}")
    ex = Exercise(f"Calcule log₂({num(F(2) ** n)}).", Number(n),
                    M_LN + "\nlog₂ est très utilisé en informatique : log₂(1024) = 10 car 2¹⁰ = 1024.",
                    curves(np.log2, xlim=(0.01, 40), ylim=(-4, 6), labels=["log₂ x"]))
    ex.solution = _sol
    return ex


def l3_ln_sum(r):
    a, b = r.randint(2, 9), r.randint(2, 9)
    _sol = lines("ln a + ln b = ln(a × b)", f"ln {a} + ln {b} = ln({a} × {b}) = ln {a * b}")
    ex = Exercise(f"ln {a} + ln {b} = ln(?). Donne le nombre manquant.", Number(a * b), M_LN, curves(np.log, xlim=(0.01, 30), ylim=(-4, 4), labels=["ln x"]))
    ex.solution = _sol
    return ex


def l3_ln_diff(r):
    a, b = r.randint(2, 20), r.randint(2, 9)
    _sol = lines("ln a − ln b = ln(a / b)", f"ln {a} − ln {b} = ln({a}/{b}) = ln({fmt(F(a, b))})")
    ex = Exercise(f"ln {a} − ln {b} = ln(?). Donne le nombre manquant.", Number(F(a, b)), M_LN, curves(np.log, xlim=(0.01, 30), ylim=(-4, 4), labels=["ln x"]))
    ex.solution = _sol
    return ex


def l3_ln_pow(r):
    a, k = r.randint(2, 5), r.randint(2, 4)
    _sol = lines("k·ln a = ln(aᵏ)", f"{k}·ln {a} = ln({a}{sup(k)}) = ln {a ** k}")
    ex = Exercise(f"{k}·ln {a} = ln(?). Donne le nombre manquant.", Number(a ** k), M_LN, curves(np.log, xlim=(0.01, 30), ylim=(-4, 4), labels=["ln x"]))
    ex.solution = _sol
    return ex


def l3_ln_eq(r):
    k = r.choice([-1, 0.5, 1, 2, 3])
    _sol = lines("On applique exp des deux côtés : e^(ln x) = x", f"x = e^({fmt(k)}) ≈ {math.exp(k):.4f}")
    ex = Exercise(f"Résous ln x = {num(k)}." + CALC, Number(math.exp(k), 5e-3),
                    "Applique exp des deux côtés : x = e^(...).",
                    curves(np.log, lambda t: k + 0 * t, xlim=(0.01, 25), ylim=(-4, 4)))
    ex.solution = _sol
    return ex


def l3_exp_eq(r):
    a = r.choice([2, 5, 10, 0.5, 20])
    _sol = lines("On applique ln des deux côtés : ln(eˣ) = x", f"x = ln({fmt(a)}) ≈ {math.log(a):.4f}")
    ex = Exercise(f"Résous eˣ = {num(a)}." + CALC, Number(math.log(a), 5e-3),
                    "Applique ln des deux côtés : x = ln(...).",
                    curves(np.exp, lambda t: a + 0 * t, xlim=(-3, 4), ylim=(-1, 25)))
    ex.solution = _sol
    return ex


def l3_pow2_eq(r):
    a = r.choice([3, 5, 10, 12, 100])
    _sol = lines("On applique ln : ln(2ˣ) = x·ln 2 = ln " + str(a), f"x = ln {a} / ln 2 = {math.log(a):.4f} / {math.log(2):.4f} ≈ {math.log2(a):.4f}")
    ex = Exercise(f"Résous 2ˣ = {a}." + CALC, Number(math.log2(a), 5e-3),
                    "Applique ln des deux côtés : ln(2ˣ) = x·ln 2. Isole x.",
                    curves(lambda t: 2.0 ** t, lambda t: a + 0 * t, xlim=(-2, 8), ylim=(-1, a * 1.3), labels=["2ˣ", str(a)]))
    ex.solution = _sol
    return ex


def l3_logb(r):
    b, n = r.choice([(4, 3), (9, 2), (8, 2), (25, 3), (3, 4), (16, 2)])
    _sol = lines(f"{b} à quelle puissance donne {b ** n} ?", f"{b}{sup(n)} = {b ** n}, donc log_{b}({b ** n}) = {n}")
    ex = Exercise(f"Calcule log_{b}({b ** n}).", Number(n),
                    M_LN + f"\nQuestion à te poser : « {b} à quelle puissance donne {b ** n} ? »",
                    curves(lambda t: float(b) ** t, lambda t: b ** n + 0 * t, xlim=(-1, 5), ylim=(-1, b ** n * 1.3)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 4

def l4_aebx(r):
    a, b, c = r.randint(2, 5), r.choice([-2, -1, 1, 2, 3]), r.randint(6, 40)
    _sol = lines(f"On divise par {a} : e^({b}x) = {c}/{a} = {c / a:.4f}", f"On applique ln : {b}x = ln({c}/{a}) ≈ {math.log(c / a):.4f}",
                 f"x = {math.log(c / a):.4f} / {par(b)} ≈ {math.log(c / a) / b:.4f}")
    ex = Exercise(f"Résous {a}·e^({b}x) = {c}.".replace("(1x)", "(x)").replace("(-1x)", "(−x)") + CALC,
                    Number(math.log(c / a) / b, 5e-3),
                    "Isole l'exponentielle (divise par " + str(a) + "), applique ln, puis isole x.",
                    curves(lambda t: a * np.exp(b * t), lambda t: c + 0 * t, xlim=(-3, 3), ylim=(-1, c * 1.5)))
    ex.solution = _sol
    return ex


def l4_ln_lin(r):
    a, b = nz(r, -5, 5), r.randint(-9, 9)
    _sol = lines("ln(A) = 0 ⇔ A = 1 (car ln 1 = 0)", [f"{lin(a, b)} = 1"] + steps_lin(a, b, 1))
    ex = Exercise(f"Résous ln({a}x {'+' if b >= 0 else '−'} {abs(b)}) = 0.".replace("(1x", "(x"), Number(F(1 - b, a)),
                    "ln(A) = 0 ⇔ A = 1.",
                    curves(lambda t: np.log(a * t + b), xlim=(-10, 10), ylim=(-4, 3)))
    ex.solution = _sol
    return ex


def l4_ln_eq_sum(r):
    a, b = r.randint(2, 6), r.randint(2, 40)
    _sol = lines(f"ln x + ln {a} = ln({a}x)", f"ln({a}x) = ln {b} ⇒ {a}x = {b}", f"x = {b}/{a} = {fmt(F(b, a))}")
    ex = Exercise(f"Résous ln x + ln {a} = ln {b}.", Number(F(b, a)),
                    "Regroupe à gauche avec ln a + ln b = ln(ab), puis ln A = ln B ⇔ A = B.", curves(np.log, xlim=(0.01, 30), ylim=(-4, 4), labels=["ln x"]))
    ex.solution = _sol
    return ex


def l4_doubling_time(r):
    rate = r.choice([2, 3, 5, 7, 10])
    _g = 1 + rate / 100
    _sol = lines(f"On cherche n tel que {_g:g}ⁿ = 2", f"ln : n·ln({_g:g}) = ln 2", f"n = ln 2 / ln({_g:g}) = {math.log(2):.4f} / {math.log(_g):.5f} ≈ {math.log(2) / math.log(_g):.3f} ans")
    ex = Exercise(f"Un placement rapporte {rate} % par an. Au bout de combien d'années (non entier) le capital "
                    "a-t-il doublé ?" + CALC, Number(math.log(2) / math.log(1 + rate / 100), 5e-3),
                    "Résous (1 + r)ⁿ = 2. Applique ln : n·ln(1 + r) = ln 2.",
                    curves(lambda t: (1 + rate / 100) ** t, lambda t: 2 + 0 * t, xlim=(0, 80), ylim=(0, 4),
                           labels=["capital / capital initial", "2"]))
    ex.solution = _sol
    return ex


def l4_decay_const(r):
    lam = r.choice([0.01, 0.05, 0.1, 0.2, 0.5])
    _sol = lines(f"On cherche t tel que e^(−{lam}t) = 1/2", f"ln : −{lam}t = ln(1/2) = −ln 2", f"t = ln 2 / {lam} = {math.log(2):.4f} / {lam} ≈ {math.log(2) / lam:.3f}")
    ex = Exercise(f"N(t) = N₀·e^(−{lam}t). Au bout de combien de temps N est-il divisé par 2 ?" + CALC,
                    Number(math.log(2) / lam, 5e-3),
                    "Résous e^(−λt) = 1/2. Applique ln : −λt = ln(1/2) = −ln 2.",
                    curves(lambda t: np.exp(-lam * t), xlim=(0, 3 * math.log(2) / lam)))
    ex.solution = _sol
    return ex


def l4_quad_exp(r):
    a, b = r.sample([2, 3, 4, 5, 6], 2)
    _sol = lines(f"On pose X = eˣ : X² − {a + b}X + {a * b} = 0", f"Δ = {a + b}² − 4 × {a * b} = {(a - b) ** 2} = {abs(a - b)}²",
                 f"X = ({a + b} ± {abs(a - b)}) / 2 → X = {min(a, b)} ou X = {max(a, b)} (tous deux > 0)",
                 f"x = ln {min(a, b)} ≈ {math.log(min(a, b)):.4f} ou x = ln {max(a, b)} ≈ {math.log(max(a, b)):.4f}")
    ex = Exercise(f"Résous e^(2x) − {a + b}eˣ + {a * b} = 0." + CALC, Numbers([math.log(a), math.log(b)], tol=5e-3),
                    "Pose X = eˣ (et e^(2x) = X²). Résous l'équation du second degré en X, puis reviens à x "
                    "avec x = ln X (seules les valeurs X > 0 sont possibles).",
                    curves(lambda t: np.exp(2 * t) - (a + b) * np.exp(t) + a * b, xlim=(-1, 2.5), ylim=(-8, 15)))
    ex.solution = _sol
    return ex


def l4_domain(r):
    a = r.randint(-9, 9)
    _sol = lines(f"ln n'existe que pour un nombre > 0 : x {'+' if -a >= 0 else '−'} {abs(a)} > 0", f"donc x > {a}")
    ex = Exercise(f"ln(x {'+' if -a >= 0 else '−'} {abs(a)}) n'existe que pour x > ? Donne la borne.", Number(a),
                    "ln n'est défini que pour un nombre strictement positif.",
                    curves(lambda t: np.log(t - a), xlim=(a - 2, a + 10), ylim=(-4, 3)))
    ex.solution = _sol
    return ex


def l4_ln_square(r):
    k = r.choice([2, 4, 6])
    _sol = lines(f"On applique exp : x² = e^{k}", f"x = ±√(e^{k}) = ±e^{k // 2} ≈ ±{math.exp(k / 2):.4f}")
    ex = Exercise(f"Résous ln(x²) = {k}." + CALC, Numbers([math.exp(k / 2), -math.exp(k / 2)], tol=5e-3),
                    "Applique exp : x² = e^(...). N'oublie pas qu'un carré a deux racines.",
                    curves(lambda t: np.log(t ** 2), lambda t: k + 0 * t, xlim=(-25, 25), ylim=(-4, 8)))
    ex.solution = _sol
    return ex


def l4_decibel(r):
    k = r.randint(1, 12)
    _sol = lines(f"I / I₀ = 10{sup(k)}", f"10·log₁₀(10{sup(k)}) = 10 × {k} = {10 * k} dB")
    ex = Exercise(f"Un son a une intensité I = 10{sup(k)} × I₀. Son niveau en décibels vaut 10·log₁₀(I / I₀). Combien ?",
                    Number(10 * k), "log₁₀(10ᵏ) = k.",
                    curves(lambda t: 10 * np.log10(t), xlim=(1, 1000), labels=["10·log₁₀(x)"]))
    ex.solution = _sol
    return ex


def l4_change_base(r):
    a = r.choice([5, 7, 10, 20, 50])
    _sol = lines("log₂(a) = ln a / ln 2", f"= {math.log(a):.4f} / {math.log(2):.4f} ≈ {math.log2(a):.4f}")
    ex = Exercise(f"Calcule log₂({a})." + CALC, Number(math.log2(a), 5e-3),
                    "Changement de base : log_b(x) = ln x / ln b.",
                    curves(np.log2, np.log, xlim=(0.05, 60), ylim=(-3, 7), labels=["log₂", "ln"]))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 5

def _bars(scores):
    """Diagramme en barres des scores bruts (logits) z."""
    def draw(ax):
        ax.bar([f"z{i + 1}" for i in range(len(scores))], scores, color=BLUE)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("scores bruts (avant softmax)")
    return draw


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def l5_sigmoid(r):
    z = r.choice([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
    _sol = lines(f"e^(−z) = e^({fmt(-z)}) ≈ {math.exp(-z):.4f}", f"1 + e^(−z) ≈ {1 + math.exp(-z):.4f}",
                 f"σ({fmt(z)}) = 1 / {1 + math.exp(-z):.4f} ≈ {sigmoid(z):.4f}")
    ex = Exercise(f"La sigmoïde σ(z) = 1 / (1 + e^(−z)) transforme un score en probabilité. Calcule σ({num(z)})." + CALC,
                    Number(sigmoid(z), 5e-3),
                    "Calcule e^(−z) d'abord, ajoute 1, puis prends l'inverse. Vérifie : σ(0) = 0.5.",
                    curves(lambda t: 1 / (1 + np.exp(-t)), xlim=(-6, 6), ylim=(-0.1, 1.1), points=[(z, sigmoid(z))]))
    ex.solution = _sol
    return ex


def l5_logit(r):
    p = r.choice([0.1, 0.2, 0.25, 0.75, 0.8, 0.9])
    _sol = lines(f"{p} = 1 / (1 + e^(−z)) ⇒ 1 + e^(−z) = 1/{p} = {1 / p:.4f}", f"e^(−z) = {1 / p - 1:.4f}",
                 f"−z = ln({1 / p - 1:.4f}) ⇒ z = −ln({1 / p - 1:.4f}) = ln({p}/{1 - p:g}) ≈ {math.log(p / (1 - p)):.4f}")
    ex = Exercise(f"Pour quel z a-t-on σ(z) = {p} ? (σ(z) = 1/(1 + e^(−z)))" + CALC, Number(math.log(p / (1 - p)), 5e-3),
                    "Écris p = 1/(1 + e^(−z)), passe à l'inverse, isole e^(−z), puis applique ln.\n"
                    "Résultat général : z = ln(p / (1 − p)), appelé le « logit ».",
                    curves(lambda t: 1 / (1 + np.exp(-t)), lambda t: p + 0 * t, xlim=(-5, 5), ylim=(-0.1, 1.1)))
    ex.solution = _sol
    return ex


def l5_softmax2(r):
    a, b = r.randint(-2, 3), r.randint(-2, 3)
    _ea, _eb = math.exp(a), math.exp(b)
    _sol = lines(f"e^{a} ≈ {_ea:.4f}, e^{b} ≈ {_eb:.4f}", f"p₁ = {_ea:.4f} / ({_ea:.4f} + {_eb:.4f}) ≈ {_ea / (_ea + _eb):.4f}",
                 f"(Équivalent : 1 / (1 + e^({b} − {par(a)})) = σ({a - b}).)")
    ex = Exercise(f"Softmax de deux scores z₁ = {a}, z₂ = {b} : p₁ = e^(z₁) / (e^(z₁) + e^(z₂)). Calcule p₁." + CALC,
                    Number(math.exp(a) / (math.exp(a) + math.exp(b)), 5e-3),
                    "Calcule les deux exponentielles, puis le quotient. Astuce : divise haut et bas par e^(z₁) "
                    "→ 1/(1 + e^(z₂−z₁)) : c'est une sigmoïde !",
                    _bars([a, b]))
    ex.solution = _sol
    return ex


def l5_logloss(r):
    p = r.choice([0.9, 0.5, 0.2, 0.99, 0.01, 0.7])
    _sol = lines(f"−ln({p}) ≈ −({math.log(p):.4f}) = {-math.log(p):.4f}",
                 "Plus p est proche de 1, plus la perte est faible." if p >= 0.5 else "p est petit : la perte est élevée (mauvaise prédiction).")
    ex = Exercise(f"Le modèle donne la probabilité p = {p} à la bonne classe. La perte vaut −ln(p). Calcule-la." + CALC,
                    Number(-math.log(p), 5e-3),
                    "−ln(p) : proche de 0 si p ≈ 1 (bonne prédiction), énorme si p ≈ 0. C'est pour ça qu'on "
                    "minimise cette quantité.", curves(lambda t: -np.log(t), xlim=(0.005, 1), ylim=(0, 6),
                                                       points=[(p, -math.log(p))]))
    ex.solution = _sol
    return ex


def l5_bce(r):
    y, p = r.choice([0, 1]), r.choice([0.1, 0.3, 0.6, 0.8, 0.95])
    _sol = lines(f"y = {y} : " + ("seul le terme y·ln(p) reste : L = −ln(p)" if y == 1 else "seul le terme (1 − y)·ln(1 − p) reste : L = −ln(1 − p)"),
                 f"L = −ln({p if y == 1 else round(1 - p, 4)}) ≈ {-(y * math.log(p) + (1 - y) * math.log(1 - p)):.4f}")
    ex = Exercise(f"Entropie croisée binaire : L = −[y·ln(p) + (1 − y)·ln(1 − p)]. Calcule L pour y = {y}, p = {p}." + CALC,
                    Number(-(y * math.log(p) + (1 - y) * math.log(1 - p)), 5e-3),
                    "Remplace y : l'un des deux termes disparaît (multiplié par 0). Calcule le terme restant.",
                    curves(lambda t: -np.log(t), lambda t: -np.log(1 - t), xlim=(0.005, 0.995), ylim=(0, 5),
                           labels=["perte si y = 1", "perte si y = 0"]))
    ex.solution = _sol
    return ex


def l5_lse(r):
    a, b = r.randint(0, 4), r.randint(0, 4)
    _sol = lines(f"e{sup(a)} + e{sup(b)} ≈ {math.exp(a):.4f} + {math.exp(b):.4f} = {math.exp(a) + math.exp(b):.4f}",
                 f"ln({math.exp(a) + math.exp(b):.4f}) ≈ {math.log(math.exp(a) + math.exp(b)):.4f} (proche de max({a}, {b}) = {max(a, b)})")
    ex = Exercise(f"Calcule ln(e{sup(a)} + e{sup(b)})." + CALC, Number(math.log(math.exp(a) + math.exp(b)), 5e-3),
                    "Attention : ln(A + B) ≠ ln A + ln B ! Calcule la somme d'abord. Remarque : le résultat est "
                    "proche du maximum des deux exposants (« log-sum-exp » = un max adouci).",
                    curves(lambda t: np.log(np.exp(t) + np.exp(-t)), lambda t: np.abs(t), xlim=(-4, 4),
                           labels=["ln(eˣ + e⁻ˣ)", "max(x, −x)"]))
    ex.solution = _sol
    return ex


def l5_log_product(r):
    ps = r.sample([0.5, 0.25, 0.8, 0.1, 0.9, 0.4], 3)
    _terms = " + ".join(f"ln {p}" for p in ps)
    _vals = " + ".join(f"({math.log(p):.4f})" for p in ps)
    _sol = lines(f"ln({ps[0]} × {ps[1]} × {ps[2]}) = {_terms}", f"= {_vals}", f"≈ {sum(math.log(p) for p in ps):.4f}")
    ex = Exercise(f"Log-vraisemblance : calcule ln({ps[0]} × {ps[1]} × {ps[2]})." + CALC,
                    Number(sum(math.log(p) for p in ps), 5e-3),
                    "ln(abc) = ln a + ln b + ln c. En IA on additionne des log plutôt que de multiplier des "
                    "probabilités (le produit deviendrait trop petit pour un ordinateur).",
                    curves(np.log, xlim=(0.01, 1), ylim=(-5, 0.5), points=[(p, math.log(p)) for p in ps]))
    ex.solution = _sol
    return ex


def l5_softmax3(r):
    z = [r.randint(-1, 3) for _ in range(3)]
    i = z.index(max(z))
    s = sum(math.exp(v) for v in z)
    _e = [math.exp(v) for v in z]
    _sol = lines("Exponentielles : " + ", ".join(f"e^{v} ≈ {ev:.4f}" for v, ev in zip(z, _e)), f"Somme ≈ {s:.4f}",
                 f"p_{i + 1} = {_e[i]:.4f} / {s:.4f} ≈ {_e[i] / s:.4f}")
    ex = Exercise(f"Scores z = ({z[0]}, {z[1]}, {z[2]}). Calcule la probabilité softmax de la classe {i + 1}, "
                    f"e^(z_{i + 1}) / Σ e^(z_j)." + CALC, Number(math.exp(z[i]) / s, 5e-3),
                    "Calcule les trois exponentielles, leur somme, puis le quotient. Les probabilités "
                    "obtenues sont positives et somment à 1.", _bars(z))
    ex.solution = _sol
    return ex


def l5_tanh(r):
    z = r.choice([-2, -1, -0.5, 0.5, 1, 2])
    _a, _b = math.exp(z), math.exp(-z)
    _sol = lines(f"e^{fmt(z)} ≈ {_a:.4f}, e^{fmt(-z)} ≈ {_b:.4f}", f"tanh = ({_a:.4f} − {_b:.4f}) / ({_a:.4f} + {_b:.4f}) ≈ {math.tanh(z):.4f}")
    ex = Exercise(f"tanh(z) = (e^z − e^(−z)) / (e^z + e^(−z)). Calcule tanh({num(z)})." + CALC,
                    Number(math.tanh(z), 5e-3), "Calcule e^z et e^(−z), puis le quotient. tanh varie entre −1 et 1.",
                    curves(np.tanh, lambda t: 1 / (1 + np.exp(-t)), xlim=(-5, 5), labels=["tanh", "σ"]))
    ex.solution = _sol
    return ex


def l5_sig_sym(r):
    z = r.choice([0.7, 1.3, 2.5, 4])
    _sol = lines(f"σ({z}) ≈ {sigmoid(z):.4f} et σ(−{z}) ≈ {sigmoid(-z):.4f}", f"Somme ≈ {sigmoid(z) + sigmoid(-z):.4f} = 1",
                 "En général : 1/(1 + e^(−z)) + 1/(1 + e^z) = 1/(1 + e^(−z)) + e^(−z)/(e^(−z) + 1) = 1.")
    ex = Exercise(f"Calcule σ({z}) + σ(−{z}) (σ = sigmoïde). Quel nombre obtiens-tu ?", Number(1),
                    "Calcule les deux, ou démontre-le : réduis au même dénominateur 1/(1 + e^(−z)) + 1/(1 + e^z).",
                    curves(lambda t: 1 / (1 + np.exp(-t)), xlim=(-6, 6), points=[(z, sigmoid(z)), (-z, sigmoid(-z))]))
    ex.solution = _sol
    return ex


LEVELS = {
    1: [l1_power, l1_mul, l1_div, l1_powpow, l1_neg_exp, l1_scientific, l1_sqrt_pow, l1_frac_pow, l1_solve_pow,
        l1_product_pow],
    2: [l2_exp_mul, l2_exp_div, l2_exp_value, l2_exp_eq, l2_doubling, l2_interest, l2_exp_ln, l2_halflife,
        l2_exp_one, l2_exp_eq2],
    3: [l3_log10, l3_ln_e, l3_log2, l3_ln_sum, l3_ln_diff, l3_ln_pow, l3_ln_eq, l3_exp_eq, l3_pow2_eq, l3_logb],
    4: [l4_aebx, l4_ln_lin, l4_ln_eq_sum, l4_doubling_time, l4_decay_const, l4_quad_exp, l4_domain, l4_ln_square,
        l4_decibel, l4_change_base],
    5: [l5_sigmoid, l5_logit, l5_softmax2, l5_logloss, l5_bce, l5_lse, l5_log_product, l5_softmax3, l5_tanh,
        l5_sig_sym],
}
