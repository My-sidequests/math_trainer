"""Règle de la chaîne, dérivées partielles, gradient, graphes de calcul (backprop), Hessienne."""

from __future__ import annotations

import math

import numpy as np

from ..answers import Choice, Expr, Number, Numbers
from ..core import Exercise, coef, contour, curves, lin, nz, poly, signed, sterm, vec

DESCRIPTIONS = {
    1: "Règle de la chaîne (une variable)",
    2: "Dérivées partielles",
    3: "Le gradient : calcul, norme, direction de plus forte pente",
    4: "Chaîne à plusieurs étapes : graphes de calcul (avant-goût de la backprop)",
    5: "Dérivées secondes partielles, Hessienne, points critiques, Jacobienne",
}

M_CHAIN = ("Règle de la chaîne : si y = f(u) et u = g(x), alors dy/dx = dy/du · du/dx.\n"
           "En pratique : dérive la fonction extérieure en gardant l'intérieur tel quel, puis multiplie par la "
           "dérivée de l'intérieur.\n"
           "  (uⁿ)' = n·u'·uⁿ⁻¹   (eᵘ)' = u'·eᵘ   (ln u)' = u'/u   (sin u)' = u'·cos u   (√u)' = u'/(2√u)")
M_PARTIAL = ("Dérivée partielle ∂f/∂x : on dérive par rapport à x en traitant y comme une CONSTANTE (un simple nombre).\n"
             "∂f/∂y : l'inverse. Exemple : f = 3x²y → ∂f/∂x = 6xy, ∂f/∂y = 3x².")
M_GRAD = ("Le gradient rassemble les dérivées partielles : ∇f(x, y) = (∂f/∂x, ∂f/∂y).\n"
          "Il pointe dans la direction de plus forte MONTÉE ; −∇f est la direction de plus forte descente.\n"
          "Sa norme ‖∇f‖ = √((∂f/∂x)² + (∂f/∂y)²) mesure la raideur de la pente.")
M_GRAPH = ("Découpe le calcul en étapes simples (un graphe de calcul). Calcule d'abord toutes les valeurs "
           "intermédiaires (passe avant), puis multiplie les dérivées locales en remontant de la sortie vers "
           "l'entrée (passe arrière) : dL/dx = dL/dv · dv/du · du/dx.")
GRID = "\nRéponse : les composantes séparées par des virgules, dans l'ordre (ex : 2, -4)."


# ================================================================ NIVEAU 1

def l1_square_lin(r):
    a, b = nz(r, -6, 6), nz(r, -9, 9)
    return Exercise(f"Dérive f(x) = ({lin(a, b)})².", Expr(f"2*{a}*({a}*x + {b})"), M_CHAIN,
                    curves(lambda t: (a * t + b) ** 2, xlim=(-b / a - 3, -b / a + 3)))


def l1_power_quad(r):
    a, n = r.randint(1, 6), r.randint(2, 4)
    return Exercise(f"Dérive f(x) = (x² + {a})^{n}.", Expr(f"{n}*2*x*(x**2 + {a})**{n - 1}"), M_CHAIN, None)


def l1_exp_poly(r):
    a = nz(r, -5, 5)
    return Exercise(f"Dérive f(x) = e^(x² {sterm(a, 'x')}).", Expr(f"(2*x + {a})*exp(x**2 + {a}*x)"), M_CHAIN, None)


def l1_value(r):
    a, b, k = nz(r, -4, 4), r.randint(-5, 5), r.randint(-3, 3)
    return Exercise(f"f(u) = u² et u = g(x) = {lin(a, b)}. Calcule (f∘g)'({k}) = d f(g(x))/dx en x = {k}.",
                    Number(2 * (a * k + b) * a), M_CHAIN + "\nCalcule u = g(" + str(k) + ") d'abord.", None)


def l1_table(r):
    gk, dg, df = r.randint(-5, 5), nz(r, -6, 6), nz(r, -6, 6)
    return Exercise(f"On sait que g(1) = {gk}, g'(1) = {dg} et f'({gk}) = {df}. Calcule (f∘g)'(1).",
                    Number(df * dg),
                    "(f∘g)'(x) = f'(g(x))·g'(x). Remplace x = 1 : tout est donné, il suffit de choisir les bonnes valeurs.",
                    None)


def l1_ln_quad(r):
    a = r.randint(1, 9)
    return Exercise(f"Dérive f(x) = ln(x² + {a}).", Expr(f"2*x/(x**2 + {a})"), M_CHAIN,
                    curves(lambda t: np.log(t ** 2 + a), xlim=(-5, 5)))


def l1_sin_sq(r):
    return Exercise("Dérive f(x) = sin(x²).", Expr("2*x*cos(x**2)"), M_CHAIN,
                    curves(lambda t: np.sin(t ** 2), xlim=(-3, 3)))


def l1_sqrt_value(r):
    a, b, c = r.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17)])
    k = a
    return Exercise(f"f(x) = √(x² + {b * b}). Calcule f'({k}).", Number(k / c),
                    M_CHAIN + f"\nIndice : √({k}² + {b * b}) est un entier.", None)


def l1_sin_power(r):
    fn = r.choice(["sin", "cos"])
    target = "2*sin(x)*cos(x)" if fn == "sin" else "-2*cos(x)*sin(x)"
    return Exercise(f"Dérive f(x) = ({fn} x)².", Expr(target), M_CHAIN + "\nIci u = " + fn + " x et f = u².", None)


def l1_triple(r):
    a, k = nz(r, -2, 2), r.choice([0, 1, -1])
    u = a * k + 1
    return Exercise(f"f(x) = e^(({lin(a, 1)})²). Calcule f'({k}) (exact avec e, ou décimal précis).",
                    Number(math.exp(u * u) * 2 * u * a),
                    M_CHAIN + "\nTrois étages : x → u = ax + 1 → v = u² → f = eᵛ. Multiplie les trois dérivées.", None)


# ================================================================ NIVEAU 2

def _poly2(r):
    a, b, c = nz(r, -5, 5), nz(r, -5, 5), nz(r, -6, 6)
    return a, b, c, f"{coef(a)}x²y {sterm(b, 'y²')} {sterm(c, 'x')}"


def l2_dx(r):
    a, b, c, s = _poly2(r)
    return Exercise(f"f(x, y) = {s}. Calcule ∂f/∂x.", Expr(f"2*{a}*x*y + {c}"), M_PARTIAL,
                    contour(lambda x, y: a * x * x * y + b * y * y + c * x, (-3, 3), (-3, 3)))


def l2_dy(r):
    a, b, c, s = _poly2(r)
    return Exercise(f"f(x, y) = {s}. Calcule ∂f/∂y.", Expr(f"{a}*x**2 + 2*{b}*y + 0*x"), M_PARTIAL,
                    contour(lambda x, y: a * x * x * y + b * y * y + c * x, (-3, 3), (-3, 3)))


def l2_dx_value(r):
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    x, y = r.randint(-3, 3), r.randint(-3, 3)
    return Exercise(f"f(x, y) = {coef(a)}x³ {sterm(b, 'xy²')}. Calcule ∂f/∂x au point ({x}, {y}).",
                    Number(3 * a * x * x + b * y * y), M_PARTIAL + "\nDérive d'abord, puis remplace x et y.",
                    contour(lambda X, Y: a * X ** 3 + b * X * Y ** 2, (-4, 4), (-4, 4), points=[(x, y)]))


def l2_dy_value(r):
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    x, y = r.randint(-3, 3), r.randint(-3, 3)
    return Exercise(f"f(x, y) = {coef(a)}x²y {sterm(b, 'y³')}. Calcule ∂f/∂y au point ({x}, {y}).",
                    Number(a * x * x + 3 * b * y * y), M_PARTIAL,
                    contour(lambda X, Y: a * X ** 2 * Y + b * Y ** 3, (-4, 4), (-4, 4), points=[(x, y)]))


def l2_xexpy(r):
    return Exercise("f(x, y) = x·eʸ. Calcule ∂f/∂y.", Expr("x*exp(y)"), M_PARTIAL + "\nx est une constante multiplicative.",
                    None)


def l2_sincos(r):
    which = r.choice(["x", "y"])
    target = "cos(x)*cos(y)" if which == "x" else "-sin(x)*sin(y)"
    return Exercise(f"f(x, y) = sin(x)·cos(y). Calcule ∂f/∂{which}.", Expr(target), M_PARTIAL,
                    contour(lambda x, y: np.sin(x) * np.cos(y), (-4, 4), (-4, 4)))


def l2_grad_bowl(r):
    a, b = r.randint(-4, 4), r.randint(-4, 4)
    return Exercise(f"f(x, y) = x² + y². Calcule ∇f au point ({a}, {b})." + GRID, Numbers([2 * a, 2 * b], ordered=True),
                    M_GRAD, contour(lambda x, y: x * x + y * y, (-5, 5), (-5, 5), points=[(a, b)]))


def l2_grad_linear(r):
    a, b, c = nz(r, -7, 7), nz(r, -7, 7), r.randint(-9, 9)
    return Exercise(f"f(x, y) = {coef(a)}x {sterm(b, 'y')} {signed(c)}. Calcule ∇f (il est le même partout)." + GRID,
                    Numbers([a, b], ordered=True), M_GRAD,
                    contour(lambda x, y: a * x + b * y + c, (-4, 4), (-4, 4)))


def l2_grad_xy(r):
    x, y = nz(r, -4, 4), nz(r, -4, 4)
    return Exercise(f"f(x, y) = xy. Calcule ∇f au point ({x}, {y})." + GRID, Numbers([y, x], ordered=True), M_GRAD,
                    contour(lambda X, Y: X * Y, (-5, 5), (-5, 5), points=[(x, y)]))


def l2_ln_partial(r):
    x, y = r.choice([(3, 4), (1, 1), (1, 2), (2, 1), (0, 1)])
    return Exercise(f"f(x, y) = ln(x² + y²). Calcule ∂f/∂x au point ({x}, {y}).", Number(2 * x / (x * x + y * y)),
                    M_PARTIAL + "\nRègle de la chaîne : (ln u)' = u'/u avec u = x² + y².", None)


# ================================================================ NIVEAU 3

def l3_grad_ellipse(r):
    a, b = r.randint(1, 5), r.randint(1, 5)
    x, y = r.randint(-3, 3), r.randint(-3, 3)
    return Exercise(f"f(x, y) = {coef(a)}x² + {coef(b)}y². Calcule ∇f({x}, {y})." + GRID, Numbers([2 * a * x, 2 * b * y], ordered=True),
                    M_GRAD, contour(lambda X, Y: a * X * X + b * Y * Y, (-4, 4), (-4, 4), points=[(x, y)]))


def l3_norm(r):
    (p, q, n) = r.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (0, 2, 2)])
    x, y = p / 2, q / 2
    return Exercise(f"f(x, y) = x² + y². Calcule la norme du gradient ‖∇f‖ au point ({x:g}, {y:g}).", Number(n),
                    M_GRAD, contour(lambda X, Y: X * X + Y * Y, (-7, 7), (-7, 7), points=[(x, y)]))


def l3_descent_dir(r):
    a, b = r.randint(1, 4), r.randint(1, 4)
    x, y = nz(r, -3, 3), nz(r, -3, 3)
    return Exercise(f"f(x, y) = {coef(a)}x² + {coef(b)}y². Donne la direction de plus forte DESCENTE (le vecteur −∇f) au point ({x}, {y})." + GRID,
                    Numbers([-2 * a * x, -2 * b * y], ordered=True), M_GRAD,
                    contour(lambda X, Y: a * X * X + b * Y * Y, (-4, 4), (-4, 4), points=[(x, y)]))


def l3_zero_grad(r):
    a, b = r.randint(-5, 5), r.randint(-5, 5)
    return Exercise(f"f(x, y) = (x {signed(-a)})² + (y {signed(-b)})². En quel point le gradient est-il nul ?" + GRID,
                    Numbers([a, b], ordered=True), M_GRAD + "\nRésous ∂f/∂x = 0 et ∂f/∂y = 0.",
                    contour(lambda X, Y: (X - a) ** 2 + (Y - b) ** 2, (-7, 7), (-7, 7)))


def l3_x2y(r):
    x, y = r.randint(-3, 3), r.randint(-3, 3)
    return Exercise(f"f(x, y) = x²y. Calcule ∇f({x}, {y})." + GRID, Numbers([2 * x * y, x * x], ordered=True), M_GRAD,
                    contour(lambda X, Y: X * X * Y, (-4, 4), (-4, 4), points=[(x, y)]))


def l3_exp_grad(r):
    a, b = nz(r, -3, 3), nz(r, -3, 3)
    return Exercise(f"f(x, y) = e^({coef(a)}x {sterm(b, 'y')}). Calcule ∇f(0, 0)." + GRID, Numbers([a, b], ordered=True),
                    M_GRAD + "\nRègle de la chaîne pour chaque dérivée partielle ; e⁰ = 1.", None)


def l3_step(r):
    a, b = r.randint(1, 3), r.randint(1, 3)
    x, y = nz(r, -4, 4), nz(r, -4, 4)
    al = r.choice([0.1, 0.05, 0.25])
    nx, ny = x - al * 2 * a * x, y - al * 2 * b * y
    return Exercise(f"f(x, y) = {coef(a)}x² + {coef(b)}y². Fais un pas de descente de gradient depuis ({x}, {y}) avec α = {al} :\n"
                    "(x, y) ← (x, y) − α·∇f(x, y)." + GRID, Numbers([nx, ny], ordered=True),
                    M_GRAD + "\nCalcule ∇f au point, multiplie par α, soustrais composante par composante.",
                    contour(lambda X, Y: a * X * X + b * Y * Y, (-5, 5), (-5, 5), points=[(x, y)]))


def l3_directional(r):
    x, y = r.randint(-3, 3), r.randint(-3, 3)
    u = r.choice([(0.6, 0.8), (0.8, 0.6), (1, 0), (0, 1), (-0.6, 0.8)])
    g = (2 * x, 2 * y)
    return Exercise(f"f(x, y) = x² + y². Dérivée directionnelle au point ({x}, {y}) dans la direction du vecteur unitaire "
                    f"u = {vec(u)} ?", Number(g[0] * u[0] + g[1] * u[1]),
                    "La pente dans la direction u vaut ∇f · u (produit scalaire) : ∂f/∂x·u₁ + ∂f/∂y·u₂.",
                    contour(lambda X, Y: X * X + Y * Y, (-4, 4), (-4, 4), points=[(x, y)], arrows=[((x, y), u)]))


def l3_unit_ascent(r):
    (p, q, n) = r.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 6, 10)])
    sx, sy = r.choice([1, -1]), r.choice([1, -1])
    x, y = sx * p / 2, sy * q / 2
    return Exercise(f"f(x, y) = x² + y². Donne le vecteur UNITAIRE de plus forte montée au point ({x:g}, {y:g})." + GRID,
                    Numbers([sx * p / n, sy * q / n], ordered=True),
                    M_GRAD + "\nVecteur unitaire = ∇f / ‖∇f‖.",
                    contour(lambda X, Y: X * X + Y * Y, (-7, 7), (-7, 7), points=[(x, y)]))


def l3_grad3(r):
    x, y, z = (r.randint(-3, 3) for _ in range(3))
    return Exercise(f"f(x, y, z) = x² + 2y² + 3z². Calcule ∇f({x}, {y}, {z}).\nRéponse : 3 composantes séparées par des virgules.",
                    Numbers([2 * x, 4 * y, 6 * z], ordered=True), M_GRAD + "\nEn 3D il y a simplement 3 dérivées partielles.", None)


# ================================================================ NIVEAU 4

def l4_dz_dw(r):
    w, x, b = nz(r, -3, 3), nz(r, -3, 3), r.randint(-3, 3)
    return Exercise(f"z = (w·x + b)² avec w = {w}, x = {x}, b = {b}. Calcule ∂z/∂w.", Number(2 * (w * x + b) * x),
                    M_GRAPH + "\nIci : u = w·x + b, z = u². ∂z/∂w = 2u · ∂u/∂w = 2u · x.", None)


def l4_dz_db(r):
    w, x, b = nz(r, -3, 3), nz(r, -3, 3), r.randint(-3, 3)
    return Exercise(f"z = (w·x + b)² avec w = {w}, x = {x}, b = {b}. Calcule ∂z/∂b.", Number(2 * (w * x + b)),
                    M_GRAPH + "\nu = w·x + b, z = u². Que vaut ∂u/∂b ?", None)


def l4_loss_w(r):
    w, x, y = nz(r, -3, 3), nz(r, -3, 3), r.randint(-6, 6)
    return Exercise(f"Perte L = (w·x − y)² avec w = {w}, x = {x}, y = {y}. Calcule ∂L/∂w.", Number(2 * (w * x - y) * x),
                    M_GRAPH + "\nerreur e = w·x − y, L = e². ∂L/∂w = 2e·x.",
                    curves(lambda t: (t * x - y) ** 2, xlim=(w - 4, w + 4), points=[(w, (w * x - y) ** 2)]))


def l4_sigmoid_w(r):
    w, x = r.choice([0.5, 1, -1, 2]), r.choice([1, 2, -1])
    u = w * x
    s = 1 / (1 + math.exp(-u))
    return Exercise(f"a = σ(w·x) avec σ la sigmoïde, w = {w}, x = {x}. Calcule ∂a/∂w (décimal, 3 chiffres)."
                    "\nRappel : σ'(u) = σ(u)·(1 − σ(u)).", Number(s * (1 - s) * x, 5e-3),
                    M_GRAPH + "\nu = w·x, a = σ(u). ∂a/∂w = σ'(u) · x.", None)


def l4_graph(r):
    x = r.randint(-2, 2)
    u = x * x
    v = 3 * u + 1
    return Exercise(f"Graphe de calcul : u = x², v = 3u + 1, w = v². Calcule dw/dx en x = {x}.", Number(2 * v * 3 * 2 * x),
                    M_GRAPH + "\nValeurs : u, puis v. Dérivées locales : dw/dv = 2v, dv/du = 3, du/dx = 2x.", None)


def l4_mse_w(r):
    w, x, y = r.choice([0.5, 1, 2, -1]), nz(r, -3, 3), r.randint(-5, 5)
    yh = w * x
    return Exercise(f"Modèle ŷ = w·x, perte L = ½(ŷ − y)². Avec w = {w}, x = {x}, y = {y}, calcule ∂L/∂w.",
                    Number((yh - y) * x), M_GRAPH + "\n∂L/∂ŷ = (ŷ − y), ∂ŷ/∂w = x. Le ½ annule le 2 de la dérivée du carré.",
                    None)


def l4_param(r):
    a, t = nz(r, -3, 3), r.randint(-2, 3)
    return Exercise(f"z = x·y avec x = {coef(a)}t et y = t². Calcule dz/dt en t = {t}.", Number(3 * a * t * t),
                    "Chaîne à plusieurs variables : dz/dt = ∂z/∂x·dx/dt + ∂z/∂y·dy/dt. "
                    "(Ou remplace directement x et y en fonction de t, puis dérive.)", None)


def l4_param2(r):
    t = r.randint(-3, 3)
    x, y = t + 1, 2 * t
    return Exercise(f"z = x²·y avec x = t + 1 et y = 2t. Calcule dz/dt en t = {t}.", Number(2 * x * y * 1 + x * x * 2),
                    "dz/dt = ∂z/∂x·dx/dt + ∂z/∂y·dy/dt. Calcule x, y en t, puis chaque morceau.", None)


def l4_batch(r):
    w = r.choice([0, 1, 2, -1])
    pts = [(nz(r, -3, 3), r.randint(-5, 5)) for _ in range(2)]
    g = sum((w * x - y) * x for x, y in pts) / 2
    return Exercise(f"L(w) = ½·moyenne des (w·xᵢ − yᵢ)² sur 2 points : (x, y) = {pts[0]} et {pts[1]}. "
                    f"Calcule dL/dw en w = {w}.", Number(g),
                    "Avec eᵢ = w·xᵢ − yᵢ : L = ½·moyenne(eᵢ²). La dérivée d'une moyenne est la moyenne des dérivées, "
                    "et le ½ annule le 2 du carré : dL/dw = moyenne(eᵢ·xᵢ).", None)


def l4_relu(r):
    w, x, b = nz(r, -3, 3), nz(r, -3, 3), r.randint(-4, 4)
    z = w * x + b
    if z == 0:
        b += 1
        z += 1
    return Exercise(f"a = ReLU(w·x + b) = max(0, w·x + b) avec w = {w}, x = {x}, b = {b}. Calcule ∂a/∂w.",
                    Number(x if z > 0 else 0),
                    "ReLU'(u) = 1 si u > 0, 0 si u < 0. Calcule u d'abord : c'est un « interrupteur » pour le gradient.",
                    curves(lambda t: np.maximum(0, t), xlim=(-5, 5), points=[(z, max(0, z))]))


# ================================================================ NIVEAU 5

def l5_dxx(r):
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    return Exercise(f"f(x, y) = {coef(a)}x³y {sterm(b, 'x²y²')}. Calcule ∂²f/∂x².", Expr(f"{6 * a}*x*y + {2 * b}*y**2"),
                    "Dérive deux fois par rapport à x (y constant).", None)


def l5_dxy(r):
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    return Exercise(f"f(x, y) = {coef(a)}x³y {sterm(b, 'x²y²')}. Calcule ∂²f/∂x∂y (dérive par rapport à x, puis à y).",
                    Expr(f"{3 * a}*x**2 + {4 * b}*x*y"),
                    "Dérivée croisée : dérive par rapport à une variable, puis le résultat par rapport à l'autre. "
                    "(L'ordre ne change rien pour les fonctions usuelles : théorème de Schwarz.)", None)


def l5_hessian(r):
    a, b, c = nz(r, -4, 4), nz(r, -4, 4), nz(r, -4, 4)
    return Exercise(f"f(x, y) = {poly({2: a})} {sterm(b, 'xy')} {sterm(c, 'y²')}. Donne la Hessienne "
                    "H = [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]].\nRéponse : 4 nombres ligne par ligne.",
                    Numbers([2 * a, b, b, 2 * c], ordered=True),
                    "La Hessienne range toutes les dérivées secondes. Pour une forme quadratique elle est constante.", None)


def l5_critical(r):
    x0, y0 = r.randint(-3, 3), r.randint(-3, 3)
    a, b = r.randint(1, 3), r.randint(1, 3)
    # f = a(x - x0)^2 + b(y - y0)^2 développé
    return Exercise(f"f(x, y) = {poly({2: a, 1: -2 * a * x0})} + {poly({2: b, 1: -2 * b * y0}, 'y')}. "
                    "Trouve le point critique (∇f = 0)." + GRID, Numbers([x0, y0], ordered=True),
                    "Écris ∂f/∂x = 0 et ∂f/∂y = 0 : deux équations, deux inconnues.",
                    contour(lambda X, Y: a * (X - x0) ** 2 + b * (Y - y0) ** 2, (-6, 6), (-6, 6)))


def l5_nature(r):
    kind = r.choice(["minimum", "maximum", "col"])
    a, c = r.randint(1, 4), r.randint(1, 4)
    sa, sc = {"minimum": (1, 1), "maximum": (-1, -1), "col": (1, -1)}[kind]
    return Exercise(f"f(x, y) = {poly({2: sa * a})} {signed(sc * c)}y². Le point (0, 0) est critique. "
                    "Est-ce un minimum, un maximum ou un col ?", Choice(kind, ["minimum", "maximum", "col"]),
                    "Calcule la Hessienne et son déterminant D = f_xx·f_yy − f_xy².\n"
                    "D > 0 et f_xx > 0 → minimum ; D > 0 et f_xx < 0 → maximum ; D < 0 → col (point selle).",
                    contour(lambda X, Y: sa * a * X * X + sc * c * Y * Y, (-3, 3), (-3, 3)))


def l5_jacobian(r):
    x, y = r.randint(-3, 3), r.randint(-3, 3)
    return Exercise(f"F(x, y) = (x + y, x·y). Calcule la Jacobienne J = [[∂F₁/∂x, ∂F₁/∂y], [∂F₂/∂x, ∂F₂/∂y]] "
                    f"au point ({x}, {y}).\nRéponse : 4 nombres ligne par ligne.", Numbers([1, 1, y, x], ordered=True),
                    "Chaque ligne de la Jacobienne est le gradient d'une composante de F.", None)


def l5_laplacian(r):
    a, b = nz(r, -5, 5), nz(r, -5, 5)
    return Exercise(f"f(x, y) = {coef(a)}x² {sterm(b, 'y²')} + 7xy. Calcule la trace de la Hessienne (∂²f/∂x² + ∂²f/∂y²).",
                    Number(2 * a + 2 * b), "Calcule les deux dérivées secondes « pures » et additionne-les.", None)


def l5_taylor(r):
    h = r.choice([0.1, 0.2, 0.5, -0.1, -0.3])
    return Exercise(f"Approximation de Taylor d'ordre 2 : estime e^{h} avec f(0) + f'(0)·h + f''(0)·h²/2 (f = exp).",
                    Number(1 + h + h * h / 2, 1e-5),
                    "Toutes les dérivées de eˣ valent eˣ, donc 1 en 0. Remplace dans la formule.",
                    curves(np.exp, lambda t: 1 + t + t * t / 2, xlim=(-2, 2), labels=["eˣ", "Taylor ordre 2"]))


def l5_grad_norm2(r):
    w = [r.randint(-4, 4) for _ in range(3)]
    return Exercise(f"f(w) = ‖w‖² = w₁² + w₂² + w₃². Calcule ∇f en w = {vec(w)}.\nRéponse : 3 composantes.",
                    Numbers([2 * v for v in w], ordered=True),
                    "Chaque ∂f/∂wᵢ ne dépend que de wᵢ. (C'est le gradient de la régularisation L2.)", None)


def l5_grad_dot(r):
    a = [nz(r, -6, 6) for _ in range(3)]
    return Exercise(f"f(w) = a·w (produit scalaire) avec a = {vec(a)}. Calcule ∇f.\nRéponse : 3 composantes.",
                    Numbers(a, ordered=True),
                    "Écris f = a₁w₁ + a₂w₂ + a₃w₃ et dérive par rapport à chaque wᵢ.", None)


LEVELS = {
    1: [l1_square_lin, l1_power_quad, l1_exp_poly, l1_value, l1_table, l1_ln_quad, l1_sin_sq, l1_sqrt_value,
        l1_sin_power, l1_triple],
    2: [l2_dx, l2_dy, l2_dx_value, l2_dy_value, l2_xexpy, l2_sincos, l2_grad_bowl, l2_grad_linear, l2_grad_xy,
        l2_ln_partial],
    3: [l3_grad_ellipse, l3_norm, l3_descent_dir, l3_zero_grad, l3_x2y, l3_exp_grad, l3_step, l3_directional,
        l3_unit_ascent, l3_grad3],
    4: [l4_dz_dw, l4_dz_db, l4_loss_w, l4_sigmoid_w, l4_graph, l4_mse_w, l4_param, l4_param2, l4_batch, l4_relu],
    5: [l5_dxx, l5_dxy, l5_hessian, l5_critical, l5_nature, l5_jacobian, l5_laplacian, l5_taylor, l5_grad_norm2,
        l5_grad_dot],
}
