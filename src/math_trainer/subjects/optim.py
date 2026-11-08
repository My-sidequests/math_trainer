"""Optimisation : minimum d'une fonction, descente de gradient, Newton, contraintes, optimiseurs du ML."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Number, Numbers
from ..core import Exercise, bars, contour, curves, nz, poly, signed

from ..core import fmt, lin, lines, par, term  # noqa: E402,F811

DESCRIPTIONS = {
    1: "Minimum d'une fonction et un pas de descente de gradient (1D)",
    2: "Descente de gradient : plusieurs pas, 2D, learning rate, momentum",
    3: "Newton, convexité, contraintes, Lagrange",
    4: "Optimisation en ML : SGD, régularisation, decay, clipping, epochs",
    5: "Régression logistique, Adam, conditionnement, pas optimal",
}

M_GD = ("Descente de gradient : x ← x − α·f'(x) (ou θ ← θ − α·∇f(θ) en plusieurs dimensions).\n"
        "α est le learning rate. Le signe « − » fait aller dans le sens où f diminue.")
M_MIN = "Un minimum (ou maximum) d'une fonction dérivable se trouve là où f'(x) = 0. Vérifie la nature avec f'' (> 0 → minimum)."
G2 = "\nRéponse : les composantes séparées par des virgules."


def gd_path(f, xs, xlim):
    return curves(f, xlim=xlim, points=[(x, f(x)) for x in xs])


# ================================================================ NIVEAU 1

def l1_argmin(r):
    a, h, k = r.randint(1, 4), r.randint(-6, 6), r.randint(-9, 9)
    _sol = lines(f"f'(x) = {lin(2 * a, -2 * a * h)}", f"f'(x) = 0 ⇔ {term(2 * a, 'x')} = {2 * a * h} ⇔ x = {h}", f"f'' = {2 * a} > 0 : c'est bien un minimum.")
    ex = Exercise(f"En quel x f(x) = {poly({2: a, 1: -2 * a * h, 0: a * h * h + k})} est-elle minimale ?", Number(h), M_MIN,
                    curves(lambda t: a * (t - h) ** 2 + k, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l1_min_value(r):
    a, h, k = r.randint(1, 4), r.randint(-6, 6), r.randint(-9, 9)
    _sol = lines(f"f'(x) = {lin(2 * a, -2 * a * h)} = 0 ⇔ x = {h}", f"f({h}) = {a} × {par(h)}² {signed(-2 * a * h)} × {par(h)} {signed(a * h * h + k)} = {k}")
    ex = Exercise(f"Valeur minimale de f(x) = {poly({2: a, 1: -2 * a * h, 0: a * h * h + k})} ?", Number(k),
                    M_MIN + "\nTrouve x*, puis calcule f(x*).", curves(lambda t: a * (t - h) ** 2 + k, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l1_step_x2(r):
    x, al = nz(r, -6, 6), r.choice([0.1, 0.25, 0.05, 0.4])
    _sol = lines(f"f'(x) = 2x, donc f'({x}) = {2 * x}", f"x ← {x} − {al} × {par(2 * x)} = {x} {signed(-al * 2 * x)} = {fmt(x - al * 2 * x)}")
    ex = Exercise(f"f(x) = x². Un pas de descente depuis x = {x} avec α = {al} ?", Number(x - al * 2 * x), M_GD,
                    gd_path(lambda t: t ** 2, [x, x - al * 2 * x], (-abs(x) - 1, abs(x) + 1)))
    ex.solution = _sol
    return ex


def l1_step_shift(r):
    a, x, al = r.randint(-5, 5), r.randint(-6, 6), r.choice([0.1, 0.25, 0.5])
    _g = 2 * (x - a)
    _sol = lines(f"f'(x) = 2(x {signed(-a)}), donc f'({x}) = {_g}", f"x ← {x} − {al} × {par(_g)} = {fmt(x - al * _g)}")
    ex = Exercise(f"f(x) = (x {signed(-a)})². Un pas de descente depuis x = {x} avec α = {al} ?", Number(x - al * 2 * (x - a)),
                    M_GD, gd_path(lambda t: (t - a) ** 2, [x, x - al * 2 * (x - a)], (min(a, x) - 3, max(a, x) + 3)))
    ex.solution = _sol
    return ex


def l1_two_steps(r):
    x, al = nz(r, -8, 8), r.choice([0.1, 0.25])
    x1 = x - al * 2 * x
    x2 = x1 - al * 2 * x1
    _sol = lines(f"Pas 1 : f'({x}) = {2 * x} ; x₁ = {x} − {al} × {par(2 * x)} = {fmt(x1)}",
                 f"Pas 2 : f'({fmt(x1)}) = {fmt(2 * x1)} ; x₂ = {fmt(x1)} − {al} × {par(2 * x1)} = {fmt(x2)}")
    ex = Exercise(f"f(x) = x². Fais DEUX pas de descente depuis x = {x} avec α = {al}. Valeur de x après 2 pas ?", Number(x2),
                    M_GD + "\nRecalcule la dérivée au nouveau point à chaque pas.", gd_path(lambda t: t ** 2, [x, x1, x2], (-abs(x) - 1, abs(x) + 1)))
    ex.solution = _sol
    return ex


def l1_direction(r):
    a, b = nz(r, -4, 4), r.randint(-8, 8)
    x = r.randint(-5, 5)
    d = 2 * a * x + b
    if d == 0:
        x += 1
        d = 2 * a * x + b
    _sol = lines(f"f'(x) = {lin(2 * a, b)}, donc f'({x}) = {d}", f"Le pas vaut −α·f'(x) : signe {'négatif' if d > 0 else 'positif'}, x va vers la {'gauche' if d > 0 else 'droite'}.")
    ex = Exercise(f"f(x) = {poly({2: a, 1: b})}. Depuis x = {x}, la descente de gradient fait-elle aller x vers la "
                    "gauche ou la droite ? (gauche / droite)", Choice("gauche" if d > 0 else "droite", ["gauche", "droite"]),
                    M_GD + "\nCalcule le signe de f'(x) : on va dans le sens opposé.", None)
    ex.solution = _sol
    return ex


def l1_factor(r):
    al = r.choice([0.1, 0.2, 0.25, 0.4, 0.05])
    _sol = lines("x − α·2x = (1 − 2α)·x", f"c = 1 − 2 × {al} = {1 - 2 * al:g}")
    ex = Exercise(f"Pour f(x) = x², chaque pas fait x ← x − α·2x = c·x. Que vaut c pour α = {al} ?", Number(1 - 2 * al),
                    "Factorise x dans x − 2αx.", None)
    ex.solution = _sol
    return ex


def l1_diverge(r):
    k = r.choice([1, 2, 3, 5])
    _sol = lines(f"Il faut |1 − {2 * k}α| < 1 pour que x rétrécisse.", f"La limite : 1 − {2 * k}α = −1 ⇔ {2 * k}α = 2 ⇔ α = {fmt(F(1, k))}",
                 f"Au-delà de α = {fmt(F(1, k))}, chaque pas fait grossir |x| : ça diverge.")
    ex = Exercise(f"f(x) = {k}x². x ← x − α·f'(x) = (1 − {2 * k}α)·x. À partir de quel α la descente diverge-t-elle ? "
                    "(donne la valeur seuil)", Number(F(1, k)),
                    "Ça diverge si |1 − 2kα| > 1. Résous 1 − 2kα = −1.",
                    curves(lambda t: k * t * t, xlim=(-3, 3)))
    ex.solution = _sol
    return ex


def l1_argmax(r):
    h, k = r.randint(-5, 5), r.randint(-5, 9)
    _sol = lines(f"f'(x) = −2x {signed(2 * h)}", f"f'(x) = 0 ⇔ x = {h}", "f'' = −2 < 0 : c'est un maximum.")
    ex = Exercise(f"En quel x f(x) = {poly({2: -1, 1: 2 * h, 0: k - h * h})} est-elle maximale ?", Number(h), M_MIN,
                    curves(lambda t: -(t - h) ** 2 + k, xlim=(h - 5, h + 5)))
    ex.solution = _sol
    return ex


def l1_step_size(r):
    x, al = nz(r, -6, 6), r.choice([0.1, 0.01, 0.5])
    _sol = lines(f"f'(x) = 3x², donc f'({x}) = {3 * x * x}", f"|α·f'(x)| = {al} × {3 * x * x} = {fmt(al * 3 * x * x)}")
    ex = Exercise(f"f(x) = x³. Depuis x = {x}, avec α = {al}, de combien x bouge-t-il (|α·f'(x)|) ?",
                    Number(abs(al * 3 * x * x)), "Taille du pas = α × |pente|. Pente forte → grand pas.", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 2

def l2_step2d(r):
    a, b = r.randint(1, 3), r.randint(1, 3)
    x, y, al = nz(r, -4, 4), nz(r, -4, 4), r.choice([0.1, 0.05, 0.2])
    _sol = lines(f"∇f = ({2 * a}x, {2 * b}y) = ({2 * a * x}, {2 * b * y})",
                 f"(x, y) ← ({x} − {al} × {par(2 * a * x)}, {y} − {al} × {par(2 * b * y)}) = ({fmt(x - al * 2 * a * x)}, {fmt(y - al * 2 * b * y)})")
    ex = Exercise(f"f(x, y) = {a}x² + {b}y². Un pas de descente depuis ({x}, {y}) avec α = {al} ?" + G2,
                    Numbers([x - al * 2 * a * x, y - al * 2 * b * y], ordered=True), M_GD,
                    contour(lambda X, Y: a * X * X + b * Y * Y, (-5, 5), (-5, 5), points=[(x, y)]))
    ex.solution = _sol
    return ex


def l2_n_steps(r):
    x0, al, n = r.choice([1, 2, 4, 10]), r.choice([0.25, 0.4, 0.1]), r.randint(3, 6)
    _c = 1 - 2 * al
    _sol = lines(f"Chaque pas : x ← (1 − 2 × {al})·x = {_c:g}·x", f"Après {n} pas : {x0} × {_c:g}^{n} = {x0} × {_c ** n:.6g} ≈ {x0 * _c ** n:.4f}")
    ex = Exercise(f"f(x) = x², x₀ = {x0}, α = {al}. Valeur de x après {n} pas ?" + "\n(calculatrice autorisée)",
                    Number(x0 * (1 - 2 * al) ** n, 1e-3),
                    "Chaque pas multiplie x par (1 − 2α). Après n pas : x₀·(1 − 2α)ⁿ.", None)
    ex.solution = _sol
    return ex


def l2_best_alpha(r):
    k = r.choice([1, 2, 4, 5])
    _sol = lines(f"f'(x) = {2 * k}x ; un pas donne x − α·{2 * k}x = (1 − {2 * k}α)·x", f"On veut 1 − {2 * k}α = 0 ⇔ α = 1/{2 * k}")
    ex = Exercise(f"f(x) = {k}x². Quel learning rate α envoie n'importe quel x au minimum en UN seul pas ?", Number(F(1, 2 * k)),
                    "x − α·2kx = 0 pour tout x : résous 1 − 2kα = 0.", None)
    ex.solution = _sol
    return ex


def l2_momentum(r):
    beta, v, g = r.choice([0.9, 0.5]), r.randint(-3, 3), r.randint(-4, 4)
    _sol = lines("v ← β·v + g", f"= {beta} × {par(v)} + {par(g)} = {beta * v:g} + {par(g)} = {beta * v + g:g}")
    ex = Exercise(f"Momentum : v ← β·v + g. Avec β = {beta}, v = {v}, g = {g}, nouvelle vitesse v ?", Number(beta * v + g),
                    "La vitesse accumule les gradients passés (avec un facteur d'oubli β). Puis θ ← θ − α·v.", None)
    ex.solution = _sol
    return ex


def l2_momentum_step(r):
    beta, v, g, th, al = 0.9, r.randint(-2, 2), r.randint(-4, 4), r.randint(-5, 5), 0.1
    nv = beta * v + g
    _sol = lines(f"v ← 0.9 × {par(v)} + {par(g)} = {fmt(nv)}", f"θ ← {th} − {al} × {par(nv)} = {fmt(th - al * nv)}")
    ex = Exercise(f"Momentum : v ← 0.9·v + g, puis θ ← θ − α·v. Avec v = {v}, g = {g}, θ = {th}, α = {al}, nouveau θ ?",
                    Number(th - al * nv), "Mets d'abord v à jour, puis utilise le NOUVEAU v pour déplacer θ.", None)
    ex.solution = _sol
    return ex


def l2_linreg_step(r):
    w, x, y, al = r.randint(-2, 2), nz(r, -3, 3), r.randint(-6, 6), r.choice([0.1, 0.05])
    g = 2 * (w * x - y) * x
    _sol = lines(f"Erreur : w·x − y = {w} × {par(x)} − {par(y)} = {w * x - y}", f"dL/dw = 2 × {par(w * x - y)} × {par(x)} = {g}",
                 f"w ← {w} − {al} × {par(g)} = {fmt(w - al * g)}")
    ex = Exercise(f"Modèle ŷ = w·x, perte L = (ŷ − y)². Point (x, y) = ({x}, {y}), w = {w}, α = {al}. Nouveau w après un pas ?",
                    Number(w - al * g), "dL/dw = 2(w·x − y)·x. Puis w ← w − α·dL/dw.", None)
    ex.solution = _sol
    return ex


def l2_stop(r):
    x0, al, eps = r.choice([8, 16, 4]), 0.25, r.choice([1, 0.5, 2])
    x, k = x0, 0
    while abs(2 * x) >= eps:
        x -= al * 2 * x
        k += 1
    _xs, _xx = [], x0
    while abs(2 * _xx) >= eps:
        _xx -= al * 2 * _xx
        _xs.append(_xx)
    _sol = lines(f"Chaque pas divise x par 2 (1 − 2 × {al} = 0.5) : " + " → ".join(fmt(v) for v in [x0] + _xs),
                 "|f'| = |2x| : " + ", ".join(fmt(abs(2 * v)) for v in _xs), f"|f'| < {eps} pour la première fois après {k} pas.")
    ex = Exercise(f"f(x) = x², x₀ = {x0}, α = {al}. On s'arrête dès que |f'(x)| < {eps}. Combien de pas effectue-t-on ?",
                    Number(k), "Simule : à chaque pas x est divisé par 2 (ici 1 − 2α = 0.5). Teste |2x| < seuil après chaque pas.",
                    None)
    ex.solution = _sol
    return ex


def l2_x4(r):
    x, al = r.choice([1, 2, -1]), r.choice([0.1, 0.01, 0.05])
    _sol = lines(f"f'(x) = 4x³, donc f'({x}) = {4 * x ** 3}", f"x ← {x} − {al} × {par(4 * x ** 3)} = {fmt(x - al * 4 * x ** 3)}")
    ex = Exercise(f"f(x) = x⁴. Un pas de descente depuis x = {x} avec α = {al} ?", Number(x - al * 4 * x ** 3), M_GD,
                    gd_path(lambda t: t ** 4, [x], (-2.5, 2.5)))
    ex.solution = _sol
    return ex


def l2_grad_norm(r):
    a, b, n = r.choice([(3, 4, 5), (6, 8, 10), (5, 12, 13)])
    _sol = lines(f"Déplacement = α·∇f ; ‖∇f‖ = √({a}² + {b}²) = {n}", f"Distance = 0.1 × {n} = {0.1 * n:g}")
    ex = Exercise(f"Au point courant, ∇f = ({a}, {b}). Avec α = 0.1, quelle distance parcourt-on en un pas ?",
                    Number(0.1 * n), "Le déplacement est α·∇f ; sa longueur est α·‖∇f‖.", None)
    ex.solution = _sol
    return ex


def l2_ravine(r):
    k = r.choice([10, 25, 50])
    _sol = lines(f"Sur y : y ← y − α·{2 * k}y = (1 − {2 * k}α)·y", f"Il faut |1 − {2 * k}α| < 1 ⇔ 0 < α < 2/{2 * k} = 1/{k}",
                 f"α doit rester < {fmt(F(1, k))} : la direction raide (y) impose un petit pas, qui rend x très lent.")
    ex = Exercise(f"f(x, y) = x² + {k}y². Plus grand α (strictement) pour que la descente ne diverge pas sur y ?",
                    Number(F(1, k)), "Sur y : y ← (1 − 2kα)·y. Il faut |1 − 2kα| < 1. C'est la direction la plus raide qui limite α "
                    "(et rend la descente lente dans l'autre direction).",
                    contour(lambda X, Y: X * X + k * Y * Y, (-3, 3), (-1, 1)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

def l3_newton(r):
    x0 = r.choice([1, 2, 0.5])
    f1, f2 = math.exp(x0) - 2, math.exp(x0)
    _sol = lines("f'(x) = eˣ − 2, f''(x) = eˣ", f"f'({x0}) = {f1:.4f} ; f''({x0}) = {f2:.4f}", f"x₁ = {x0} − ({f1:.4f}) / {f2:.4f} ≈ {x0 - f1 / f2:.4f}",
                 f"(Le vrai minimum est en ln 2 ≈ {math.log(2):.4f}.)")
    ex = Exercise(f"Minimise f(x) = eˣ − 2x par Newton : x₁ = x₀ − f'(x₀)/f''(x₀) avec x₀ = {x0}." + "\n(calculatrice ; 3 décimales)",
                    Number(x0 - f1 / f2, 5e-3), "f'(x) = eˣ − 2, f''(x) = eˣ. Newton utilise la courbure (f'') pour choisir le pas.",
                    curves(lambda t: np.exp(t) - 2 * t, xlim=(-1, 2.5)))
    ex.solution = _sol
    return ex


def l3_newton_quad(r):
    a, h, x0 = r.randint(1, 5), r.randint(-5, 5), r.randint(-9, 9)
    _sol = lines(f"f'(x) = {2 * a}(x {signed(-h)}), f''(x) = {2 * a}", f"f'({x0}) = {2 * a * (x0 - h)}",
                 f"x₁ = {x0} − {par(2 * a * (x0 - h))} / {2 * a} = {x0} {signed(-(x0 - h))} = {h}")
    ex = Exercise(f"f(x) = {a}(x {signed(-h)})². Un pas de Newton depuis x₀ = {x0} ?", Number(h),
                    "x₁ = x₀ − f'(x₀)/f''(x₀). Sur une parabole, Newton tombe pile sur le minimum en un pas.", None)
    ex.solution = _sol
    return ex


def l3_convex(r):
    kind = r.choice(["oui", "non"])
    f = r.choice(["x² + eˣ", "|x| + x²", "eˣ", "x⁴ + 3x²"]) if kind == "oui" else r.choice(["x³", "sin(x)", "−x²", "x⁴ − 5x²"])
    _why = {"x² + eˣ": "f'' = 2 + eˣ > 0", "|x| + x²": "somme de deux fonctions convexes", "eˣ": "f'' = eˣ > 0",
            "x⁴ + 3x²": "f'' = 12x² + 6 > 0", "x³": "f'' = 6x, négatif pour x < 0", "sin(x)": "f'' = −sin x change de signe",
            "−x²": "f'' = −2 < 0", "x⁴ − 5x²": "f'' = 12x² − 10, négatif près de 0"}[f]
    _sol = lines(f"{_why}", "Donc convexe." if kind == "oui" else "Donc pas convexe.")
    ex = Exercise(f"f(x) = {f} est-elle convexe sur ℝ ? (oui / non)", Choice(kind, ["oui", "non"]),
                    "Convexe ⇔ f'' ≥ 0 partout (pour une fonction deux fois dérivable). Une somme de fonctions convexes est convexe.",
                    None)
    ex.solution = _sol
    return ex


def l3_interval(r):
    h = r.randint(-8, 8)
    a, b = sorted(r.sample(range(-6, 7), 2))
    best = min(max(h, a), b)
    _sol = lines(f"Minimum libre de (x {signed(-h)})² : x = {h}",
                 f"{h} est dans [{a}, {b}] : c'est le minimum." if a <= h <= b else f"{h} est hors de [{a}, {b}] : le minimum est à la borne la plus proche, x = {best}.")
    ex = Exercise(f"Minimise f(x) = (x {signed(-h)})² sur l'intervalle [{a}, {b}]. En quel x est le minimum ?", Number(best),
                    "Si le minimum « libre » est dans l'intervalle, c'est lui. Sinon, le minimum est à la borne la plus proche.",
                    curves(lambda t: (t - h) ** 2, xlim=(min(a, h) - 2, max(b, h) + 2)))
    ex.solution = _sol
    return ex


def l3_lagrange(r):
    c = r.choice([2, 4, 6, -2, 10])
    _sol = lines("∇f = (2x, 2y) doit être parallèle à ∇g = (1, 1) : 2x = λ et 2y = λ, donc x = y", f"Contrainte : x + x = {c} ⇒ x = {fmt(F(c, 2))}", f"Point optimal : ({fmt(F(c, 2))}, {fmt(F(c, 2))})")
    ex = Exercise(f"Minimise x² + y² sous la contrainte x + y = {c}. Point optimal ?" + G2, Numbers([c / 2, c / 2], ordered=True),
                    "Lagrange : ∇f = λ·∇g. Ici (2x, 2y) = λ(1, 1) → x = y. Utilise la contrainte. (Ou remplace y = c − x.)",
                    contour(lambda X, Y: X * X + Y * Y, (-6, 6), (-6, 6)))
    ex.solution = _sol
    return ex


def l3_rectangle(r):
    p = r.choice([20, 40, 100, 12])
    _sol = lines(f"Largeur x, longueur {p}/2 − x = {p // 2} − x", f"A(x) = x({p // 2} − x) = {p // 2}x − x²", f"A'(x) = {p // 2} − 2x = 0 ⇔ x = {fmt(F(p, 4))} (c'est un carré)")
    ex = Exercise(f"Rectangle de périmètre {p}. Quelle largeur maximise l'aire ?", Number(p / 4),
                    "Largeur x, longueur p/2 − x. Aire A(x) = x(p/2 − x). Résous A'(x) = 0.", None)
    ex.solution = _sol
    return ex


def l3_x_plus_inv(r):
    a = r.choice([4, 9, 16, 25, 2])
    _sol = lines(f"f'(x) = 1 − {a}/x²", f"f'(x) = 0 ⇔ x² = {a} ⇔ x = √{a} (x > 0)", f"x = {fmt(math.sqrt(a)) if math.sqrt(a).is_integer() else f'{math.sqrt(a):.4f}'}")
    ex = Exercise(f"Minimise f(x) = x + {a}/x pour x > 0. En quel x est le minimum ?", Number(math.sqrt(a)),
                    "f'(x) = 1 − a/x². Résous f'(x) = 0 (garde la solution positive).",
                    curves(lambda t: t + a / t, xlim=(0.2, 3 * math.sqrt(a) + 1), ylim=(0, 6 * math.sqrt(a))))
    ex.solution = _sol
    return ex


def l3_critical(r):
    a = r.randint(1, 4)
    _sol = lines(f"f'(x) = 3x² − {3 * a * a} = 0 ⇔ x = ±{a}", f"f''(x) = 6x : f''({a}) = {6 * a} > 0 → minimum local ; f''(−{a}) < 0 → maximum local", f"Minimum local en x = {a}")
    ex = Exercise(f"f(x) = x³ − {3 * a * a}x. Valeur de x du minimum LOCAL ?", Number(a),
                    "Points critiques : f'(x) = 0. Puis f''(x) > 0 → minimum local.", curves(lambda t: t ** 3 - 3 * a * a * t, xlim=(-2 * a, 2 * a)))
    ex.solution = _sol
    return ex


def l3_second_test(r):
    a, b = nz(r, -4, 4), r.randint(-5, 5)
    _sol = lines(f"f''(1) = {6 * a}", "f'' > 0 → minimum (forme ∪)" if a > 0 else "f'' < 0 → maximum (forme ∩)")
    ex = Exercise(f"f(x) = {poly({3: a, 1: b})}. En x = 1, f''(1) vaut {6 * a}. Si f'(1) = 0, ce point serait un minimum ou un maximum ?",
                    Choice("minimum" if a > 0 else "maximum", ["minimum", "maximum"]),
                    "Test de la dérivée seconde : f'' > 0 → minimum (bol), f'' < 0 → maximum.", None)
    ex.solution = _sol
    return ex


def l3_sum_squares(r):
    pts = [r.randint(-5, 10) for _ in range(r.randint(3, 5))]
    _sol = lines(f"d/dx Σ(x − aᵢ)² = 2Σ(x − aᵢ) = 0 ⇔ {len(pts)}x = Σaᵢ = {sum(pts)}", f"x = {sum(pts)}/{len(pts)} = {fmt(F(sum(pts), len(pts)))} (la moyenne)")
    ex = Exercise(f"Quel x minimise Σ (x − aᵢ)² pour aᵢ ∈ {pts} ?", Number(F(sum(pts), len(pts))),
                    "Dérive par rapport à x : 2Σ(x − aᵢ) = 0. Ce résultat explique pourquoi la MSE est minimisée par la moyenne.",
                    curves(lambda t: sum((t - a) ** 2 for a in pts), xlim=(min(pts) - 2, max(pts) + 2)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 4

def l4_sgd(r):
    w, b, x, y, al = r.randint(-2, 2), r.randint(-2, 2), nz(r, -3, 3), r.randint(-5, 5), 0.1
    e = w * x + b - y
    _sol = lines(f"ŷ = {w} × {par(x)} {signed(b)} = {w * x + b} ; e = ŷ − y = {e}", f"∂L/∂w = 2e·x = {2 * e * x} ; ∂L/∂b = 2e = {2 * e}",
                 f"w ← {w} − 0.1 × {par(2 * e * x)} = {fmt(w - al * 2 * e * x)} ; b ← {b} − 0.1 × {par(2 * e)} = {fmt(b - al * 2 * e)}")
    ex = Exercise(f"SGD sur un point : ŷ = w·x + b, L = (ŷ − y)². (x, y) = ({x}, {y}), w = {w}, b = {b}, α = {al}. "
                    "Nouveaux (w, b) ?\nRéponse : w, b", Numbers([w - al * 2 * e * x, b - al * 2 * e], ordered=True),
                    "∂L/∂w = 2e·x, ∂L/∂b = 2e avec e = ŷ − y. Mets à jour les deux avec les gradients calculés à l'ANCIEN point.",
                    None)
    ex.solution = _sol
    return ex


def l4_batch_grad(r):
    w = r.randint(-1, 2)
    pts = [(nz(r, -3, 3), r.randint(-5, 5)) for _ in range(3)]
    g = sum(2 * (w * x - y) * x for x, y in pts) / 3
    _t = [2 * (w * x - y) * x for x, y in pts]
    _sol = lines("Pour chaque point : 2(w·x − y)·x → " + ", ".join(str(t) for t in _t), f"Moyenne : ({' + '.join(par(t) for t in _t)}) / 3 = {fmt(F(sum(_t), 3))}")
    ex = Exercise(f"MSE sur un mini-batch de 3 points {pts}, modèle ŷ = w·x, w = {w}. Gradient dL/dw (L = moyenne des (ŷ − y)²) ?",
                    Number(g), "dL/dw = (1/n)·Σ 2(w·xᵢ − yᵢ)·xᵢ.", None)
    ex.solution = _sol
    return ex


def l4_l2_step(r):
    w, g, lam, al = r.randint(-5, 5), r.randint(-3, 3), r.choice([0.1, 0.01, 0.5]), 0.1
    _sol = lines(f"Gradient total = g + 2λw = {g} + 2 × {lam} × {par(w)} = {fmt(g + 2 * lam * w)}", f"w ← {w} − 0.1 × {par(round(g + 2 * lam * w, 6))} = {fmt(w - al * (g + 2 * lam * w))}")
    ex = Exercise(f"Perte + λ·w² (régularisation L2). Gradient de la perte seule g = {g}, w = {w}, λ = {lam}, α = {al}. Nouveau w ?",
                    Number(w - al * (g + 2 * lam * w)), "Le gradient total est g + 2λw. La régularisation tire w vers 0 (« weight decay »).",
                    None)
    ex.solution = _sol
    return ex


def l4_decay(r):
    a0, k, t = r.choice([0.1, 1, 0.5]), r.choice([0.1, 0.5, 1]), r.randint(1, 10)
    _sol = lines(f"α_{t} = {a0} / (1 + {k} × {t}) = {a0} / {1 + k * t:g}", f"= {a0 / (1 + k * t):.5g}")
    ex = Exercise(f"Learning rate décroissant α_t = α₀ / (1 + k·t). α₀ = {a0}, k = {k}, t = {t}. α_t ?", Number(a0 / (1 + k * t)),
                    "Remplace dans la formule. On réduit α au fil du temps pour se stabiliser près du minimum.",
                    curves(lambda u: a0 / (1 + k * u), xlim=(0, 20)))
    ex.solution = _sol
    return ex


def l4_exp_decay(r):
    a0, g, t = r.choice([0.1, 1]), r.choice([0.9, 0.5, 0.95]), r.randint(1, 6)
    _sol = lines(f"α_{t} = {a0} × {g}^{t}", f"= {a0} × {g ** t:.5f} ≈ {a0 * g ** t:.4f}")
    ex = Exercise(f"Décroissance exponentielle α_t = α₀·γᵗ avec α₀ = {a0}, γ = {g}, t = {t}. α_t ?" + "\n(calculatrice autorisée)",
                    Number(a0 * g ** t, 1e-3), "Multiplie α₀ par γ, t fois.", curves(lambda u: a0 * g ** u, xlim=(0, 20)))
    ex.solution = _sol
    return ex


def l4_clip(r):
    a, b, n = r.choice([(3, 4, 5), (6, 8, 10), (12, 5, 13)])
    m = r.choice([1, 2])
    _sol = lines(f"‖g‖ = √({a}² + {b}²) = {n} > {m} : on réduit", f"g × {m}/{n} = ({a * m}/{n}, {b * m}/{n}) = ({fmt(F(a * m, n))}, {fmt(F(b * m, n))})")
    ex = Exercise(f"Gradient clipping : g = ({a}, {b}), norme maximale {m}. Gradient après clipping ?" + G2,
                    Numbers([a * m / n, b * m / n], ordered=True),
                    "Si ‖g‖ > max : g ← g · max / ‖g‖ (même direction, longueur ramenée à max).", None)
    ex.solution = _sol
    return ex


def l4_early_stop(r):
    losses = [round(r.uniform(0.3, 1.0), 2) for _ in range(8)]
    best = losses.index(min(losses)) + 1
    _sol = lines(f"Plus petite perte de validation : {min(losses)}", f"Elle est atteinte à l'epoch {best}.")
    ex = Exercise(f"Pertes de validation par epoch : {losses}. Avec early stopping, quelle epoch (numérotée à partir de 1) garde-t-on ?",
                    Number(best), "On garde le modèle de l'epoch où la perte de VALIDATION est minimale.",
                    bars([str(i + 1) for i in range(8)], losses, "perte de validation"))
    ex.solution = _sol
    return ex


def l4_batches(r):
    n, b = r.choice([1000, 50000, 60000, 1234]), r.choice([32, 64, 100, 128])
    _sol = lines(f"{n} / {b} = {n / b:.3f}", f"On arrondit au-dessus (dernier batch incomplet) : {math.ceil(n / b)}")
    ex = Exercise(f"{n} exemples, batch de taille {b}. Combien de mises à jour par epoch (dernier batch incomplet compris) ?",
                    Number(math.ceil(n / b)), "Nombre de batchs = arrondi supérieur de n / taille du batch.", None)
    ex.solution = _sol
    return ex


def l4_iterations(r):
    n, b, e = r.choice([1000, 6400, 3200]), r.choice([32, 64, 100]), r.randint(2, 20)
    _sol = lines(f"Par epoch : ⌈{n}/{b}⌉ = {math.ceil(n / b)}", f"Total : {math.ceil(n / b)} × {e} = {math.ceil(n / b) * e}")
    ex = Exercise(f"{n} exemples, batch de {b}, {e} epochs. Nombre total de mises à jour (dernier batch incomplet compris) ?",
                    Number(math.ceil(n / b) * e), "Mises à jour par epoch × nombre d'epochs.", None)
    ex.solution = _sol
    return ex


def l4_weight_decay(r):
    w, lam, al = r.randint(1, 10), r.choice([0.01, 0.1]), r.choice([0.1, 0.5])
    _sol = lines("w ← w(1 − αλ)", f"= {w} × (1 − {al} × {lam}) = {w} × {1 - al * lam:g} = {w * (1 - al * lam):g}")
    ex = Exercise(f"Weight decay seul (gradient de la perte nul) : w ← w − α·λ·w. w = {w}, α = {al}, λ = {lam}. Nouveau w ?",
                    Number(w * (1 - al * lam)), "Factorise : w·(1 − αλ). Les poids rétrécissent un peu à chaque pas.", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 5

def sig(z):
    return 1 / (1 + math.exp(-z))


def l5_logreg_grad(r):
    w, b, x, y = r.choice([0.5, 1, -1]), r.choice([0, 0.5]), r.choice([1, 2, -1]), r.choice([0, 1])
    p = sig(w * x + b)
    _sol = lines(f"z = {w} × {par(x)} + {b} = {w * x + b:g} ; p = σ(z) ≈ {p:.4f}", f"(p − y)·x = ({p:.4f} − {y}) × {par(x)} ≈ {(p - y) * x:.4f}")
    ex = Exercise(f"Régression logistique p = σ(w·x + b), perte entropie croisée. w = {w}, b = {b}, x = {x}, y = {y}. "
                    "∂L/∂w = (p − y)·x ?" + "\n(calculatrice ; 3 décimales)", Number((p - y) * x, 5e-3),
                    "Calcule p, puis (p − y)·x. Ce gradient très simple vient de la combinaison sigmoïde + entropie croisée.",
                    None)
    ex.solution = _sol
    return ex


def l5_logreg_step(r):
    w, x, y, al = r.choice([0, 1, -1]), r.choice([1, 2]), r.choice([0, 1]), r.choice([0.1, 0.5])
    p = sig(w * x)
    _sol = lines(f"p = σ({w} × {x}) = σ({w * x}) ≈ {p:.4f}", f"gradient = (p − y)·x ≈ ({p:.4f} − {y}) × {x} = {(p - y) * x:.4f}",
                 f"w ← {w} − {al} × {(p - y) * x:.4f} ≈ {w - al * (p - y) * x:.4f}")
    ex = Exercise(f"p = σ(w·x), w = {w}, x = {x}, y = {y}, α = {al}. Nouveau w après un pas de descente ?"
                    "\n(calculatrice ; 3 décimales)", Number(w - al * (p - y) * x, 5e-3),
                    "w ← w − α·(p − y)·x.", None)
    ex.solution = _sol
    return ex


def l5_adam_first(r):
    g = r.choice([2, -4, 0.5, 10])
    al = 0.001
    m = 0.1 * g
    v = 0.001 * g * g
    mh, vh = m / 0.1, v / 0.001
    step = al * mh / (math.sqrt(vh) + 1e-8)
    _sol = lines(f"m = 0.1 × {g} = {m:g} ; v = 0.001 × {g}² = {v:g}", f"m̂ = {m:g} / 0.1 = {mh:g} ; v̂ = {v:g} / 0.001 = {vh:g}",
                 f"pas = {al} × {mh:g} / √{vh:g} = {al} × {mh:g} / {math.sqrt(vh):g} = {step:g}")
    ex = Exercise(f"Adam, premier pas (t = 1), β₁ = 0.9, β₂ = 0.999, α = {al}, gradient g = {g}, m₀ = v₀ = 0.\n"
                    "m = 0.1·g, v = 0.001·g², m̂ = m/(1 − 0.9), v̂ = v/(1 − 0.999), pas = α·m̂/√v̂ (ε négligé). Valeur du pas ?",
                    Number(step, 1e-3),
                    "Après correction du biais, m̂ = g et v̂ = g². Le pas vaut donc α·g/|g| : Adam fait des pas de taille ≈ α "
                    "quelle que soit l'échelle du gradient.", None)
    ex.solution = _sol
    return ex


def l5_condition(r):
    a, b = r.sample([1, 2, 4, 10, 50, 100], 2)
    _sol = lines(f"Hessienne = diag({2 * a}, {2 * b}) : valeurs propres {2 * a} et {2 * b}", f"Conditionnement = {2 * max(a, b)} / {2 * min(a, b)} = {fmt(F(max(a, b), min(a, b)))}")
    ex = Exercise(f"f(x, y) = {a}x² + {b}y². Nombre de conditionnement de la Hessienne (λ_max / λ_min) ?",
                    Number(F(max(a, b), min(a, b))),
                    "Hessienne = diag(2a, 2b). Un grand conditionnement = vallée très allongée = descente de gradient lente.",
                    contour(lambda X, Y: a * X * X + b * Y * Y, (-3, 3), (-3, 3)))
    ex.solution = _sol
    return ex


def l5_opt_fixed(r):
    l1, l2 = r.sample([2, 4, 6, 10, 20], 2)
    _sol = lines(f"α = 2 / ({min(l1, l2)} + {max(l1, l2)}) = 2/{l1 + l2}", f"= {fmt(F(2, l1 + l2))}")
    ex = Exercise(f"Pour une fonction quadratique dont la Hessienne a pour valeurs propres {l1} et {l2}, le pas fixe optimal "
                    "de la descente de gradient est α = 2/(λ_min + λ_max). Calcule-le.", Number(F(2, l1 + l2)),
                    "Ce choix équilibre la convergence le long des deux axes propres.", None)
    ex.solution = _sol
    return ex


def l5_line_search(r):
    a, b = r.randint(1, 3), r.randint(1, 3)
    x, y = nz(r, -3, 3), nz(r, -3, 3)
    g = (2 * a * x, 2 * b * y)
    Hg = (2 * a * g[0], 2 * b * g[1])
    alpha = F(g[0] ** 2 + g[1] ** 2, g[0] * Hg[0] + g[1] * Hg[1])
    _gg, _gHg = g[0] ** 2 + g[1] ** 2, g[0] * Hg[0] + g[1] * Hg[1]
    _sol = lines(f"g = ∇f = ({g[0]}, {g[1]}) ; H = diag({2 * a}, {2 * b})", f"gᵀg = {par(g[0])}² + {par(g[1])}² = {_gg}",
                 f"gᵀHg = {2 * a} × {par(g[0])}² + {2 * b} × {par(g[1])}² = {_gHg}", f"α* = {_gg}/{_gHg} = {fmt(alpha)}")
    ex = Exercise(f"f(x, y) = {a}x² + {b}y², point ({x}, {y}). Pas optimal exact le long de −∇f : α* = (gᵀg)/(gᵀHg) ?",
                    Number(alpha), "Calcule g = ∇f, H = diag(2a, 2b), puis les deux produits.",
                    contour(lambda X, Y: a * X * X + b * Y * Y, (-4, 4), (-4, 4), points=[(x, y)]))
    ex.solution = _sol
    return ex


def l5_saddle(r):
    kind = r.choice(["col", "minimum"])
    expr = "x² − y²" if kind == "col" else "x² + y²"
    _sol = lines("Hessienne = diag(2, −2) : une valeur propre positive, une négative" if kind == "col" else "Hessienne = diag(2, 2) : valeurs propres toutes positives",
                 "Signes mélangés → col (minimum dans une direction, maximum dans l'autre)." if kind == "col" else "→ minimum.")
    ex = Exercise(f"Pour f(x, y) = {expr}, le gradient est nul en (0, 0). Est-ce un minimum ou un col ? (minimum / col)",
                    Choice(kind, ["minimum", "col"]),
                    "Regarde les valeurs propres de la Hessienne : toutes > 0 → minimum ; signes mélangés → col. "
                    "Les cols sont très fréquents dans les réseaux de neurones.",
                    contour(lambda X, Y: X * X - Y * Y if kind == "col" else X * X + Y * Y, (-3, 3), (-3, 3)))
    ex.solution = _sol
    return ex


def l5_softmax_grad(r):
    z = [r.randint(-1, 2) for _ in range(3)]
    k = r.randint(0, 2)
    s = sum(math.exp(v) for v in z)
    p = [math.exp(v) / s for v in z]
    y = [1 if i == k else 0 for i in range(3)]
    _sol = lines("p = softmax(z) : " + ", ".join(f"{v:.4f}" for v in p), f"y = one-hot de la classe {k + 1} : {y}",
                 "p − y = " + ", ".join(f"{a - b:.4f}" for a, b in zip(p, y)))
    ex = Exercise(f"Logits z = {z}, vraie classe {k + 1}. Gradient de l'entropie croisée par rapport aux logits : p − y.\n"
                    "Réponse : 3 composantes (calculatrice ; 3 décimales).", Numbers([a - b for a, b in zip(p, y)], ordered=True, tol=5e-3),
                    "Calcule p = softmax(z), puis retire 1 à la composante de la vraie classe.", bars(["1", "2", "3"], z, "logits"))
    ex.solution = _sol
    return ex


def l5_convex_combo(r):
    a, b, t = r.randint(-5, 5), r.randint(-5, 5), r.choice([0.25, 0.5, 0.75])
    _sol = lines(f"f(a) = {par(a)}² = {a * a} ; f(b) = {par(b)}² = {b * b}", f"t·f(a) + (1 − t)·f(b) = {t} × {a * a} + {1 - t:g} × {b * b} = {t * a * a + (1 - t) * b * b:g}")
    ex = Exercise(f"Pour f convexe, f(t·a + (1−t)·b) ≤ t·f(a) + (1−t)·f(b). Avec f(x) = x², a = {a}, b = {b}, t = {t}, "
                    "calcule le membre de DROITE.", Number(t * a * a + (1 - t) * b * b),
                    "C'est la hauteur de la corde entre (a, f(a)) et (b, f(b)). Pour une fonction convexe la corde est au-dessus de la courbe.",
                    curves(lambda u: u * u, xlim=(min(a, b) - 1, max(a, b) + 1), points=[(a, a * a), (b, b * b)]))
    ex.solution = _sol
    return ex


def l5_lasso(r):
    w, lam = nz(r, -5, 5) / 2, r.choice([0.1, 0.5, 1])
    _sol = lines("d/dw (λ|w|) = λ × signe(w)", f"w = {w:g} est {'positif' if w > 0 else 'négatif'} : λ × {'1' if w > 0 else '(−1)'} = {lam * (1 if w > 0 else -1):g}")
    ex = Exercise(f"Régularisation L1 (Lasso) : pénalité λ·|w|. Avec λ = {lam} et w = {w:g}, dérivée de la pénalité par rapport à w ?",
                    Number(lam * (1 if w > 0 else -1)),
                    "La dérivée de |w| vaut le signe de w (+1 ou −1, pour w ≠ 0). Elle ne diminue pas près de 0 : "
                    "c'est pour ça que L1 pousse des poids exactement à 0.",
                    curves(lambda t: lam * np.abs(t), xlim=(-4, 4), points=[(w, lam * abs(w))]))
    ex.solution = _sol
    return ex


LEVELS = {
    1: [l1_argmin, l1_min_value, l1_step_x2, l1_step_shift, l1_two_steps, l1_direction, l1_factor, l1_diverge, l1_argmax,
        l1_step_size],
    2: [l2_step2d, l2_n_steps, l2_best_alpha, l2_momentum, l2_momentum_step, l2_linreg_step, l2_stop, l2_x4, l2_grad_norm,
        l2_ravine],
    3: [l3_newton, l3_newton_quad, l3_convex, l3_interval, l3_lagrange, l3_rectangle, l3_x_plus_inv, l3_critical,
        l3_second_test, l3_sum_squares],
    4: [l4_sgd, l4_batch_grad, l4_l2_step, l4_decay, l4_exp_decay, l4_clip, l4_early_stop, l4_batches, l4_iterations,
        l4_weight_decay],
    5: [l5_logreg_grad, l5_logreg_step, l5_adam_first, l5_condition, l5_opt_fixed, l5_line_search, l5_saddle,
        l5_softmax_grad, l5_convex_combo, l5_lasso],
}
