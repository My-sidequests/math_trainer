"""Vecteurs : opérations, produit scalaire, 3D, combinaisons linéaires, vecteurs en IA."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Number, Numbers
from ..core import Exercise, bars, nz, vec, vectors

from ..core import fmt, lines, par  # noqa: E402,F811


def dstr(u, v) -> str:
    """Développement d'un produit scalaire : « 2 × 3 + (−1) × 4 »."""
    return " + ".join(f"{par(a)} × {par(b)}" for a, b in zip(u, v))


def sq(u) -> str:
    """Somme des carrés développée : « 3² + (−4)² »."""
    return " + ".join(f"{par(c)}²" for c in u)

DESCRIPTIONS = {
    1: "Opérations : addition, multiplication par un nombre, coordonnées, norme",
    2: "Produit scalaire, orthogonalité, vecteur unitaire, colinéarité",
    3: "Vecteurs en 3D : produit scalaire, produit vectoriel, projection",
    4: "Combinaisons linéaires, indépendance, bases, normes L1 / L∞",
    5: "Vecteurs en IA : similarité cosinus, distances, embeddings, neurone",
}

GRID = "\nRéponse : les composantes séparées par des virgules (ex : 3, -1)."
M_OPS = ("Les opérations se font composante par composante :\n"
         "  u + v = (u₁ + v₁, u₂ + v₂)    k·u = (k·u₁, k·u₂)    u − v = (u₁ − v₁, u₂ − v₂)\n"
         "Géométriquement : u + v = mettre v au bout de u ; k·u = étirer u (et le retourner si k < 0).")
M_DOT = ("Produit scalaire : u·v = u₁v₁ + u₂v₂ (+ u₃v₃ en 3D). C'est un NOMBRE.\n"
         "u·v = ‖u‖·‖v‖·cos θ : positif si angle aigu, nul si perpendiculaires, négatif si obtus.\n"
         "Norme : ‖u‖ = √(u·u) = √(u₁² + u₂²).")
TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 6, 10), (0, 7, 7), (9, 12, 15)]


def v2(r, lo=-5, hi=5):
    while True:
        v = (r.randint(lo, hi), r.randint(lo, hi))
        if v != (0, 0):
            return v


def v3(r, lo=-4, hi=4):
    while True:
        v = tuple(r.randint(lo, hi) for _ in range(3))
        if any(v):
            return v


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def norm(u):
    return math.sqrt(dot(u, u))


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


# ================================================================ NIVEAU 1

def l1_add(r):
    u, v = v2(r), v2(r)
    _sol = lines(f"u + v = ({u[0]} + {par(v[0])}, {u[1]} + {par(v[1])})", f"= {vec((u[0] + v[0], u[1] + v[1]))}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}. Calcule u + v." + GRID, Numbers([u[0] + v[0], u[1] + v[1]], ordered=True),
                    M_OPS, vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l1_sub(r):
    u, v = v2(r), v2(r)
    _sol = lines(f"u − v = ({u[0]} − {par(v[0])}, {u[1]} − {par(v[1])})", f"= {vec((u[0] - v[0], u[1] - v[1]))}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}. Calcule u − v." + GRID, Numbers([u[0] - v[0], u[1] - v[1]], ordered=True),
                    M_OPS, vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l1_scale(r):
    u, k = v2(r), nz(r, -4, 4)
    _sol = lines(f"{k}·u = ({k} × {par(u[0])}, {k} × {par(u[1])})", f"= {vec((k * u[0], k * u[1]))}")
    ex = Exercise(f"u = {vec(u)}. Calcule {k}·u." + GRID, Numbers([k * u[0], k * u[1]], ordered=True), M_OPS,
                    vectors([u], labels=["u"]))
    ex.solution = _sol
    return ex


def l1_AB(r):
    A, B = v2(r), v2(r)
    _sol = lines(f"AB = B − A = ({B[0]} − {par(A[0])}, {B[1]} − {par(A[1])})", f"= {vec((B[0] - A[0], B[1] - A[1]))}")
    ex = Exercise(f"A = {vec(A)}, B = {vec(B)}. Calcule les coordonnées du vecteur AB." + GRID,
                    Numbers([B[0] - A[0], B[1] - A[1]], ordered=True),
                    "AB = B − A : arrivée moins départ, composante par composante.", vectors([A, B], labels=["A", "B"]))
    ex.solution = _sol
    return ex


def l1_mid(r):
    A, B = v2(r), v2(r)
    _sol = lines(f"Milieu = (({A[0]} + {par(B[0])})/2, ({A[1]} + {par(B[1])})/2)", f"= ({A[0] + B[0]}/2, {A[1] + B[1]}/2) = {vec((F(A[0] + B[0], 2), F(A[1] + B[1], 2)))}")
    ex = Exercise(f"A = {vec(A)}, B = {vec(B)}. Coordonnées du milieu de [AB] ?" + GRID,
                    Numbers([(A[0] + B[0]) / 2, (A[1] + B[1]) / 2], ordered=True),
                    "Milieu = moyenne des coordonnées : ((x_A + x_B)/2, (y_A + y_B)/2).", vectors([A, B], labels=["A", "B"]))
    ex.solution = _sol
    return ex


def l1_norm(r):
    a, b, c = r.choice(TRIPLES)
    u = (a * r.choice([1, -1]), b * r.choice([1, -1]))
    _sol = lines(f"‖u‖ = √({sq(u)}) = √({u[0] ** 2} + {u[1] ** 2}) = √{c * c}", f"= {c}")
    ex = Exercise(f"Calcule la norme (longueur) de u = {vec(u)}.", Number(c), M_DOT + "\nC'est Pythagore.",
                    vectors([u], labels=["u"]))
    ex.solution = _sol
    return ex


def l1_opposite(r):
    u = v2(r)
    _sol = lines("−u : on change le signe de chaque composante.", f"−u = {vec((-u[0], -u[1]))}")
    ex = Exercise(f"u = {vec(u)}. Donne le vecteur opposé −u." + GRID, Numbers([-u[0], -u[1]], ordered=True), M_OPS,
                    vectors([u], labels=["u"]))
    ex.solution = _sol
    return ex


def l1_read(r):
    u = v2(r, -4, 4)
    _sol = lines(f"Depuis l'origine, la flèche avance de {u[0]} horizontalement et de {u[1]} verticalement.", f"u = {vec(u)}")
    ex = Exercise("Lis les coordonnées du vecteur tracé (il part de l'origine)." + GRID, Numbers(list(u), ordered=True),
                    "Coordonnées = déplacement horizontal puis vertical pour aller du départ à la pointe.",
                    vectors([u], lim=5))
    ex.solution = _sol
    return ex


def l1_combo(r):
    u, v = v2(r, -3, 3), v2(r, -3, 3)
    a, b = nz(r, -3, 3), nz(r, -3, 3)
    _au, _bv = (a * u[0], a * u[1]), (b * v[0], b * v[1])
    _sol = lines(f"{a}u = {vec(_au)} ; {b}v = {vec(_bv)}", f"Somme : ({_au[0]} + {par(_bv[0])}, {_au[1]} + {par(_bv[1])}) = {vec((_au[0] + _bv[0], _au[1] + _bv[1]))}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}. Calcule {a}u + {b}v.".replace("+ -", "− ") + GRID,
                    Numbers([a * u[0] + b * v[0], a * u[1] + b * v[1]], ordered=True),
                    M_OPS + "\nCalcule chaque multiple, puis additionne.", vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l1_equal(r):
    k = r.randint(-6, 6)
    u2 = r.randint(-5, 5)
    _sol = lines(f"Égalité des 1res composantes : 2k − 1 = {2 * k - 1}", f"2k = {2 * k}", f"k = {k}")
    ex = Exercise(f"u = (2k − 1, {u2}) et v = ({2 * k - 1}, {u2}). Pour quelle valeur de k a-t-on u = v ?", Number(k),
                    "Deux vecteurs sont égaux si TOUTES leurs composantes sont égales. Écris l'équation sur la 1re composante.",
                    None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 2

def l2_dot(r):
    u, v = v2(r), v2(r)
    _sol = lines(f"u·v = {dstr(u, v)}", f"= {u[0] * v[0]} + {par(u[1] * v[1])} = {dot(u, v)}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}. Calcule le produit scalaire u·v.", Number(dot(u, v)), M_DOT,
                    vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l2_norm_sqrt(r):
    u = v2(r)
    _sol = lines(f"‖u‖ = √({sq(u)}) = √{dot(u, u)}", f"≈ {norm(u):.4f}")
    ex = Exercise(f"Calcule ‖u‖ pour u = {vec(u)} (exact avec sqrt, ou décimal précis).", Number(norm(u)), M_DOT,
                    vectors([u], labels=["u"]))
    ex.solution = _sol
    return ex


def l2_unit(r):
    a, b, c = r.choice(TRIPLES)
    u = (a * r.choice([1, -1]), b * r.choice([1, -1]))
    _sol = lines(f"‖u‖ = √({sq(u)}) = {c}", f"u / ‖u‖ = ({u[0]}/{c}, {u[1]}/{c}) = {vec((F(u[0], c), F(u[1], c)))}")
    ex = Exercise(f"Normalise u = {vec(u)} : donne le vecteur unitaire u / ‖u‖." + GRID,
                    Numbers([F(u[0], c), F(u[1], c)], ordered=True),
                    "Vecteur unitaire = même direction, longueur 1 : divise chaque composante par la norme.",
                    vectors([u], labels=["u"]))
    ex.solution = _sol
    return ex


def l2_orth(r):
    u = v2(r)
    k = r.choice([1, 2, -1])
    v = (-u[1] * k, u[0] * k) if r.random() < 0.5 else v2(r)
    _sol = lines(f"u·v = {dstr(u, v)} = {dot(u, v)}", "u·v = 0 : ils sont orthogonaux." if dot(u, v) == 0 else "u·v ≠ 0 : ils ne sont pas orthogonaux.")
    ex = Exercise(f"u = {vec(u)} et v = {vec(v)} sont-ils orthogonaux (perpendiculaires) ? (oui / non)",
                    Choice("oui" if dot(u, v) == 0 else "non", ["oui", "non"]),
                    "Orthogonaux ⇔ u·v = 0. Calcule le produit scalaire.", vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l2_orth_k(r):
    u = (nz(r, -5, 5), nz(r, -5, 5))
    a = nz(r, -4, 4)
    _sol = lines(f"u·v = {u[0]} × {par(a)} + {u[1]} × k = {u[0] * a} + {u[1]}k", f"u·v = 0 ⇔ {u[1]}k = {fmt(-u[0] * a)}",
                 f"k = {fmt(-u[0] * a)} / {par(u[1])} = {fmt(F(-u[0] * a, u[1]))}")
    ex = Exercise(f"u = {vec(u)}, v = ({a}, k). Pour quelle valeur de k sont-ils orthogonaux ?",
                    Number(F(-u[0] * a, u[1])), "Écris u·v = 0 : c'est une équation du premier degré en k.",
                    vectors([u], labels=["u"]))
    ex.solution = _sol
    return ex


def l2_angle(r):
    u, v = v2(r), v2(r)
    _c = dot(u, v) / (norm(u) * norm(v))
    _sol = lines(f"u·v = {dstr(u, v)} = {dot(u, v)}", f"‖u‖ = √{dot(u, u)} ≈ {norm(u):.4f} ; ‖v‖ = √{dot(v, v)} ≈ {norm(v):.4f}",
                 f"cos θ = {dot(u, v)} / ({norm(u):.4f} × {norm(v):.4f}) ≈ {_c:.4f}", f"θ = arccos({_c:.4f}) ≈ {math.degrees(math.acos(max(-1, min(1, _c)))):.3f}°")
    ex = Exercise(f"Angle (en degrés) entre u = {vec(u)} et v = {vec(v)} ?\n(calculatrice autorisée ; 0.01 près)",
                    Number(math.degrees(math.acos(max(-1, min(1, dot(u, v) / (norm(u) * norm(v)))))), 5e-3),
                    M_DOT + "\ncos θ = u·v / (‖u‖·‖v‖), puis θ = arccos(...).", vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l2_dist(r):
    a, b, c = r.choice(TRIPLES)
    A = v2(r, -3, 3)
    B = (A[0] + a, A[1] - b)
    _sol = lines(f"AB = ({B[0]} − {par(A[0])}, {B[1]} − {par(A[1])}) = {vec((a, -b))}", f"‖AB‖ = √({a}² + {b}²) = √{c * c} = {c}")
    ex = Exercise(f"Distance entre A = {vec(A)} et B = {vec(B)} ?", Number(c),
                    "Distance = norme du vecteur AB = √((x_B − x_A)² + (y_B − y_A)²).", vectors([A, B], labels=["A", "B"], lim=12))
    ex.solution = _sol
    return ex


def l2_scalar_proj(r):
    u = r.choice([(1, 0), (0, 1), (F(3, 5), F(4, 5)), (F(4, 5), F(-3, 5))])
    v = v2(r)
    _sol = lines("u est unitaire : la projection scalaire vaut u·v.", f"u·v = {dstr(u, v)} = {fmt(dot(u, v))}")
    ex = Exercise(f"u = {vec(u)} est unitaire. Longueur (signée) de la projection de v = {vec(v)} sur u ?",
                    Number(dot(u, v)), "Si u est unitaire, la projection scalaire de v sur u est simplement u·v.",
                    vectors([tuple(float(c) for c in u), v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l2_colinear(r):
    u = v2(r, -4, 4)
    k = r.choice([2, -1, 3, F(1, 2)])
    v = (u[0] * k, u[1] * k) if r.random() < 0.5 else v2(r, -6, 6)
    det = u[0] * v[1] - u[1] * v[0]
    _sol = lines(f"det(u, v) = u₁v₂ − u₂v₁ = {u[0]} × {par(v[1])} − {par(u[1])} × {par(v[0])} = {fmt(det)}",
                 "det = 0 : colinéaires." if det == 0 else "det ≠ 0 : pas colinéaires.")
    ex = Exercise(f"u = {vec(u)} et v = {vec(v)} sont-ils colinéaires (même direction ou opposée) ? (oui / non)",
                    Choice("oui" if det == 0 else "non", ["oui", "non"]),
                    "Colinéaires ⇔ l'un est un multiple de l'autre ⇔ u₁v₂ − u₂v₁ = 0 (déterminant nul).",
                    vectors([u, tuple(float(c) for c in v)], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l2_det_area(r):
    u, v = v2(r), v2(r)
    _d = u[0] * v[1] - u[1] * v[0]
    _sol = lines(f"det(u, v) = {u[0]} × {par(v[1])} − {par(u[1])} × {par(v[0])} = {_d}", f"Aire = |det| = {abs(_d)}")
    ex = Exercise(f"Aire du parallélogramme construit sur u = {vec(u)} et v = {vec(v)} ?",
                    Number(abs(u[0] * v[1] - u[1] * v[0])),
                    "Aire = |det(u, v)| = |u₁v₂ − u₂v₁|.", vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

G3 = "\nRéponse : 3 composantes séparées par des virgules."


def l3_add(r):
    u, v = v3(r), v3(r)
    a, b = nz(r, -3, 3), nz(r, -3, 3)
    _r = [a * x + b * y for x, y in zip(u, v)]
    _sol = lines(f"{a}u = {vec([a * c for c in u])} ; {b}v = {vec([b * c for c in v])}", f"Somme composante par composante : {vec(_r)}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}. Calcule {a}u + {b}v.".replace("+ -", "− ") + G3,
                    Numbers([a * x + b * y for x, y in zip(u, v)], ordered=True), M_OPS, None)
    ex.solution = _sol
    return ex


def l3_dot(r):
    u, v = v3(r), v3(r)
    _sol = lines(f"u·v = {dstr(u, v)}", f"= {' + '.join(par(a * b) for a, b in zip(u, v))} = {dot(u, v)}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}. Calcule u·v.", Number(dot(u, v)), M_DOT, None)
    ex.solution = _sol
    return ex


def l3_norm(r):
    u = r.choice([(1, 2, 2), (2, 3, 6), (1, 4, 8), (2, 6, 9), (4, 4, 7), (2, 10, 11)])
    s = tuple(c * r.choice([1, -1]) for c in u)
    _sol = lines(f"‖u‖ = √({sq(s)}) = √{dot(s, s)}", f"= {int(norm(s))}")
    ex = Exercise(f"Calcule ‖u‖ pour u = {vec(s)}.", Number(norm(s)), M_DOT + "\nEn 3D : √(u₁² + u₂² + u₃²).", None)
    ex.solution = _sol
    return ex


def l3_cross(r):
    u, v = v3(r, -3, 3), v3(r, -3, 3)
    _c = cross(u, v)
    _sol = lines(f"1re : u₂v₃ − u₃v₂ = {par(u[1])} × {par(v[2])} − {par(u[2])} × {par(v[1])} = {_c[0]}",
                 f"2e : u₃v₁ − u₁v₃ = {par(u[2])} × {par(v[0])} − {par(u[0])} × {par(v[2])} = {_c[1]}",
                 f"3e : u₁v₂ − u₂v₁ = {par(u[0])} × {par(v[1])} − {par(u[1])} × {par(v[0])} = {_c[2]}", f"u × v = {vec(_c)}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}. Calcule le produit vectoriel u × v." + G3,
                    Numbers(list(cross(u, v)), ordered=True),
                    "u × v = (u₂v₃ − u₃v₂, u₃v₁ − u₁v₃, u₁v₂ − u₂v₁).\nLe résultat est perpendiculaire à u et à v.", None)
    ex.solution = _sol
    return ex


def l3_orth(r):
    u = v3(r)
    v = cross(u, v3(r)) if r.random() < 0.5 else v3(r)
    if not any(v):
        v = (1, 0, 0)
    _sol = lines(f"u·v = {dstr(u, v)} = {dot(u, v)}", "= 0 : orthogonaux." if dot(u, v) == 0 else "≠ 0 : pas orthogonaux.")
    ex = Exercise(f"u = {vec(u)} et v = {vec(v)} sont-ils orthogonaux ? (oui / non)",
                    Choice("oui" if dot(u, v) == 0 else "non", ["oui", "non"]), M_DOT, None)
    ex.solution = _sol
    return ex


def l3_proj(r):
    u = r.choice([(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 1), (1, 2, 2)])
    v = v3(r)
    k = F(dot(u, v), dot(u, u))
    _sol = lines(f"u·v = {dstr(u, v)} = {dot(u, v)} ; u·u = {dot(u, u)}", f"coefficient = {dot(u, v)}/{dot(u, u)} = {fmt(k)}",
                 f"proj = {fmt(k)} × {vec(u)} = {vec([k * c for c in u])}")
    ex = Exercise(f"Projection orthogonale du vecteur v = {vec(v)} sur u = {vec(u)} (un vecteur)." + G3,
                    Numbers([k * c for c in u], ordered=True),
                    "proj_u(v) = (u·v / u·u) · u. Calcule le coefficient, puis multiplie u par ce coefficient.", None)
    ex.solution = _sol
    return ex


def l3_dist(r):
    A = v3(r)
    d = r.choice([(1, 2, 2), (2, 3, 6), (1, 4, 8)])
    B = tuple(a + b * r.choice([1, -1]) for a, b in zip(A, d))
    _dd = tuple(b - a for a, b in zip(A, B))
    _sol = lines(f"B − A = {vec(_dd)}", f"‖B − A‖ = √({sq(_dd)}) = √{dot(_dd, _dd)} = {int(norm(_dd))}")
    ex = Exercise(f"Distance entre A = {vec(A)} et B = {vec(B)} ?", Number(norm(d)),
                    "Distance = ‖B − A‖ (Pythagore en 3D).", None)
    ex.solution = _sol
    return ex


def l3_angle(r):
    u, v = v3(r), v3(r)
    _c = dot(u, v) / (norm(u) * norm(v))
    _sol = lines(f"u·v = {dot(u, v)} ; ‖u‖ ≈ {norm(u):.4f} ; ‖v‖ ≈ {norm(v):.4f}", f"cos θ ≈ {_c:.4f}",
                 f"θ ≈ {math.degrees(math.acos(max(-1, min(1, _c)))):.3f}°")
    ex = Exercise(f"Angle (en degrés) entre u = {vec(u)} et v = {vec(v)} ?\n(calculatrice autorisée ; 0.01 près)",
                    Number(math.degrees(math.acos(max(-1, min(1, dot(u, v) / (norm(u) * norm(v)))))), 5e-3),
                    M_DOT + "\ncos θ = u·v / (‖u‖·‖v‖).", None)
    ex.solution = _sol
    return ex


def l3_unit(r):
    u = r.choice([(1, 2, 2), (2, 3, 6), (2, 6, 9), (4, 4, 7)])
    s = tuple(c * r.choice([1, -1]) for c in u)
    n = int(norm(s))
    _sol = lines(f"‖u‖ = √({sq(s)}) = {n}", f"u / ‖u‖ = {vec([F(c, n) for c in s])}")
    ex = Exercise(f"Vecteur unitaire dans la direction de u = {vec(s)} ?" + G3, Numbers([F(c, n) for c in s], ordered=True),
                    "Divise chaque composante par ‖u‖.", None)
    ex.solution = _sol
    return ex


def l3_triangle_area(r):
    u, v = v3(r, -3, 3), v3(r, -3, 3)
    c = cross(u, v)
    if not any(c):
        v = (v[0] + 1, v[1], v[2] + 2)
        c = cross(u, v)
    _sol = lines(f"u × v = {vec(c)}", f"‖u × v‖ = √({sq(c)}) = √{dot(c, c)} ≈ {norm(c):.4f}", f"Aire du triangle = ‖u × v‖ / 2 ≈ {norm(c) / 2:.4f}")
    ex = Exercise(f"Aire du triangle construit sur u = {vec(u)} et v = {vec(v)} (exact avec sqrt ou décimal précis) ?",
                    Number(norm(c) / 2), "‖u × v‖ est l'aire du parallélogramme ; le triangle en est la moitié.", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 4

def l4_express(r):
    while True:
        u, v = v2(r, -3, 3), v2(r, -3, 3)
        if u[0] * v[1] - u[1] * v[0] != 0:
            break
    a, b = nz(r, -3, 3), nz(r, -3, 3)
    w = (a * u[0] + b * v[0], a * u[1] + b * v[1])
    _det = u[0] * v[1] - u[1] * v[0]
    _sol = lines(f"Système : a·{par(u[0])} + b·{par(v[0])} = {w[0]}  et  a·{par(u[1])} + b·{par(v[1])} = {w[1]}",
                 f"Déterminant : {u[0]} × {par(v[1])} − {par(v[0])} × {par(u[1])} = {_det}",
                 f"a = ({w[0]} × {par(v[1])} − {par(v[0])} × {par(w[1])}) / {par(_det)} = {a}",
                 f"b = ({u[0]} × {par(w[1])} − {par(w[0])} × {par(u[1])}) / {par(_det)} = {b}")
    ex = Exercise(f"u = {vec(u)}, v = {vec(v)}, w = {vec(w)}. Trouve a et b tels que w = a·u + b·v.\nRéponse : a, b",
                    Numbers([a, b], ordered=True), "Écris l'égalité composante par composante : un système de 2 équations "
                    "à 2 inconnues (a, b).", vectors([u, v, w], labels=["u", "v", "w"]))
    ex.solution = _sol
    return ex


def l4_independent(r):
    u = v3(r, -3, 3)
    v = v3(r, -3, 3)
    dep = r.random() < 0.5
    w = tuple(a + b for a, b in zip(u, v)) if dep else v3(r, -3, 3)
    det = np.linalg.det(np.array([u, v, w], dtype=float))
    ans = "dependants" if abs(det) < 1e-9 else "independants"
    _sol = lines(f"det [u ; v ; w] ≈ {det:.4f}" + (" (= 0)" if abs(det) < 1e-9 else ""),
                 f"Ici w = u + v : {vec(u)} + {vec(v)} = {vec(w)}, donc ils sont dépendants." if dep else
                 ("det ≠ 0 : aucun n'est combinaison des autres, ils sont indépendants." if abs(det) > 1e-9 else "det = 0 : ils sont dépendants."))
    ex = Exercise(f"Les vecteurs u = {vec(u)}, v = {vec(v)}, w = {vec(w)} sont-ils linéairement indépendants ou "
                    "dépendants ? (indépendants / dépendants)", Choice(ans, ["indépendants", "dépendants"]),
                    "Dépendants ⇔ l'un est combinaison linéaire des autres ⇔ le déterminant de la matrice formée par "
                    "les 3 vecteurs est nul. Regarde aussi si w = u + v, w = u − v, etc.", None)
    ex.solution = _sol
    return ex


def l4_basis(r):
    u, v = v2(r, -4, 4), v2(r, -4, 4)
    if r.random() < 0.4:
        k = nz(r, -2, 2)
        v = (k * u[0], k * u[1])
    det = u[0] * v[1] - u[1] * v[0]
    _sol = lines(f"det(u, v) = {u[0]} × {par(v[1])} − {par(u[1])} × {par(v[0])} = {det}",
                 "det ≠ 0 : non colinéaires, ils forment une base." if det != 0 else "det = 0 : colinéaires, pas une base.")
    ex = Exercise(f"u = {vec(u)} et v = {vec(v)} forment-ils une base du plan ? (oui / non)",
                    Choice("oui" if det != 0 else "non", ["oui", "non"]),
                    "Deux vecteurs du plan forment une base ⇔ ils ne sont pas colinéaires ⇔ det(u, v) ≠ 0.",
                    vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l4_span_dim(r):
    u = v3(r, -3, 3)
    kind = r.choice([1, 2, 3])
    if kind == 1:
        vs = [u, tuple(2 * c for c in u), tuple(-c for c in u)]
    elif kind == 2:
        v = v3(r, -3, 3)
        while not any(cross(u, v)):
            v = v3(r, -3, 3)
        vs = [u, v, tuple(a + b for a, b in zip(u, v))]
    else:
        vs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] if r.random() < 0.3 else [u, (1, 0, 1), (0, 1, 1)]
    rank = np.linalg.matrix_rank(np.array(vs, dtype=float))
    _why = {1: "les trois vecteurs sont des multiples du premier : une seule direction",
            2: "le 3e est la somme des deux premiers (qui ne sont pas colinéaires) : deux directions indépendantes",
            3: "aucun n'est combinaison des autres"}[kind]
    _sol = lines(f"Ici {_why}.", f"Rang (nombre de vecteurs indépendants) = {rank}, donc dimension {rank}.")
    ex = Exercise(f"Quelle est la dimension de l'espace engendré par {', '.join(vec(v) for v in vs)} ?", Number(rank),
                    "Dimension = nombre maximal de vecteurs indépendants parmi eux (le rang). Cherche si certains "
                    "sont des multiples ou des sommes des autres.", None)
    ex.solution = _sol
    return ex


def l4_coords(r):
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    w = (a + b, a - b)
    _sol = lines(f"a·(1, 1) + b·(1, −1) = (a + b, a − b) = {vec(w)}", f"a + b = {w[0]} et a − b = {w[1]}",
                 f"En additionnant : 2a = {w[0] + w[1]} ⇒ a = {a} ; en soustrayant : 2b = {w[0] - w[1]} ⇒ b = {b}")
    ex = Exercise(f"Dans la base e₁ = (1, 1), e₂ = (1, −1), quelles sont les coordonnées de w = {vec(w)} ?\nRéponse : a, b "
                    "tels que w = a·e₁ + b·e₂", Numbers([a, b], ordered=True),
                    "Résous a + b = w₁ et a − b = w₂ (additionne et soustrais les équations).",
                    vectors([(1, 1), (1, -1), w], labels=["e₁", "e₂", "w"]))
    ex.solution = _sol
    return ex


def l4_gram(r):
    u = r.choice([(1, 0), (0, 1), (1, 1)])
    v = v2(r)
    k = F(dot(u, v), dot(u, u))
    _p = (k * u[0], k * u[1])
    _sol = lines(f"u·v = {dot(u, v)}, u·u = {dot(u, u)} ⇒ proj_u(v) = {fmt(k)} × {vec(u)} = {vec(_p)}",
                 f"v − proj = {vec(v)} − {vec(_p)} = {vec((v[0] - _p[0], v[1] - _p[1]))}", "Vérification : son produit scalaire avec u vaut 0 ✔")
    ex = Exercise(f"Gram-Schmidt : rends v = {vec(v)} orthogonal à u = {vec(u)} en calculant v − proj_u(v)." + GRID,
                    Numbers([v[0] - k * u[0], v[1] - k * u[1]], ordered=True),
                    "proj_u(v) = (u·v / u·u)·u. Retire cette projection à v : le reste est perpendiculaire à u.",
                    vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l4_orthonormal(r):
    good = r.random() < 0.5
    if good:
        u, v = r.choice([((1, 0), (0, 1)), ((F(3, 5), F(4, 5)), (F(-4, 5), F(3, 5))), ((F(4, 5), F(3, 5)), (F(3, 5), F(-4, 5)))])
    else:
        u, v = r.choice([((1, 1), (1, -1)), ((F(3, 5), F(4, 5)), (F(4, 5), F(3, 5))), ((1, 0), (1, 1))])
    ok = dot(u, v) == 0 and dot(u, u) == 1 and dot(v, v) == 1
    _sol = lines(f"u·v = {fmt(dot(u, v))} ; ‖u‖² = {fmt(dot(u, u))} ; ‖v‖² = {fmt(dot(v, v))}",
                 "Orthogonaux et de norme 1 : base orthonormée." if ok else "Une des conditions (u·v = 0, normes = 1) n'est pas remplie : pas orthonormée.")
    ex = Exercise(f"La base u = {vec(u)}, v = {vec(v)} est-elle orthonormée ? (oui / non)",
                    Choice("oui" if ok else "non", ["oui", "non"]),
                    "Orthonormée = vecteurs orthogonaux deux à deux (u·v = 0) ET de norme 1 (u·u = 1).",
                    vectors([tuple(float(c) for c in u), tuple(float(c) for c in v)], labels=["u", "v"], lim=1.5))
    ex.solution = _sol
    return ex


def l4_l1(r):
    u = [nz(r, -6, 6) for _ in range(4)]
    _sol = lines(f"‖u‖₁ = {' + '.join(str(abs(c)) for c in u)} = {sum(abs(c) for c in u)}")
    ex = Exercise(f"Norme L1 de u = {vec(u)} ?", Number(sum(abs(c) for c in u)),
                    "‖u‖₁ = somme des valeurs absolues (distance « Manhattan »).", bars(range(1, 5), u))
    ex.solution = _sol
    return ex


def l4_linf(r):
    u = [nz(r, -9, 9) for _ in range(5)]
    _sol = lines(f"Valeurs absolues : {', '.join(str(abs(c)) for c in u)}", f"La plus grande : {max(abs(c) for c in u)}")
    ex = Exercise(f"Norme L∞ de u = {vec(u)} ?", Number(max(abs(c) for c in u)),
                    "‖u‖∞ = la plus grande valeur absolue des composantes.", bars(range(1, 6), u))
    ex.solution = _sol
    return ex


def l4_l2_4d(r):
    u = r.choice([(1, 1, 1, 1), (2, 2, 2, 2), (1, 2, 2, 4), (3, 4, 0, 0), (2, 4, 5, 6)])
    s = tuple(c * r.choice([1, -1]) for c in u)
    _sol = lines(f"‖u‖ = √({sq(s)}) = √{dot(s, s)}", f"= {fmt(norm(s))}" if norm(s).is_integer() else f"≈ {norm(s):.4f}")
    ex = Exercise(f"Norme L2 (euclidienne) de u = {vec(s)} (dimension 4) ?", Number(norm(s)),
                    "Même formule qu'en 2D ou 3D : racine de la somme des carrés, quelle que soit la dimension.",
                    bars(range(1, 5), s))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 5

def l5_cosine(r):
    u, v = v3(r), v3(r)
    _sol = lines(f"u·v = {dstr(u, v)} = {dot(u, v)}", f"‖u‖ = √{dot(u, u)} ≈ {norm(u):.4f} ; ‖v‖ = √{dot(v, v)} ≈ {norm(v):.4f}",
                 f"cos = {dot(u, v)} / ({norm(u):.4f} × {norm(v):.4f}) ≈ {dot(u, v) / (norm(u) * norm(v)):.4f}")
    ex = Exercise(f"Similarité cosinus entre les embeddings u = {vec(u)} et v = {vec(v)} ? (3 décimales)",
                    Number(dot(u, v) / (norm(u) * norm(v)), 5e-3),
                    "cos(u, v) = u·v / (‖u‖·‖v‖). 1 = même direction, 0 = sans rapport, −1 = opposés.", None)
    ex.solution = _sol
    return ex


def l5_euclid(r):
    u = [r.randint(-3, 3) for _ in range(4)]
    v = [r.randint(-3, 3) for _ in range(4)]
    _dd = [a - b for a, b in zip(u, v)]
    _sol = lines(f"u − v = {vec(_dd)}", f"‖u − v‖ = √({sq(_dd)}) = √{dot(_dd, _dd)} ≈ {norm(_dd):.4f}")
    ex = Exercise(f"Distance euclidienne entre u = {vec(u)} et v = {vec(v)} ? (exact ou décimal précis)",
                    Number(norm([a - b for a, b in zip(u, v)])), "‖u − v‖ : différence composante par composante, puis norme.",
                    None)
    ex.solution = _sol
    return ex


def l5_nearest(r):
    q = v2(r, -3, 3)
    cands = [v2(r, -4, 4) for _ in range(3)]
    d = [norm((c[0] - q[0], c[1] - q[1])) for c in cands]
    while len(set(round(x, 6) for x in d)) < 3:
        cands = [v2(r, -4, 4) for _ in range(3)]
        d = [norm((c[0] - q[0], c[1] - q[1])) for c in cands]
    _d2 = [(c[0] - q[0]) ** 2 + (c[1] - q[1]) ** 2 for c in cands]
    _sol = lines("Distances au carré (pas besoin de racine pour comparer) :",
                 *[f"vecteur {i + 1} : ({c[0]} − {par(q[0])})² + ({c[1]} − {par(q[1])})² = {d2}" for i, (c, d2) in enumerate(zip(cands, _d2))],
                 f"La plus petite est celle du vecteur {_d2.index(min(_d2)) + 1}.")
    ex = Exercise(f"Requête q = {vec(q)}. Lequel des vecteurs 1 : {vec(cands[0])}, 2 : {vec(cands[1])}, "
                    f"3 : {vec(cands[2])} est le plus proche (distance euclidienne) ? Donne son numéro.",
                    Number(d.index(min(d)) + 1), "Calcule les 3 distances ‖cᵢ − q‖ (ou leurs carrés, c'est plus rapide) et compare.",
                    vectors([q] + cands, labels=["q", "1", "2", "3"]))
    ex.solution = _sol
    return ex


def l5_mean(r):
    vs = [v3(r) for _ in range(3)]
    _sol = lines(*[f"Composante {i + 1} : ({' + '.join(par(v[i]) for v in vs)}) / 3 = {sum(v[i] for v in vs)}/3 = {fmt(F(sum(v[i] for v in vs), 3))}" for i in range(3)])
    ex = Exercise(f"Moyenne (centroïde) des vecteurs {', '.join(vec(v) for v in vs)} ?" + G3,
                    Numbers([F(sum(v[i] for v in vs), 3) for i in range(3)], ordered=True),
                    "Moyenne composante par composante.", None)
    ex.solution = _sol
    return ex


def l5_neuron(r):
    w, x = v3(r, -3, 3), v3(r, -3, 3)
    b = r.randint(-5, 5)
    _sol = lines(f"w·x = {dstr(w, x)} = {dot(w, x)}", f"z = {dot(w, x)} {'+' if b >= 0 else '−'} {abs(b)} = {dot(w, x) + b}")
    ex = Exercise(f"Un neurone calcule z = w·x + b. Avec w = {vec(w)}, x = {vec(x)}, b = {b}, que vaut z ?",
                    Number(dot(w, x) + b), "Produit scalaire des poids et des entrées, puis ajoute le biais.", None)
    ex.solution = _sol
    return ex


def l5_l2reg(r):
    w = [r.randint(-3, 3) for _ in range(4)]
    lam = r.choice([0.1, 0.01, 0.5])
    _sol = lines(f"‖w‖² = {sq(w)} = {dot(w, w)}", f"λ·‖w‖² = {lam} × {dot(w, w)} = {fmt(lam * dot(w, w))}")
    ex = Exercise(f"Pénalité L2 : λ·‖w‖² avec λ = {lam} et w = {vec(w)} ?", Number(lam * dot(w, w)),
                    "‖w‖² = somme des carrés (sans racine). Multiplie par λ.", bars(range(1, 5), w))
    ex.solution = _sol
    return ex


def l5_hadamard(r):
    u, v = [r.randint(-4, 4) for _ in range(3)], [r.randint(-4, 4) for _ in range(3)]
    _sol = lines(f"u ⊙ v = ({', '.join(f'{par(a)} × {par(b)}' for a, b in zip(u, v))})", f"= {vec([a * b for a, b in zip(u, v)])}")
    ex = Exercise(f"Produit de Hadamard (élément par élément) u ⊙ v avec u = {vec(u)}, v = {vec(v)} ?" + G3,
                    Numbers([a * b for a, b in zip(u, v)], ordered=True),
                    "u ⊙ v = (u₁v₁, u₂v₂, u₃v₃). Contrairement au produit scalaire, le résultat est un vecteur.", None)
    ex.solution = _sol
    return ex


def l5_sse(r):
    y = [r.randint(0, 5) for _ in range(4)]
    p = [r.randint(0, 5) for _ in range(4)]
    _e = [a - b for a, b in zip(p, y)]
    _sol = lines(f"Erreurs ŷ − y : {vec(_e)}", f"Carrés : {', '.join(str(e * e) for e in _e)}", f"Somme : {sum(e * e for e in _e)}")
    ex = Exercise(f"Prédictions ŷ = {vec(p)}, cibles y = {vec(y)}. Somme des erreurs au carré Σ(ŷᵢ − yᵢ)² ?",
                    Number(sum((a - b) ** 2 for a, b in zip(p, y))),
                    "C'est ‖ŷ − y‖² : la distance euclidienne au carré entre prédictions et cibles.",
                    bars(["e1", "e2", "e3", "e4"], [a - b for a, b in zip(p, y)], "erreurs ŷ − y"))
    ex.solution = _sol
    return ex


def l5_proj_len(r):
    u = r.choice([(3, 4), (6, 8), (5, 12), (8, 6)])
    v = v2(r)
    _n = int(norm(u))
    _sol = lines(f"u·v = {dstr(u, v)} = {dot(u, v)}", f"‖u‖ = √({sq(u)}) = {_n}", f"Projection = {dot(u, v)} / {_n} = {fmt(F(dot(u, v), _n))}")
    ex = Exercise(f"Longueur de la projection (scalaire) de v = {vec(v)} sur la direction de u = {vec(u)} ?",
                    Number(F(dot(u, v), int(norm(u)))),
                    "Projection scalaire = u·v / ‖u‖ (c'est v·û avec û unitaire).", vectors([u, v], labels=["u", "v"]))
    ex.solution = _sol
    return ex


def l5_normalize_batch(r):
    u = [r.randint(1, 9) for _ in range(4)]
    s = sum(u)
    _sol = lines(f"Somme = {' + '.join(map(str, u))} = {s}", f"1re composante : {u[0]} / {s} = {fmt(F(u[0], s))}")
    ex = Exercise(f"Normalise u = {vec(u)} pour que ses composantes somment à 1 (division par la somme). "
                    "Valeur de la première composante ?", Number(F(u[0], s)),
                    "Divise chaque composante par la somme totale (c'est ce que fait softmax, après avoir pris les exponentielles).",
                    bars(range(1, 5), u))
    ex.solution = _sol
    return ex


LEVELS = {
    1: [l1_add, l1_sub, l1_scale, l1_AB, l1_mid, l1_norm, l1_opposite, l1_read, l1_combo, l1_equal],
    2: [l2_dot, l2_norm_sqrt, l2_unit, l2_orth, l2_orth_k, l2_angle, l2_dist, l2_scalar_proj, l2_colinear, l2_det_area],
    3: [l3_add, l3_dot, l3_norm, l3_cross, l3_orth, l3_proj, l3_dist, l3_angle, l3_unit, l3_triangle_area],
    4: [l4_express, l4_independent, l4_basis, l4_span_dim, l4_coords, l4_gram, l4_orthonormal, l4_l1, l4_linf, l4_l2_4d],
    5: [l5_cosine, l5_euclid, l5_nearest, l5_mean, l5_neuron, l5_l2reg, l5_hadamard, l5_sse, l5_proj_len,
        l5_normalize_batch],
}
