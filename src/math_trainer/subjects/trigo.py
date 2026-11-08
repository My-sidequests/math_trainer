"""Trigonométrie : triangle rectangle, cercle trigonométrique, identités, équations, triangles quelconques."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Number, Numbers
from ..core import ORANGE, Exercise, curves, num, triangle, unit_circle, vectors

DESCRIPTIONS = {
    1: "Triangle rectangle : Pythagore, cos / sin / tan",
    2: "Radians, cercle trigonométrique, valeurs remarquables",
    3: "Identités et propriétés (cos² + sin² = 1, symétries, période)",
    4: "Équations trigonométriques, formules d'addition",
    5: "Triangles quelconques, coordonnées polaires, angles entre vecteurs",
}

CALC = "\n(calculatrice autorisée ; 3 décimales)"
SOHCAHTOA = ("Dans un triangle rectangle, pour un angle θ (pas l'angle droit) :\n"
             "  cos θ = adjacent / hypoténuse   (CAH)\n"
             "  sin θ = opposé / hypoténuse     (SOH)\n"
             "  tan θ = opposé / adjacent       (TOA)\n"
             "L'hypoténuse est le plus grand côté, face à l'angle droit.\n"
             "Ta calculatrice (ou Python : math.cos(math.radians(θ))) doit être en DEGRÉS ici.")
TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (6, 8, 10), (9, 12, 15), (20, 21, 29)]
REMARKABLE = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]


def rad_str(deg: int) -> str:
    """60 -> 'π/3', 150 -> '5π/6', 0 -> '0'."""
    f = F(deg, 180)
    if f == 0:
        return "0"
    n, d = f.numerator, f.denominator
    top = "π" if n == 1 else ("−π" if n == -1 else f"{n}π".replace("-", "−"))
    return top if d == 1 else f"{top}/{d}"


def right_tri(a_label, b_label, c_label, a=4.0, b=3.0, angle_label=None):
    """Triangle rectangle en C=(0,0) ; A=(a,0) ; B=(0,b). Côtés : CA (adjacent à A), CB (opposé à A), AB (hyp)."""
    marks = [(1, angle_label)] if angle_label else []
    return triangle([(0, 0), (a, 0), (0, b)],
                    labels_sides=[(0, 1, a_label), (0, 2, b_label), (1, 2, c_label)],
                    right_angle_at=0, angle_marks=marks)


# ================================================================ NIVEAU 1

def l1_pyth_hyp(r):
    a, b, c = r.choice(TRIPLES)
    return Exercise(f"Triangle rectangle de côtés {a} et {b} autour de l'angle droit. Longueur de l'hypoténuse ?",
                    Number(c), "Pythagore : hypoténuse² = côté1² + côté2². Calcule, puis prends la racine carrée.",
                    right_tri(str(b), str(a), "?", b, a))


def l1_pyth_leg(r):
    a, b, c = r.choice(TRIPLES)
    return Exercise(f"Triangle rectangle d'hypoténuse {c}, et dont un côté de l'angle droit mesure {a}. "
                    "Longueur du troisième côté ?", Number(b),
                    "Pythagore : côté² = hypoténuse² − autre côté². Puis racine carrée.",
                    right_tri("?", str(a), str(c), b, a))


def _ratio(r, fn):
    a, b, c = r.choice(TRIPLES)
    # angle en A : adjacent = b (horizontal), opposé = a (vertical)
    val = {"cos": F(b, c), "sin": F(a, c), "tan": F(a, b)}[fn]
    return Exercise(f"Dans ce triangle rectangle, calcule {fn}(θ) (fraction ou décimal).", Number(val), SOHCAHTOA,
                    right_tri(str(b), str(a), str(c), b, a, "θ"))


def l1_cos(r):
    return _ratio(r, "cos")


def l1_sin(r):
    return _ratio(r, "sin")


def l1_tan(r):
    return _ratio(r, "tan")


def l1_opp_from_hyp(r):
    th, h = r.randint(15, 75), r.randint(3, 20)
    return Exercise(f"Hypoténuse = {h}, angle θ = {th}°. Longueur du côté OPPOSÉ à θ ?" + CALC,
                    Number(h * math.sin(math.radians(th)), 5e-3), SOHCAHTOA + "\nIsole l'inconnue dans la bonne formule.",
                    right_tri("", "?", str(h), 4 * math.cos(math.radians(th)), 4 * math.sin(math.radians(th)), f"{th}°"))


def l1_adj_from_hyp(r):
    th, h = r.randint(15, 75), r.randint(3, 20)
    return Exercise(f"Hypoténuse = {h}, angle θ = {th}°. Longueur du côté ADJACENT à θ ?" + CALC,
                    Number(h * math.cos(math.radians(th)), 5e-3), SOHCAHTOA,
                    right_tri("?", "", str(h), 4 * math.cos(math.radians(th)), 4 * math.sin(math.radians(th)), f"{th}°"))


def l1_opp_from_adj(r):
    th, a = r.randint(15, 75), r.randint(3, 20)
    return Exercise(f"Côté adjacent = {a}, angle θ = {th}°. Longueur du côté OPPOSÉ ?" + CALC,
                    Number(a * math.tan(math.radians(th)), 5e-3), SOHCAHTOA,
                    right_tri(str(a), "?", "", 4, 4 * math.tan(math.radians(th)), f"{th}°"))


def l1_angle(r):
    a, b, c = r.choice(TRIPLES)
    return Exercise(f"Triangle rectangle : adjacent = {b}, hypoténuse = {c}. Que vaut l'angle θ en degrés ?" + CALC,
                    Number(math.degrees(math.acos(b / c)), 5e-3),
                    SOHCAHTOA + "\nPour retrouver un angle à partir d'un rapport, on utilise la fonction réciproque : "
                    "θ = arccos(rapport) (touche cos⁻¹, ou math.degrees(math.acos(...))).",
                    right_tri(str(b), "", str(c), b, a, "θ ?"))


def l1_angle_sum(r):
    a, b = r.randint(20, 100), r.randint(20, 59)
    return Exercise(f"Un triangle a deux angles de {a}° et {b}°. Combien mesure le troisième ?", Number(180 - a - b),
                    "La somme des angles d'un triangle vaut toujours 180°.",
                    triangle([(0, 0), (5, 0), (1.5, 3)], angle_marks=[(0, f"{a}°"), (1, f"{b}°"), (2, "?")]))


# ================================================================ NIVEAU 2

M_RAD = ("180° = π radians. Donc : radians = degrés × π / 180, et degrés = radians × 180 / π.\n"
         "Un radian, c'est l'angle qui découpe sur le cercle un arc de longueur égale au rayon.")
M_CIRCLE = ("Sur le cercle trigonométrique (rayon 1), le point à l'angle θ a pour coordonnées (cos θ, sin θ).\n"
            "Valeurs à connaître : cos 0 = 1, cos 30° = √3/2, cos 45° = √2/2, cos 60° = 1/2, cos 90° = 0 "
            "(et sin dans l'ordre inverse).\nPour les autres quadrants, utilise les symétries du cercle et regarde "
            "le signe de x (cos) et de y (sin).\nTu peux répondre avec sqrt : ex. sqrt(3)/2.")


def l2_deg_to_rad(r):
    d = r.choice(REMARKABLE[1:])
    return Exercise(f"Convertis {d}° en radians. (ex : pi/4)", Number(math.radians(d)), M_RAD,
                    unit_circle([math.radians(d)], label_angles=[f"{d}°"]))


def l2_rad_to_deg(r):
    d = r.choice(REMARKABLE[1:])
    return Exercise(f"Convertis {rad_str(d)} radians en degrés.", Number(d), M_RAD,
                    unit_circle([math.radians(d)], label_angles=[rad_str(d)]))


def _remarkable(r, fn, in_rad, pool):
    d = r.choice(pool)
    val = {"cos": math.cos, "sin": math.sin, "tan": math.tan}[fn](math.radians(d))
    shown = rad_str(d) if in_rad else f"{d}°"
    return Exercise(f"Valeur exacte de {fn}({shown}) ?", Number(val), M_CIRCLE,
                    unit_circle([math.radians(d)], label_angles=[shown]))


def l2_cos_rem(r):
    return _remarkable(r, "cos", False, [0, 30, 45, 60, 90])


def l2_sin_rem(r):
    return _remarkable(r, "sin", True, [0, 30, 45, 60, 90])


def l2_tan_rem(r):
    return _remarkable(r, "tan", False, [0, 30, 45, 60])


def l2_point(r):
    d = r.choice(REMARKABLE)
    a = math.radians(d)
    return Exercise(f"Coordonnées (x, y) du point du cercle trigonométrique à l'angle {rad_str(d)} ?",
                    Numbers([math.cos(a), math.sin(a)], ordered=True), M_CIRCLE + "\nRéponse : x, y",
                    unit_circle([a], label_angles=[rad_str(d)]))


def l2_cos_other(r):
    return _remarkable(r, "cos", True, [120, 135, 150, 180, 210, 225, 240, 300, 315, 330])


def l2_sin_neg(r):
    d = r.choice([30, 45, 60, 90, 120, 150])
    return Exercise(f"Valeur exacte de sin({rad_str(-d)}) ?", Number(math.sin(math.radians(-d))),
                    M_CIRCLE + "\nUn angle négatif se mesure dans le sens des aiguilles d'une montre.",
                    unit_circle([math.radians(-d)], label_angles=[rad_str(-d)]))


def l2_arc(r):
    R, d = r.randint(2, 12), r.choice([30, 45, 60, 90, 120, 180])
    return Exercise(f"Un cercle de rayon {R} cm. Longueur de l'arc intercepté par un angle de {rad_str(d)} rad ?",
                    Number(R * math.radians(d)), "Longueur d'arc = rayon × angle (en RADIANS).",
                    unit_circle([math.radians(d)], label_angles=[rad_str(d)]))


def l2_quadrant(r):
    d = r.choice([v for v in range(5, 360, 5) if v % 90])
    q = d // 90 + 1
    return Exercise(f"Dans quel quadrant (1, 2, 3 ou 4) se trouve l'angle {d}° ?", Number(q),
                    "Quadrant 1 : 0° à 90° (x>0, y>0) ; 2 : 90° à 180° ; 3 : 180° à 270° ; 4 : 270° à 360°.",
                    unit_circle([]))


# ================================================================ NIVEAU 3

def l3_cos_from_sin(r):
    a, b, c = r.choice(TRIPLES)
    q = r.choice([1, 2])
    s = F(a, c)
    cos = F(b, c) * (1 if q == 1 else -1)
    return Exercise(f"sin θ = {num(s)} et θ est dans le quadrant {q} ({'0 à π/2' if q == 1 else 'π/2 à π'}). "
                    "Calcule cos θ.", Number(cos),
                    "cos²θ + sin²θ = 1, donc cos²θ = 1 − sin²θ. Prends la racine, puis choisis le signe "
                    "selon le quadrant (cos = abscisse).",
                    unit_circle([math.atan2(float(s), float(cos))]))


def l3_tan_from(r):
    a, b, c = r.choice(TRIPLES)
    return Exercise(f"cos θ = {b}/{c} et sin θ = {a}/{c}. Calcule tan θ.", Number(F(a, b)),
                    "tan θ = sin θ / cos θ. Diviser deux fractions de même dénominateur : les dénominateurs se simplifient.",
                    unit_circle([math.atan2(a, b)]))


def l3_pi_minus(r):
    c = r.choice([F(1, 3), F(2, 5), F(3, 4), F(-1, 2), F(1, 5)])
    return Exercise(f"cos x = {num(c)}. Que vaut cos(π − x) ?", Number(-c),
                    "π − x est le symétrique de x par rapport à l'axe vertical : même sinus, cosinus opposé. "
                    "Dessine-le sur le cercle.",
                    unit_circle([math.acos(float(c)), math.pi - math.acos(float(c))], label_angles=["x", "π − x"]))


def l3_complement(r):
    c = r.choice([F(1, 3), F(2, 5), F(3, 4), F(-1, 2), F(4, 5)])
    return Exercise(f"cos x = {num(c)}. Que vaut sin(π/2 − x) ?", Number(c),
                    "Dans un triangle rectangle, les deux angles aigus sont complémentaires (somme π/2) : "
                    "le côté adjacent de l'un est l'opposé de l'autre.",
                    unit_circle([math.acos(float(c)), math.pi / 2 - math.acos(float(c))], label_angles=["x", "π/2 − x"]))


def l3_neg(r):
    c = r.choice([F(1, 3), F(2, 7), F(3, 4), F(-1, 2)])
    fn = r.choice(["cos", "sin"])
    return Exercise(f"{fn} x = {num(c)}. Que vaut {fn}(−x) ?", Number(c if fn == "cos" else -c),
                    "−x est le symétrique de x par rapport à l'axe horizontal : même abscisse (cos), "
                    "ordonnée opposée (sin). cos est paire, sin est impaire.",
                    unit_circle([0.9, -0.9], label_angles=["x", "−x"]))


def l3_period_shift(r):
    s, k = r.choice([F(1, 4), F(2, 3), F(-3, 5)]), r.randint(1, 5)
    return Exercise(f"sin x = {num(s)}. Que vaut sin(x + {2 * k}π) ?", Number(s),
                    "Ajouter 2π, c'est faire un tour complet du cercle : on retombe au même point.",
                    unit_circle([math.asin(float(s))]))


def l3_identity(r):
    a = r.randint(-9, 9)
    return Exercise(f"Simplifie : sin²(x) + cos²(x) {'+' if a >= 0 else '−'} {abs(a)}", Number(1 + a),
                    "Pour tout x, cos²x + sin²x = 1 (Pythagore dans le cercle de rayon 1).",
                    unit_circle([0.8]))


def l3_sector(r):
    R, d = r.randint(2, 10), r.choice([30, 45, 60, 90, 120])
    return Exercise(f"Aire d'un secteur de disque de rayon {R} et d'angle {rad_str(d)} ?", Number(R * R * math.radians(d) / 2),
                    "Aire du secteur = (angle / 2π) × aire du disque = r² × θ / 2 (θ en radians).",
                    unit_circle([math.radians(d)]))


def l3_period(r):
    k = r.choice([2, 3, 4, 0.5])
    fn = r.choice(["sin", "cos"])
    return Exercise(f"Quelle est la période de f(x) = {fn}({num(k)}x) ?", Number(2 * math.pi / k),
                    "sin et cos ont une période 2π. Si on multiplie x par k, la courbe est « comprimée » de k : "
                    "période = 2π / k.",
                    curves(lambda t: (np.sin if fn == "sin" else np.cos)(k * t), lambda t: np.sin(t),
                           xlim=(-7, 7), labels=["f", "sin x (référence)"]))


def l3_amplitude(r):
    a, c = nz_float(r), r.randint(-4, 4)
    return Exercise(f"f(x) = {num(a)}·sin(x) {'+' if c >= 0 else '−'} {abs(c)}. Quelle est la valeur MAXIMALE de f ?",
                    Number(abs(a) + c),
                    "sin(x) varie entre −1 et 1. Donc a·sin(x) varie entre −|a| et |a|. Ajoute ensuite la constante.",
                    curves(lambda t: a * np.sin(t) + c, xlim=(-7, 7)))


def nz_float(r):
    return r.choice([-4, -3, -2, 2, 3, 4, 5])


# ================================================================ NIVEAU 4

M_EQ = ("1. Trouve un angle « de référence » dont le cos (ou sin) vaut la valeur (valeurs remarquables).\n"
        "2. Sur le cercle, une droite verticale (pour cos) ou horizontale (pour sin) coupe le cercle en DEUX points : "
        "trouve le deuxième angle par symétrie.\n"
        "   cos x = c → x et −x (ou 2π − x) ;  sin x = s → x et π − x.\n"
        "3. Garde les solutions dans l'intervalle demandé. Réponse : les angles séparés par des virgules (ex : pi/3, 5pi/3).")


def exact(v: float) -> str:
    """Écrit une valeur remarquable sous forme exacte : 0.866... -> '√3/2'."""
    table = {0: "0", 0.5: "1/2", 1: "1", round(math.sqrt(2) / 2, 6): "√2/2", round(math.sqrt(3) / 2, 6): "√3/2",
             round(math.sqrt(3), 6): "√3", round(math.sqrt(3) / 3, 6): "√3/3"}
    s = table[round(abs(v), 6)]
    return ("−" + s) if v < -1e-12 else s


def l4_cos_eq(r):
    d = r.choice([30, 45, 60, 120, 135, 150])
    c = math.cos(math.radians(d))
    sols = sorted({math.radians(d), 2 * math.pi - math.radians(d)})
    return Exercise(f"Résous cos x = {exact(c)} pour x dans [0, 2π[.", Numbers(sols), M_EQ,
                    _hline_circle(c, vertical=True))


def l4_sin_eq(r):
    d = r.choice([30, 45, 60, 210, 225, 240])
    s = math.sin(math.radians(d))
    a = math.radians(d)
    sols = sorted({a % (2 * math.pi), (math.pi - a) % (2 * math.pi)})
    return Exercise(f"Résous sin x = {exact(s)} pour x dans [0, 2π[.", Numbers(sols), M_EQ,
                    _hline_circle(s, vertical=False))


def _hline_circle(v, vertical):
    base = unit_circle([])

    def draw(ax):
        base(ax)
        if vertical:
            ax.axvline(v, color=ORANGE, ls="--", lw=1.8)
        else:
            ax.axhline(v, color=ORANGE, ls="--", lw=1.8)

    return draw


def l4_tan_eq(r):
    d = r.choice([30, 45, 60, 120, 135, 150])
    t = math.tan(math.radians(d))
    return Exercise(f"Résous tan x = {exact(t)} pour x dans [0, π[.", Numbers([math.radians(d)]),
                    "tan a une période π : dans [0, π[ il n'y a qu'une solution. Utilise les valeurs remarquables "
                    "(tan 30° = √3/3, tan 45° = 1, tan 60° = √3) et le signe (tan < 0 dans ]π/2, π[).",
                    curves(lambda x: np.tan(x), lambda x: t + 0 * x, xlim=(0, math.pi), ylim=(-4, 4)))


def l4_read_period(r):
    k = r.choice([1, 2, 3, 4])
    return Exercise("Lis la période de la courbe tracée (en multiple de π, ex : pi/2).", Number(2 * math.pi / k),
                    "La période est la longueur du plus petit motif qui se répète : mesure la distance entre deux "
                    "sommets consécutifs.",
                    _pi_axis(curves(lambda t: np.sin(k * t), xlim=(0, 2 * math.pi))))


def _pi_axis(draw):
    def d(ax):
        draw(ax)
        ticks = [i * math.pi / 4 for i in range(9)]
        ax.set_xticks(ticks, ["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π"])

    return d


def l4_max(r):
    a, b, d = r.choice([-3, -2, 2, 3, 4]), r.choice([1, 2, 3]), r.randint(-3, 3)
    return Exercise(f"f(x) = {a}·cos({b}x + 1) {'+' if d >= 0 else '−'} {abs(d)}. Valeur maximale de f ?",
                    Number(abs(a) + d),
                    "Peu importe ce qu'il y a dans le cos : il varie entre −1 et 1. Multiplie par a (attention au signe) "
                    "puis ajoute la constante.", None)


def l4_f0(r):
    a, d = r.randint(1, 5), r.choice([30, 45, 60, 90, 150])
    return Exercise(f"f(x) = {a}·sin(2x + {rad_str(d)}). Calcule f(0).", Number(a * math.sin(math.radians(d))),
                    "Remplace x par 0 : il reste " + str(a) + "·sin(" + rad_str(d) + "). Valeur remarquable.",
                    curves(lambda t: a * np.sin(2 * t + math.radians(d)), xlim=(-4, 4)))


def l4_sin2x_zero(r):
    k = r.choice([2, 3, 4])
    return Exercise(f"Résous sin({k}x) = 0 pour x dans [0, π].", Numbers([i * math.pi / k for i in range(k + 1)]),
                    "sin(X) = 0 ⇔ X = nπ (n entier). Ici X = " + str(k) + "x, donc x = nπ/" + str(k) +
                    ". Garde toutes les valeurs dans [0, π].",
                    _pi_axis(curves(lambda t: np.sin(k * t), xlim=(0, math.pi))))


def l4_addition(r):
    a, b = r.choice([(45, 30), (60, 45), (45, -30), (60, -45)])
    fn = r.choice(["cos", "sin"])
    val = {"cos": math.cos, "sin": math.sin}[fn](math.radians(a + b))
    return Exercise(f"Calcule la valeur exacte de {fn}({a + b}°) en écrivant {a + b}° = {a}° {'+' if b > 0 else '−'} {abs(b)}°.",
                    Number(val),
                    "cos(a + b) = cos a·cos b − sin a·sin b\nsin(a + b) = sin a·cos b + cos a·sin b\n"
                    "(pour a − b, change les signes du milieu). Réponse exacte avec sqrt, ex : (sqrt(6)+sqrt(2))/4.",
                    unit_circle([math.radians(a), math.radians(a + b)]))


def l4_double_sin(r):
    a, b, c = r.choice(TRIPLES)
    return Exercise(f"sin x = {a}/{c} et cos x = {b}/{c}. Calcule sin(2x).", Number(F(2 * a * b, c * c)),
                    "Formule de duplication : sin(2x) = 2·sin x·cos x.", unit_circle([math.atan2(a, b)]))


def l4_double_cos(r):
    c = r.choice([F(1, 3), F(2, 3), F(1, 4), F(3, 5), F(-1, 3)])
    return Exercise(f"cos x = {num(c)}. Calcule cos(2x).", Number(2 * c * c - 1),
                    "cos(2x) = 2cos²x − 1 (= 1 − 2sin²x = cos²x − sin²x).", unit_circle([math.acos(float(c))]))


# ================================================================ NIVEAU 5

def l5_alkashi(r):
    a, b, C = r.randint(3, 12), r.randint(3, 12), r.choice([40, 60, 75, 100, 120])
    c = math.sqrt(a * a + b * b - 2 * a * b * math.cos(math.radians(C)))
    return Exercise(f"Triangle : deux côtés de {a} et {b}, l'angle entre eux mesure {C}°. Longueur du 3e côté ?" + CALC,
                    Number(c, 5e-3),
                    "Loi des cosinus (Al-Kashi) : c² = a² + b² − 2ab·cos(C). C'est Pythagore généralisé "
                    "(si C = 90°, cos C = 0).",
                    triangle([(0, 0), (a, 0), (b * math.cos(math.radians(C)), b * math.sin(math.radians(C)))],
                             labels_sides=[(0, 1, str(a)), (0, 2, str(b)), (1, 2, "?")], angle_marks=[(0, f"{C}°")]))


def l5_angle_3sides(r):
    while True:
        a, b, c = sorted(r.sample(range(3, 13), 3))
        if a + b > c:
            break
    C = math.degrees(math.acos((a * a + b * b - c * c) / (2 * a * b)))
    return Exercise(f"Triangle de côtés {a}, {b} et {c}. Mesure (en degrés) de l'angle opposé au côté {c} ?" + CALC,
                    Number(C, 5e-3), "Al-Kashi, isolé pour cos(C) : cos C = (a² + b² − c²) / (2ab), puis C = arccos(...).",
                    triangle([(0, 0), (a, 0), (b * math.cos(math.radians(C)), b * math.sin(math.radians(C)))],
                             labels_sides=[(0, 1, str(a)), (0, 2, str(b)), (1, 2, str(c))], angle_marks=[(0, "?")]))


def l5_law_sines(r):
    A, B, a = r.choice([30, 40, 50, 70]), r.choice([45, 60, 80]), r.randint(4, 12)
    b = a * math.sin(math.radians(B)) / math.sin(math.radians(A))
    return Exercise(f"Dans un triangle, l'angle A = {A}°, l'angle B = {B}°, et le côté a (opposé à A) = {a}. "
                    "Longueur du côté b (opposé à B) ?" + CALC, Number(b, 5e-3),
                    "Loi des sinus : a / sin A = b / sin B = c / sin C. Isole b.", None)


def l5_area(r):
    a, b, C = r.randint(3, 12), r.randint(3, 12), r.choice([30, 45, 60, 90, 120, 150])
    return Exercise(f"Aire d'un triangle dont deux côtés mesurent {a} et {b} avec un angle de {C}° entre eux ?",
                    Number(0.5 * a * b * math.sin(math.radians(C)), 5e-3),
                    "Aire = ½·a·b·sin(C). (La hauteur vaut b·sin C, d'où base × hauteur / 2.)",
                    triangle([(0, 0), (a, 0), (b * math.cos(math.radians(C)), b * math.sin(math.radians(C)))],
                             labels_sides=[(0, 1, str(a)), (0, 2, str(b))], angle_marks=[(0, f"{C}°")]))


def l5_polar_to_cart(r):
    R, d = r.randint(1, 6), r.choice(REMARKABLE[1:])
    a = math.radians(d)
    return Exercise(f"Un point a pour coordonnées polaires r = {R}, θ = {rad_str(d)}. Coordonnées cartésiennes (x, y) ?",
                    Numbers([R * math.cos(a), R * math.sin(a)], ordered=True),
                    "x = r·cos θ, y = r·sin θ. Réponse : x, y (exact avec sqrt ou décimal précis).",
                    vectors([(R * math.cos(a), R * math.sin(a))], labels=["?"]))


def l5_cart_to_r(r):
    a, b, c = r.choice(TRIPLES)
    x, y = a * r.choice([1, -1]), b * r.choice([1, -1])
    return Exercise(f"Distance à l'origine (rayon polaire r) du point ({x} ; {y}) ?", Number(c),
                    "r = √(x² + y²) : c'est Pythagore.", vectors([(x, y)]))


def l5_cart_to_theta(r):
    x, y = nz_pt(r)
    return Exercise(f"Angle polaire θ (en degrés, entre −180 et 180) du point ({x} ; {y}) ?" + CALC,
                    Number(math.degrees(math.atan2(y, x)), 5e-3),
                    "tan θ = y / x, mais arctan seul ne donne qu'un angle entre −90° et 90° : regarde le quadrant "
                    "du point pour corriger (±180°). En Python : math.degrees(math.atan2(y, x)).",
                    vectors([(x, y)]))


def nz_pt(r):
    while True:
        x, y = r.randint(-6, 6), r.randint(-6, 6)
        if x and y:
            return x, y


def l5_rotation(r):
    x, y = nz_pt(r)
    d = r.choice([90, 180, 270, -90])
    a = math.radians(d)
    nx, ny = round(x * math.cos(a) - y * math.sin(a)), round(x * math.sin(a) + y * math.cos(a))
    return Exercise(f"On fait tourner le point ({x} ; {y}) de {d}° autour de l'origine (sens inverse des aiguilles "
                    "d'une montre si positif). Nouvelles coordonnées ?", Numbers([nx, ny], ordered=True),
                    "Rotation d'angle θ : x' = x·cos θ − y·sin θ ; y' = x·sin θ + y·cos θ.\n"
                    "(C'est une multiplication par une matrice : tu la reverras en algèbre linéaire.) Réponse : x, y",
                    vectors([(x, y)], labels=["départ"]))


def l5_dot_angle(r):
    nu, nv, d = r.randint(1, 6), r.randint(1, 6), r.choice([0, 30, 45, 60, 90, 120, 180])
    return Exercise(f"‖u‖ = {nu}, ‖v‖ = {nv}, et l'angle entre u et v est {d}°. Calcule le produit scalaire u·v.",
                    Number(nu * nv * math.cos(math.radians(d))),
                    "u·v = ‖u‖·‖v‖·cos(θ). Il est positif si l'angle est aigu, nul si perpendiculaires, "
                    "négatif si obtus.",
                    vectors([(nu, 0), (nv * math.cos(math.radians(d)), nv * math.sin(math.radians(d)))], labels=["u", "v"]))


def l5_angle_vectors(r):
    u, v = nz_pt(r), nz_pt(r)
    dot = u[0] * v[0] + u[1] * v[1]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot / (math.hypot(*u) * math.hypot(*v))))))
    return Exercise(f"u = ({u[0]} ; {u[1]}), v = ({v[0]} ; {v[1]}). Angle entre u et v en degrés ?" + CALC,
                    Number(ang, 5e-3),
                    "1. Produit scalaire avec les coordonnées : u·v = x_u·x_v + y_u·y_v.\n"
                    "2. Normes : ‖u‖ = √(x² + y²).\n3. cos θ = u·v / (‖u‖·‖v‖), puis θ = arccos(...).\n"
                    "(C'est la « similarité cosinus » utilisée partout en IA pour comparer des embeddings.)",
                    vectors([u, v], labels=["u", "v"]))


LEVELS = {
    1: [l1_pyth_hyp, l1_pyth_leg, l1_cos, l1_sin, l1_tan, l1_opp_from_hyp, l1_adj_from_hyp, l1_opp_from_adj,
        l1_angle, l1_angle_sum],
    2: [l2_deg_to_rad, l2_rad_to_deg, l2_cos_rem, l2_sin_rem, l2_tan_rem, l2_point, l2_cos_other, l2_sin_neg,
        l2_arc, l2_quadrant],
    3: [l3_cos_from_sin, l3_tan_from, l3_pi_minus, l3_complement, l3_neg, l3_period_shift, l3_identity, l3_sector,
        l3_period, l3_amplitude],
    4: [l4_cos_eq, l4_sin_eq, l4_tan_eq, l4_read_period, l4_max, l4_f0, l4_sin2x_zero, l4_addition, l4_double_sin,
        l4_double_cos],
    5: [l5_alkashi, l5_angle_3sides, l5_law_sines, l5_area, l5_polar_to_cart, l5_cart_to_r, l5_cart_to_theta,
        l5_rotation, l5_dot_angle, l5_angle_vectors],
}
