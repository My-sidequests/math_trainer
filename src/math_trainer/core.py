"""Briques communes : l'exercice, le formatage des nombres et les aides au dessin."""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

import matplotlib.patches
import numpy as np
from matplotlib.axes import Axes

from .answers import Answer

Draw = Callable[[Axes], None]
Generator = Callable[[random.Random], "Exercise"]

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#4a5568", "#6b46c1"


@dataclass
class Exercise:
    statement: str          # énoncé (texte, affiché dans le terminal et sur la figure)
    answer: Answer          # réponse attendue
    method: str             # comment résoudre, SANS la solution
    draw: Draw | None = None  # représentation géométrique
    solution: str = ""      # démonstration complète sur les nombres de l'exercice (la ligne « Réponse » est ajoutée automatiquement)


# ---------------------------------------------------------------- formatage

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sup(n) -> str:
    return str(n).translate(SUP)


def num(v) -> str:
    """Affiche un nombre proprement : 3, -2.5, 7/3."""
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def poly(coeffs: dict[int, float], var: str = "x") -> str:
    """poly({2: 3, 1: -1, 0: 4}) -> '3x² − x + 4'."""
    parts = []
    for p in sorted(coeffs, reverse=True):
        c = coeffs[p]
        if c == 0:
            continue
        sign = "−" if c < 0 else "+"
        a = abs(c)
        if p == 0:
            body = num(a)
        else:
            coef = "" if a == 1 else num(a)
            body = coef + var + (sup(p) if p > 1 else "")
        parts.append((sign, body))
    if not parts:
        return "0"
    first_sign, first = parts[0]
    s = ("−" if first_sign == "−" else "") + first
    for sign, body in parts[1:]:
        s += f" {sign} {body}"
    return s


def coef(c) -> str:
    """Coefficient devant une parenthèse : '' pour 1, '−' pour −1, sinon le nombre."""
    return "" if c == 1 else ("−" if c == -1 else num(c).replace("-", "−"))


def term(c, var: str) -> str:
    """Premier terme d'une somme : 'x', '−x', '3y', '−2z'."""
    return poly({1: c}, var)


def sterm(c, var: str) -> str:
    """Terme suivant d'une somme : '+ y', '− 3z' (vide si c = 0)."""
    if c == 0:
        return ""
    a = abs(c)
    return f"{'−' if c < 0 else '+'} {'' if a == 1 else num(a)}{var}"


def lin(a, b, var: str = "x") -> str:
    return poly({1: a, 0: b}, var)


def signed(v) -> str:
    """'+ 3' ou '− 3' (pour écrire « x + 3 » ou « x − 3 »)."""
    return f"− {num(abs(v))}" if v < 0 else f"+ {num(v)}"


def par(v) -> str:
    """Met les négatifs entre parenthèses : (−3)."""
    return f"({num(v).replace('-', '−')})" if v < 0 else num(v)


def nz(rng: random.Random, lo: int, hi: int) -> int:
    """Entier non nul dans [lo, hi]."""
    while True:
        v = rng.randint(lo, hi)
        if v != 0:
            return v


# ---------------------------------------------------------------- dessin

def setup(ax: Axes, xlim=None, ylim=None, grid: bool = True, axes0: bool = True, equal: bool = False):
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, alpha=0.3)
    if axes0:
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8)
    if equal:
        ax.set_aspect("equal")


def curves(*fs, xlim=(-6, 6), ylim=None, labels=None, colors=None, points=None, equal=False) -> Draw:
    """Trace une ou plusieurs fonctions f(x) (vectorisées numpy)."""
    colors = colors or [BLUE, ORANGE, GREEN, PURPLE, RED]

    def draw(ax: Axes):
        x = np.linspace(*xlim, 800)
        for i, f in enumerate(fs):
            with np.errstate(all="ignore"):
                y = np.asarray(f(x), dtype=float) * np.ones_like(x)
            if ylim:
                y = np.where((y < ylim[0] - 50) | (y > ylim[1] + 50), np.nan, y)
            lab = labels[i] if labels and i < len(labels) else None
            ax.plot(x, y, color=colors[i % len(colors)], lw=2.2, label=lab)
        for p in points or []:
            ax.plot(p[0], p[1], "o", color=RED, ms=7, zorder=5)
            if len(p) > 2:
                ax.annotate(p[2], (p[0], p[1]), textcoords="offset points", xytext=(6, 6), color=RED)
        setup(ax, xlim, ylim, equal=equal)
        if labels:
            ax.legend(loc="best")

    return draw


def number_line(lo: float, hi: float, marks=(), label="x") -> Draw:
    def draw(ax: Axes):
        ax.axhline(0, color="black", lw=1.5)
        for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
            ax.plot([k, k], [-0.08, 0.08], color="black", lw=1)
            ax.text(k, -0.25, str(k), ha="center", fontsize=9)
        for m in marks:
            ax.plot(m[0], 0, "o", color=m[2] if len(m) > 2 else RED, ms=9)
            ax.text(m[0], 0.2, m[1], ha="center", color=m[2] if len(m) > 2 else RED)
        ax.text(hi + 0.3, 0, label, va="center")
        ax.set_xlim(lo - 0.5, hi + 0.8)
        ax.set_ylim(-1, 1)
        ax.axis("off")

    return draw


def text_only(lines: list[str]) -> Draw:
    def draw(ax: Axes):
        ax.axis("off")
        for i, line in enumerate(lines):
            ax.text(0.5, 0.8 - i * 0.15, line, ha="center", va="center", fontsize=15, transform=ax.transAxes)

    return draw


def triangle(pts, labels_sides=(), labels_pts=(), right_angle_at=None, angle_marks=()) -> Draw:
    """pts = [(x,y)]*3 ; labels_sides = [(i, j, texte)] ; angle_marks = [(sommet, texte)]."""

    def draw(ax: Axes):
        P = np.array(pts + [pts[0]], dtype=float)
        ax.fill(P[:, 0], P[:, 1], color="#ebf8ff")
        ax.plot(P[:, 0], P[:, 1], color=BLUE, lw=2.5)
        c = P[:3].mean(axis=0)
        for i, j, t in labels_sides:
            m = (P[i] + P[j]) / 2
            d = m - c
            d = d / (np.linalg.norm(d) + 1e-9)
            span = np.ptp(P[:, 0]) + np.ptp(P[:, 1])
            ax.text(*(m + d * span * 0.06), t, ha="center", va="center", fontsize=13,
                    color=RED if "?" in t else "black", fontweight="bold")
        for i, t in labels_pts:
            d = P[i] - c
            d = d / (np.linalg.norm(d) + 1e-9)
            ax.text(*(P[i] + d * 0.35), t, ha="center", va="center", fontsize=12, color=GREY)
        if right_angle_at is not None:
            k = right_angle_at
            a, b = P[(k + 1) % 3] - P[k], P[(k + 2) % 3] - P[k]
            s = 0.08 * min(np.linalg.norm(a), np.linalg.norm(b))
            a, b = a / np.linalg.norm(a) * s, b / np.linalg.norm(b) * s
            q = np.array([P[k] + a, P[k] + a + b, P[k] + b])
            ax.plot(q[:, 0], q[:, 1], color="black", lw=1)
        for k, t in angle_marks:
            d = c - P[k]
            d = d / (np.linalg.norm(d) + 1e-9)
            span = np.ptp(P[:, 0]) + np.ptp(P[:, 1])
            ax.text(*(P[k] + d * span * 0.09), t, ha="center", va="center", fontsize=12,
                    color=RED if "?" in t else ORANGE)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.15)

    return draw


def unit_circle(angles=(), points=(), label_angles=None, radius=1.0) -> Draw:
    """Cercle trigonométrique avec rayons aux angles donnés (radians)."""

    def draw(ax: Axes):
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(radius * np.cos(t), radius * np.sin(t), color=BLUE, lw=2)
        for k, deg in enumerate(range(0, 360, 30)):
            r = np.radians(deg)
            ax.plot([0.95 * radius * np.cos(r), 1.05 * radius * np.cos(r)],
                    [0.95 * radius * np.sin(r), 1.05 * radius * np.sin(r)], color=GREY, lw=0.8)
        for i, a in enumerate(angles):
            ax.plot([0, radius * np.cos(a)], [0, radius * np.sin(a)], color=ORANGE, lw=2.2)
            ax.plot(radius * np.cos(a), radius * np.sin(a), "o", color=RED, ms=7)
            arc = np.linspace(0, a, 100)
            ax.plot(0.25 * radius * np.cos(arc), 0.25 * radius * np.sin(arc), color=ORANGE, lw=1.2)
            if label_angles:
                ax.text(0.38 * radius * np.cos(a / 2), 0.38 * radius * np.sin(a / 2), label_angles[i],
                        color=ORANGE, ha="center", va="center", fontsize=12)
        for p in points:
            ax.plot(p[0], p[1], "o", color=RED, ms=7)
        setup(ax, (-1.4 * radius, 1.4 * radius), (-1.4 * radius, 1.4 * radius), equal=True)

    return draw


def vectors(vecs, origin=(0, 0), labels=None, lim=None) -> Draw:
    def draw(ax: Axes):
        cols = [BLUE, ORANGE, GREEN, PURPLE]
        allp = [origin] + [(origin[0] + v[0], origin[1] + v[1]) for v in vecs]
        for i, v in enumerate(vecs):
            ax.annotate("", xy=(origin[0] + v[0], origin[1] + v[1]), xytext=origin,
                        arrowprops=dict(arrowstyle="-|>", color=cols[i % 4], lw=2.5))
            if labels:
                ax.text(origin[0] + v[0] * 1.08, origin[1] + v[1] * 1.08, labels[i],
                        color=cols[i % 4], fontsize=13, fontweight="bold")
        m = lim or max(3, max(max(abs(p[0]), abs(p[1])) for p in allp) + 1)
        setup(ax, (-m, m), (-m, m), equal=True)

    return draw


def rectangle(w_label: str, h_label: str, w=3.0, h=2.0) -> Draw:
    def draw(ax: Axes):
        ax.fill([0, w, w, 0, 0], [0, 0, h, h, 0], color="#ebf8ff")
        ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color=BLUE, lw=2.5)
        ax.text(w / 2, -0.2, w_label, ha="center", va="top", fontsize=13)
        ax.text(-0.15, h / 2, h_label, ha="right", va="center", fontsize=13)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.3)

    return draw


# ---------------------------------------------------------------- vecteurs, matrices, données

def fmt(v) -> str:
    """Nombre lisible (entiers sans .0, fractions, flottants courts), signe moins typographique."""
    if isinstance(v, float) and not v.is_integer():
        return f"{round(v, 4):g}".replace("-", "−")
    if isinstance(v, float):
        v = int(v)
    return num(v).replace("-", "−")


def vec(v) -> str:
    return "(" + ", ".join(fmt(c) for c in v) + ")"


def mat(M) -> str:
    return "[" + ", ".join("[" + ", ".join(fmt(c) for c in row) + "]" for row in M) + "]"


def matrices(items, ops=None) -> Draw:
    """Affiche des matrices / vecteurs côte à côte : items = [(nom, matrice ou vecteur colonne)]."""

    def draw(ax: Axes):
        ax.axis("off")
        x = 0.02
        n = len(items)
        width = 0.96 / max(n, 1)
        for k, (name, M) in enumerate(items):
            M = [list(r) if isinstance(r, (list, tuple)) else [r] for r in M]
            rows, cols = len(M), max(len(r) for r in M)
            cx = x + k * width + width / 2
            ax.text(cx, 0.88, name, ha="center", fontsize=15, color=BLUE, fontweight="bold", transform=ax.transAxes)
            cw = min(0.08, width * 0.8 / cols)
            rh = min(0.12, 0.6 / rows)
            x0 = cx - cw * cols / 2
            y0 = 0.5 + rh * rows / 2
            for i, r in enumerate(M):
                for j, c in enumerate(r):
                    ax.text(x0 + (j + 0.5) * cw, y0 - (i + 0.5) * rh, fmt(c), ha="center", va="center",
                            fontsize=14, color=RED if c == "?" else "black", transform=ax.transAxes)
            for side in (x0 - 0.005, x0 + cw * cols + 0.005):
                d = 0.012 if side < cx else -0.012
                ax.plot([side + d, side, side, side + d], [y0, y0, y0 - rh * rows, y0 - rh * rows],
                        color="black", lw=1.5, transform=ax.transAxes)
            if ops and k < len(ops) and ops[k]:
                ax.text(x + (k + 1) * width, 0.5, ops[k], ha="center", va="center", fontsize=18,
                        transform=ax.transAxes)

    return draw


def bars(labels, values, title="", color=BLUE, ylim=None) -> Draw:
    def draw(ax: Axes):
        ax.bar([str(l) for l in labels], values, color=color, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8)
        if title:
            ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.3)

    return draw


def scatter(xs, ys, line=None, xlabel="x", ylabel="y", highlight=None) -> Draw:
    def draw(ax: Axes):
        ax.scatter(xs, ys, color=BLUE, s=50, zorder=5)
        if highlight is not None:
            ax.scatter([xs[highlight]], [ys[highlight]], color=RED, s=80, zorder=6)
        if line:
            t = np.linspace(min(xs) - 1, max(xs) + 1, 50)
            ax.plot(t, line(t), color=ORANGE, lw=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    return draw


def contour(f, xlim=(-4, 4), ylim=(-4, 4), points=(), arrows=(), levels=15) -> Draw:
    """Lignes de niveau d'une fonction f(x, y) ; arrows = [((x, y), (dx, dy))]."""

    def draw(ax: Axes):
        X, Y = np.meshgrid(np.linspace(*xlim, 200), np.linspace(*ylim, 200))
        with np.errstate(all="ignore"):
            Z = f(X, Y)
        ax.contour(X, Y, Z, levels=levels, cmap="Blues")
        for p in points:
            ax.plot(p[0], p[1], "o", color=RED, ms=8, zorder=5)
            if len(p) > 2:
                ax.annotate(p[2], (p[0], p[1]), textcoords="offset points", xytext=(6, 6), color=RED)
        for (p, d) in arrows:
            ax.annotate("", xy=(p[0] + d[0], p[1] + d[1]), xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
        setup(ax, xlim, ylim, equal=True)

    return draw


def network(sizes, labels=None, title="") -> Draw:
    """Schéma d'un réseau de neurones : sizes = [entrées, cachées..., sorties]."""

    def draw(ax: Axes):
        ax.axis("off")
        L = len(sizes)
        pos = []
        for i, n in enumerate(sizes):
            xs = i / max(L - 1, 1)
            ys = [(j - (n - 1) / 2) * 0.22 for j in range(n)]
            pos.append([(xs, y) for y in ys])
        for i in range(L - 1):
            for a in pos[i]:
                for b in pos[i + 1]:
                    ax.plot([a[0], b[0]], [a[1], b[1]], color="#cbd5e0", lw=1, zorder=1)
        cols = [GREEN] + [BLUE] * (L - 2) + [ORANGE]
        for i, layer in enumerate(pos):
            for (x, y) in layer:
                ax.add_patch(matplotlib.patches.Circle((x, y), 0.045, color=cols[i], zorder=3))
            if labels and i < len(labels):
                ax.text(layer[0][0], min(p[1] for p in layer) - 0.12, labels[i], ha="center", fontsize=11)
        if title:
            ax.set_title(title)
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal")

    return draw


def transform_square(M) -> Draw:
    """Image du carré unité par la matrice 2×2 M (en bleu le carré, en orange son image)."""

    def draw(ax: Axes):
        sq = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
        im = np.array(M, dtype=float) @ sq
        ax.fill(sq[0], sq[1], color=BLUE, alpha=0.25)
        ax.plot(sq[0], sq[1], color=BLUE, lw=2, label="carré unité")
        ax.fill(im[0], im[1], color=ORANGE, alpha=0.25)
        ax.plot(im[0], im[1], color=ORANGE, lw=2, label="image par la matrice")
        m = max(2, np.abs(im).max() + 1)
        setup(ax, (-m, m), (-m, m), equal=True)
        ax.legend(loc="lower left")

    return draw


# ---------------------------------------------------------------- démonstrations

def steps_lin(a, b, c, var: str = "x") -> list[str]:
    """Étapes de résolution de a·var + b = c (a ≠ 0), en valeurs exactes."""
    a, b, c = Fraction(a), Fraction(b), Fraction(c)
    out = []
    rhs = c
    if b != 0:
        rhs = c - b
        out.append(f"On {'soustrait' if b > 0 else 'ajoute'} {fmt(abs(b))} des deux côtés : {term(a, var)} = {fmt(rhs)}")
    if a != 1:
        out.append(f"On divise les deux côtés par {par(a)} : {var} = {fmt(rhs)} / {par(a)} = {fmt(rhs / a)}")
    return out


def lines(*parts) -> str:
    """Assemble des lignes (ou listes de lignes) de démonstration en un texte numéroté."""
    flat = []
    for p in parts:
        flat.extend(p if isinstance(p, list) else [p])
    return "\n".join(f"{i}. {t}" for i, t in enumerate(flat, 1))
