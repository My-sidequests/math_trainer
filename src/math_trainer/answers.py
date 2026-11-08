"""Types de réponses attendues et vérification de la saisie utilisateur."""

from __future__ import annotations

import math
import random
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

X, Y, Z = sp.symbols("x y z")
VARS = {"x": X, "y": Y, "z": Z}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCALS = {
    "x": X,
    "y": Y,
    "z": Z,
    "e": sp.E,
    "pi": sp.pi,
    "ln": sp.log,
    "log": sp.log,
    "log10": lambda v: sp.log(v, 10),
    "log2": lambda v: sp.log(v, 2),
    "sqrt": sp.sqrt,
    "inf": sp.oo,
    "oo": sp.oo,
}
_NONE_WORDS = {"aucune", "aucun", "none", "rien", "ø", "∅", "vide"}


def parse(text: str) -> sp.Expr:
    """Transforme une saisie (« 2/3 », « √2/2 », « 3x^2 », « π/4 »...) en expression sympy."""
    t = text.strip().replace("²", "^2").replace("³", "^3").replace("√", "sqrt")
    t = t.replace("π", "pi").replace("−", "-").replace("×", "*").replace("÷", "/")
    t = t.replace("+inf", "inf").replace("+oo", "oo")
    t = re.sub(r"sqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", t)  # √2 -> sqrt(2)
    if not t:
        raise ValueError("réponse vide")
    return parse_expr(t, local_dict=_LOCALS, transformations=_TRANSFORMS)


def _to_float(expr: sp.Expr) -> float:
    if expr in (sp.oo, -sp.oo):
        return math.inf if expr == sp.oo else -math.inf
    if expr.free_symbols:
        raise ValueError("un nombre est attendu, pas une expression")
    v = complex(sp.N(expr))
    if abs(v.imag) > 1e-12:
        raise ValueError("nombre complexe non accepté")
    return v.real


def _close(a: float, b: float, tol: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    # tolérance relative, avec un plancher pour les valeurs proches de 0
    return abs(a - b) <= tol * max(0.1, abs(b))


def _split(text: str) -> list[str]:
    return [p for p in (s.strip() for s in text.replace(";", ",").split(",")) if p]


def show_num(v: float) -> str:
    """Affichage lisible d'une valeur numérique : fraction simple si possible, sinon décimal arrondi."""
    if math.isinf(v):
        return "+∞" if v > 0 else "−∞"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v))).replace("-", "−")
    if abs(v * 1000 - round(v * 1000)) < 1e-6:  # décimal court : 0.4, 5.5, 2.125
        return f"{v:.3f}".rstrip("0").rstrip(".").replace("-", "−")
    from fractions import Fraction
    fr = Fraction(v).limit_denominator(100)
    if abs(float(fr) - v) < 1e-9:
        return f"{fr.numerator}/{fr.denominator}".replace("-", "−")
    # multiples simples de √k ou de π : 0.866… → √3/2, 2.094… → 2π/3
    for unit, name in [(math.sqrt(2), "√2"), (math.sqrt(3), "√3"), (math.sqrt(6), "√6"), (math.pi, "π")]:
        c = Fraction(v / unit).limit_denominator(12)
        if c != 0 and abs(float(c) * unit - v) < 1e-9:
            n, d = c.numerator, c.denominator
            top = name if n == 1 else ("−" + name if n == -1 else f"{n}{name}".replace("-", "−"))
            return top if d == 1 else f"{top}/{d}"
    return f"{v:.4f}".rstrip("0").rstrip(".").replace("-", "−")


class Answer:
    expected: str = "une réponse"

    def check(self, text: str) -> bool:
        raise NotImplementedError

    def display(self) -> str:
        raise NotImplementedError


class Number(Answer):
    """Un seul nombre. tol est relative (avec un plancher absolu de tol × 0.1 près de 0)."""

    expected = "un nombre (ex : 3, -2.5, 7/3, sqrt(2)/2, pi/4)"

    def __init__(self, value: float, tol: float = 1e-3):
        self.value = float(value)
        self.tol = tol

    def check(self, text: str) -> bool:
        return _close(_to_float(parse(text)), self.value, self.tol)

    def display(self) -> str:
        return show_num(self.value)


class Numbers(Answer):
    """Plusieurs nombres séparés par des virgules. Liste vide = « aucune » solution."""

    def __init__(self, values, ordered: bool = False, tol: float = 1e-3):
        self.values = [float(v) for v in values]
        self.ordered = ordered
        self.tol = tol
        if ordered:
            self.expected = f"{len(self.values)} nombres dans l'ordre, séparés par des virgules (ex : 2, -1)"
        else:
            self.expected = "les nombres séparés par des virgules (ordre libre), ou « aucune »"

    def check(self, text: str) -> bool:
        if text.strip().lower() in _NONE_WORDS:
            return not self.values
        got = [_to_float(parse(p)) for p in _split(text)]
        if len(got) != len(self.values):
            return False
        if self.ordered:
            return all(_close(g, v, self.tol) for g, v in zip(got, self.values))
        remaining = list(self.values)
        for g in got:
            match = next((v for v in remaining if _close(g, v, self.tol)), None)
            if match is None:
                return False
            remaining.remove(match)
        return True

    def display(self) -> str:
        if not self.values:
            return "aucune solution"
        return ", ".join(show_num(v) for v in self.values)


class Expr(Answer):
    """Une expression en x (ou x, y, z), comparée numériquement en plusieurs points."""

    def __init__(self, expr: str | sp.Expr, domain: tuple[float, float] = (-3.0, 3.0)):
        self.expr = parse(expr) if isinstance(expr, str) else sp.sympify(expr)
        self.domain = domain
        names = sorted(str(s) for s in self.expr.free_symbols) or ["x"]
        self.vars = [VARS[n] for n in names]
        if len(self.vars) == 1:
            self.expected = f"une expression en {names[0]} (ex : 2*{names[0]} + 3, 3{names[0]}^2 - 1, e^(2{names[0]}))"
        else:
            self.expected = "une expression en " + ", ".join(names) + " (ex : 2*x*y + y^2)"

    def check(self, text: str) -> bool:
        got = parse(text)
        extra = got.free_symbols - set(self.vars)
        if extra:
            raise ValueError("variable inattendue : " + ", ".join(sorted(map(str, extra))))
        f_exp = sp.lambdify(self.vars, self.expr, "math")
        f_got = sp.lambdify(self.vars, got, "math")
        rng = random.Random(0)
        tested = 0
        for _ in range(60):
            pt = [rng.uniform(*self.domain) for _ in self.vars]
            try:
                a, b = f_exp(*pt), f_got(*pt)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                continue
            if isinstance(a, complex) or isinstance(b, complex):
                continue
            if not _close(float(b), float(a), 1e-6):
                return False
            tested += 1
            if tested >= 8:
                break
        return tested > 0

    def display(self) -> str:
        return str(self.expr).replace("**", "^").replace("log(", "ln(").replace("exp(", "e^(")


class Choice(Answer):
    """Un mot parmi une liste (insensible à la casse et aux accents simples)."""

    def __init__(self, correct: str, options: list[str]):
        self.correct = _norm(correct)
        self.options = options
        self.expected = "un mot parmi : " + " / ".join(options)

    def check(self, text: str) -> bool:
        return _norm(text) == self.correct

    def display(self) -> str:
        return next((o for o in self.options if _norm(o) == self.correct), self.correct)


def _norm(s: str) -> str:
    table = str.maketrans("éèêàâîïôûùç", "eeeaaiiouuc")
    return s.strip().lower().translate(table)
