"""Point d'entrée : menus, affichage matplotlib et boucle d'exercices."""

from __future__ import annotations

import random
import re
import textwrap

import matplotlib.pyplot as plt

from .core import GREEN, Exercise
from .subjects import SUBJECTS

C_GREEN, C_RED, C_BOLD, C_DIM, C_END = "\033[92m", "\033[91m", "\033[1m", "\033[2m", "\033[0m"


# ---------------------------------------------------------------- affichage

def show(ex: Exercise, title: str):
    plt.close("all")
    fig = plt.figure(figsize=(9, 7.5))
    fig.canvas.manager.set_window_title(f"Math Trainer — {title}")
    wrapped = "\n".join(textwrap.fill(p, 70) for p in ex.statement.split("\n"))
    fig.text(0.5, 0.97, wrapped, ha="center", va="top", fontsize=12.5,
             bbox=dict(boxstyle="round,pad=0.6", fc="#f7fafc", ec="#cbd5e0"))
    n_lines = wrapped.count("\n") + 1
    top = max(0.45, 0.9 - 0.035 * n_lines)
    ax = fig.add_axes([0.1, 0.07, 0.82, top - 0.07])
    if ex.draw:
        ex.draw(ax)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Pas de figure pour cet exercice :\nelle donnerait directement la réponse.",
                ha="center", va="center", color="#718096", fontsize=12, transform=ax.transAxes)
    plt.show(block=False)
    plt.pause(0.1)
    return fig


def mark_success(fig):
    fig.text(0.5, 0.5, "SUCCESS", ha="center", va="center", fontsize=60, fontweight="bold",
             color=GREEN, alpha=0.85, rotation=12)
    fig.canvas.draw_idle()
    plt.pause(0.1)


def typo(text: str) -> str:
    """Remplace le tiret « - » devant un nombre par le vrai signe moins « − »."""
    return re.sub(r"(?<![\w)])-(?=[\d(])", "−", text)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit(0)


# ---------------------------------------------------------------- menus

def choose_subject():
    print(f"\n{C_BOLD}══════════ MATH TRAINER ══════════{C_END}")
    for i, s in enumerate(SUBJECTS, 1):
        tag = "" if s.levels else f"  {C_DIM}(bientôt){C_END}"
        print(f"  {i:2d}. {s.name}{tag}")
    print("   q. Quitter")
    while True:
        c = ask("\nMatière > ").lower()
        if c in ("q", "quit", "exit"):
            return None
        if c.isdigit() and 1 <= int(c) <= len(SUBJECTS):
            s = SUBJECTS[int(c) - 1]
            if s.levels:
                return s
            print("Cette matière n'est pas encore disponible.")
        else:
            print(f"Tape un numéro entre 1 et {len(SUBJECTS)}, ou q.")


def choose_level(subject) -> int | None:
    print(f"\n{C_BOLD}{subject.name}{C_END}")
    for lvl in range(1, 6):
        print(f"  {lvl}. {subject.descriptions.get(lvl, '')}")
    print("   r. Retour")
    while True:
        c = ask("Niveau (1-5) > ").lower()
        if c == "r":
            return None
        if c in {"1", "2", "3", "4", "5"}:
            return int(c)
        print("Tape un chiffre entre 1 et 5, ou r.")


def run_exercise(ex: Exercise, title: str) -> bool:
    """Retourne True si réussi, False si abandonné."""
    fig = show(ex, title)
    print(f"\n{C_BOLD}── {title} ──{C_END}")
    print(ex.statement)
    print(f"{C_DIM}Réponse attendue : {ex.answer.expected}")
    print(f"Commandes : m = méthode, q = abandonner{C_END}")
    attempts = 0
    while True:
        r = ask("\nRéponse > ")
        if not r:
            continue
        low = r.lower()
        if low == "m":
            print(f"{C_BOLD}Méthode :{C_END}\n{ex.method}")
            if ex.solution:
                print(f"\n{C_BOLD}Démonstration sur cet exercice :{C_END}\n{typo(ex.solution)}")
                print(f"{C_BOLD}Réponse : {ex.answer.display()}{C_END}")
            continue
        if low in ("q", "quit"):
            plt.close("all")
            return False
        try:
            ok = ex.answer.check(r)
        except Exception as e:  # saisie illisible : ne compte pas comme une tentative
            print(f"{C_RED}Je n'arrive pas à lire « {r} » ({e}). Format : {ex.answer.expected}{C_END}")
            continue
        attempts += 1
        if ok:
            print(f"\n{C_GREEN}{C_BOLD}  ✔ SUCCESS  {C_END} (en {attempts} tentative{'s' * (attempts > 1)})")
            mark_success(fig)
            return True
        print(f"{C_RED}✘ Pas encore. Réessaie (m pour la méthode).{C_END}")


def after_menu() -> str:
    while True:
        c = ask("\n[a] autre exercice   [m] menu principal   [q] quitter > ").lower()
        if c in ("a", "m", "q"):
            return c


def main() -> None:
    plt.ion()
    rng = random.Random()
    try:
        while True:
            subject = choose_subject()
            if subject is None:
                break
            level = choose_level(subject)
            if level is None:
                continue
            gens = subject.levels[level]
            last = None
            while True:
                gen = rng.choice([g for g in gens if g is not last] or gens)
                last = gen
                ex = gen(rng)
                run_exercise(ex, f"{subject.name} — niveau {level}")
                c = after_menu()
                plt.close("all")
                if c == "a":
                    continue
                if c == "q":
                    raise KeyboardInterrupt
                break
    except KeyboardInterrupt:
        pass
    plt.close("all")
    print("\nÀ bientôt !")


if __name__ == "__main__":
    main()
