"""Maths des réseaux de neurones : neurone, couches, pertes, rétropropagation, convolution et attention."""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from ..answers import Choice, Number, Numbers
from ..core import Exercise, bars, curves, matrices, mat, network, nz, vec

from ..core import fmt, lines, par  # noqa: E402,F811


def dstr(u, v) -> str:
    return " + ".join(f"{par(a)} × {par(b)}" for a, b in zip(u, v))

DESCRIPTIONS = {
    1: "Un neurone : somme pondérée et activations (ReLU, sigmoïde, tanh...)",
    2: "Couches et passe avant : vecteurs, paramètres, softmax",
    3: "Fonctions de perte et leurs dérivées",
    4: "Rétropropagation : gradients d'un neurone et d'un petit réseau",
    5: "Avancé : backprop complète, gradients qui disparaissent, convolution, attention",
}

CALC = "\n(calculatrice autorisée ; 3 décimales)"
M_NEURON = ("Un neurone calcule z = w·x + b = Σ wᵢxᵢ + b, puis a = activation(z).\n"
            "ReLU(z) = max(0, z) ; σ(z) = 1/(1 + e⁻ᶻ) ; tanh(z) ; LeakyReLU(z) = z si z > 0, sinon 0.01·z (ou la pente indiquée).")
M_BACK = ("Rétropropagation = règle de la chaîne appliquée de la sortie vers l'entrée.\n"
          "Pour z = w·x + b : ∂z/∂w = x, ∂z/∂b = 1, ∂z/∂x = w.  ReLU' = 1 si z > 0, sinon 0.  σ' = σ(1 − σ).\n"
          "Gradient d'un paramètre = (gradient qui arrive d'en haut) × (dérivée locale).")


def sig(z):
    return 1 / (1 + math.exp(-z))


def neuron_draw(n):
    return network([n, 1], labels=["entrées", "neurone"])


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


# ================================================================ NIVEAU 1

def l1_weighted(r):
    w, x = [r.randint(-3, 3) for _ in range(3)], [r.randint(-3, 3) for _ in range(3)]
    b = r.randint(-5, 5)
    _sol = lines(f"w·x = {dstr(w, x)} = {dot(w, x)}", f"z = {dot(w, x)} {'+' if b >= 0 else '−'} {abs(b)} = {dot(w, x) + b}")
    ex = Exercise(f"Neurone : w = {vec(w)}, x = {vec(x)}, b = {b}. Calcule z = w·x + b.", Number(dot(w, x) + b), M_NEURON,
                    neuron_draw(3))
    ex.solution = _sol
    return ex


def l1_relu(r):
    z = r.choice([-3, -1.5, 0, 0.5, 2, 4, -0.2])
    _sol = lines("ReLU(z) = max(0, z)", f"max(0, {z:g}) = {max(0, z):g}")
    ex = Exercise(f"ReLU({z:g}) ?", Number(max(0, z)), M_NEURON, curves(lambda t: np.maximum(0, t), xlim=(-5, 5)))
    ex.solution = _sol
    return ex


def l1_sigmoid(r):
    z = r.choice([0, 1, -1, 2, -2, 3])
    _sol = lines(f"e^(−{z}) ≈ {math.exp(-z):.4f}", f"σ({z}) = 1 / (1 + {math.exp(-z):.4f}) ≈ {sig(z):.4f}")
    ex = Exercise(f"σ({z}) ?" + CALC, Number(sig(z), 5e-3), M_NEURON, curves(lambda t: 1 / (1 + np.exp(-t)), xlim=(-6, 6)))
    ex.solution = _sol
    return ex


def l1_step(r):
    z = r.choice([-2, -0.5, 0.3, 1, 3])
    _sol = lines(f"z = {z} {'≥' if z >= 0 else '<'} 0", f"Sortie = {1 if z >= 0 else 0}")
    ex = Exercise(f"Perceptron historique : sortie = 1 si z ≥ 0, sinon 0. Sortie pour z = {z} ?", Number(1 if z >= 0 else 0),
                    "La fonction « marche » : tout ou rien. (On l'a remplacée par des fonctions dérivables pour pouvoir "
                    "faire de la descente de gradient.)", curves(lambda t: (t >= 0).astype(float), xlim=(-4, 4)))
    ex.solution = _sol
    return ex


def l1_neuron_relu(r):
    w, x = [r.randint(-2, 2) for _ in range(2)], [r.randint(-3, 3) for _ in range(2)]
    b = r.randint(-3, 3)
    _z = dot(w, x) + b
    _sol = lines(f"z = {dstr(w, x)} {'+' if b >= 0 else '−'} {abs(b)} = {_z}", f"a = max(0, {_z}) = {max(0, _z)}")
    ex = Exercise(f"Neurone ReLU : w = {vec(w)}, x = {vec(x)}, b = {b}. Sortie a = ReLU(w·x + b) ?", Number(max(0, dot(w, x) + b)),
                    M_NEURON, neuron_draw(2))
    ex.solution = _sol
    return ex


def l1_n_params(r):
    n = r.randint(2, 100)
    _sol = lines(f"{n} poids (un par entrée) + 1 biais", f"= {n + 1}")
    ex = Exercise(f"Combien de paramètres (poids + biais) pour un neurone à {n} entrées ?", Number(n + 1),
                    "Un poids par entrée, plus un biais.", None)
    ex.solution = _sol
    return ex


def l1_and(r):
    x1, x2 = r.choice([0, 1]), r.choice([0, 1])
    _sol = lines(f"z = 1 × {x1} + 1 × {x2} − 1.5 = {x1 + x2 - 1.5:g}", f"z {'≥' if x1 + x2 >= 1.5 else '<'} 0 → sortie {1 if x1 + x2 >= 1.5 else 0}",
                 "(Seul (1, 1) donne 1 : c'est la porte ET.)")
    ex = Exercise(f"Perceptron w = (1, 1), b = −1.5, sortie 1 si z ≥ 0 sinon 0. Sortie pour x = ({x1}, {x2}) ?",
                    Number(1 if x1 + x2 - 1.5 >= 0 else 0),
                    "Calcule z = w·x + b. (Ce neurone réalise la porte logique ET.)", neuron_draw(2))
    ex.solution = _sol
    return ex


def l1_classify(r):
    z = r.choice([-2, -0.5, 0.5, 1.5, 3, -4])
    _sol = lines("σ(z) > 0.5 ⇔ z > 0", f"z = {z} {'> 0 → classe 1' if z > 0 else '< 0 → classe 0'}")
    ex = Exercise(f"Classifieur binaire : p = σ(z), on prédit la classe 1 si p > 0.5. Classe prédite pour z = {z} ?",
                    Number(1 if z > 0 else 0), "σ(z) > 0.5 ⇔ z > 0. Pas besoin de calculer σ !",
                    curves(lambda t: 1 / (1 + np.exp(-t)), lambda t: 0.5 + 0 * t, xlim=(-6, 6)))
    ex.solution = _sol
    return ex


def l1_tanh(r):
    z = r.choice([0, 0.5, 1, -1, 2])
    _sol = lines(f"e^{z} ≈ {math.exp(z):.4f}, e^{-z:g} ≈ {math.exp(-z):.4f}",
                 f"tanh = ({math.exp(z):.4f} − {math.exp(-z):.4f}) / ({math.exp(z):.4f} + {math.exp(-z):.4f}) ≈ {math.tanh(z):.4f}")
    ex = Exercise(f"tanh({z}) ?" + CALC, Number(math.tanh(z), 5e-3), M_NEURON, curves(np.tanh, xlim=(-4, 4)))
    ex.solution = _sol
    return ex


def l1_leaky(r):
    z, a = r.choice([-3, -10, 2, -0.5, 5]), r.choice([0.01, 0.1, 0.2])
    _sol = lines(f"z = {z} {'> 0 : on garde z' if z > 0 else f'< 0 : on multiplie par {a}'}", f"Sortie = {z if z > 0 else a * z:g}")
    ex = Exercise(f"LeakyReLU de pente {a} pour z < 0. Valeur en z = {z} ?", Number(z if z > 0 else a * z),
                    "LeakyReLU(z) = z si z > 0, sinon a·z. Elle laisse passer un petit gradient pour z < 0.",
                    curves(lambda t: np.where(t > 0, t, a * t), xlim=(-6, 6)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 2

def l2_layer(r):
    W = [[r.randint(-2, 2) for _ in range(3)] for _ in range(2)]
    x, b = [r.randint(-2, 2) for _ in range(3)], [r.randint(-2, 2) for _ in range(2)]
    z = [dot(row, x) + c for row, c in zip(W, b)]
    _sol = lines(*[f"neurone {i + 1} : {dstr(row, x)} + {par(c)} = {zi}" for i, (row, c, zi) in enumerate(zip(W, b, z))],
                 f"ReLU : {vec([max(0, v) for v in z])}")
    ex = Exercise(f"Couche ReLU : W = {mat(W)}, x = {vec(x)}, b = {vec(b)}. a = ReLU(W·x + b) ?\nRéponse : 2 composantes.",
                    Numbers([max(0, v) for v in z], ordered=True), M_NEURON + "\nUne couche = plusieurs neurones en parallèle (une ligne de W par neurone).",
                    network([3, 2], labels=["x", "a"]))
    ex.solution = _sol
    return ex


def l2_params_layer(r):
    i, o = r.randint(2, 50), r.randint(1, 20)
    _sol = lines(f"Poids : {i} × {o} = {i * o} ; biais : {o}", f"Total : {i * o + o}")
    ex = Exercise(f"Paramètres d'une couche dense {i} → {o} ?", Number(i * o + o), "Poids : entrées × sorties. Biais : un par sortie.",
                    network([min(i, 5), min(o, 5)]))
    ex.solution = _sol
    return ex


def l2_params_net(r):
    sizes = [r.choice([2, 3, 4, 10]), r.choice([4, 8, 16]), r.choice([4, 8]), r.choice([1, 2, 3])]
    total = sum(a * b + b for a, b in zip(sizes, sizes[1:]))
    _sol = lines(*[f"Couche {a} → {b} : {a} × {b} + {b} = {a * b + b}" for a, b in zip(sizes, sizes[1:])], f"Total : {total}")
    ex = Exercise(f"Réseau dense de tailles {' → '.join(map(str, sizes))}. Nombre total de paramètres ?", Number(total),
                    "Additionne (entrées × sorties + sorties) pour chaque couche.", network([min(s, 6) for s in sizes]))
    ex.solution = _sol
    return ex


def l2_shape(r):
    n, d, h = r.choice([16, 32]), r.choice([4, 10]), r.choice([8, 64])
    _sol = lines(f"({n}×{d}) · ({d}×{h}) = {n}×{h}", f"{n} exemples, {h} sorties chacun.")
    ex = Exercise(f"Entrée : batch de {n} exemples à {d} features. Première couche : {d} → {h}. Forme de la sortie ?\n"
                    "Réponse : lignes, colonnes", Numbers([n, h], ordered=True), "Chaque exemple du batch donne un vecteur de taille " + str(h) + ".",
                    None)
    ex.solution = _sol
    return ex


def l2_softmax(r):
    z = r.choice([[0, 0], [1, 0], [2, 0, 0], [1, 1, 1], [0, math.log(3)]])
    s = sum(math.exp(v) for v in z)
    _e = [math.exp(v) for v in z]
    _sol = lines("e^z : " + ", ".join(f"{v:.4f}" for v in _e), f"Somme : {s:.4f}", "Probabilités : " + ", ".join(f"{v / s:.4f}" for v in _e))
    ex = Exercise(f"Softmax de z = {[round(v, 4) for v in z]} ?\nRéponse : les probabilités dans l'ordre (calculatrice ; 3 décimales).",
                    Numbers([math.exp(v) / s for v in z], ordered=True, tol=5e-3),
                    "softmax(z)ᵢ = e^(zᵢ) / Σ e^(zⱼ). Les résultats sont positifs et somment à 1.",
                    bars([str(i + 1) for i in range(len(z))], z, "logits"))
    ex.solution = _sol
    return ex


def l2_argmax(r):
    p = [round(r.random(), 2) for _ in range(4)]
    while len(set(p)) < 4:
        p = [round(r.random(), 2) for _ in range(4)]
    _sol = lines(f"Score maximal : {max(p)}", f"C'est la classe {p.index(max(p)) + 1}.")
    ex = Exercise(f"Sorties du réseau (scores) : {p}. Quelle classe (numérotée à partir de 1) est prédite ?", Number(p.index(max(p)) + 1),
                    "La prédiction est la classe de score maximal (argmax). Softmax ne change pas l'ordre.",
                    bars(["1", "2", "3", "4"], p))
    ex.solution = _sol
    return ex


def l2_mlp(r):
    W1 = [[r.randint(-2, 2), r.randint(-2, 2)] for _ in range(2)]
    w2, x = [r.randint(-2, 2) for _ in range(2)], [r.randint(-2, 2) for _ in range(2)]
    h = [max(0, dot(row, x)) for row in W1]
    _z = [dot(row, x) for row in W1]
    _sol = lines(*[f"z{i + 1} = {dstr(row, x)} = {zi} → h{i + 1} = ReLU = {max(0, zi)}" for i, (row, zi) in enumerate(zip(W1, _z))],
                 f"y = w₂·h = {dstr(w2, h)} = {dot(w2, h)}")
    ex = Exercise(f"MLP 2 → 2 → 1, sans biais : h = ReLU(W₁·x), y = w₂·h. W₁ = {mat(W1)}, w₂ = {vec(w2)}, x = {vec(x)}. y ?",
                    Number(dot(w2, h)), "Passe avant couche par couche : calcule W₁·x, applique ReLU, puis le produit scalaire avec w₂.",
                    network([2, 2, 1], labels=["x", "h", "y"]))
    ex.solution = _sol
    return ex


def l2_linear_collapse(r):
    W1 = [[r.randint(-2, 2) for _ in range(2)] for _ in range(2)]
    W2 = [[r.randint(-2, 2) for _ in range(2)] for _ in range(2)]
    P = (np.array(W2) @ np.array(W1)).tolist()
    _sol = lines("y = W₂(W₁x) = (W₂W₁)x : on calcule W = W₂·W₁",
                 *[f"c{i + 1}{j + 1} = {dstr(W2[i], [W1[0][j], W1[1][j]])} = {P[i][j]}" for i in range(2) for j in range(2)], f"W = {mat(P)}")
    ex = Exercise(f"Deux couches SANS activation : y = W₂·(W₁·x). W₁ = {mat(W1)}, W₂ = {mat(W2)}. Donne la matrice unique "
                    "W équivalente (y = W·x).\nRéponse : 4 coefficients ligne par ligne.", Numbers([c for row in P for c in row], ordered=True),
                    "W = W₂·W₁. Sans activation non linéaire, empiler des couches ne sert à rien : c'est une seule couche linéaire.",
                    matrices([("W₂", W2), ("W₁", W1)], ["·"]))
    ex.solution = _sol
    return ex


def l2_xor(r):
    _sol = lines("Points à 1 : (0,1) et (1,0) ; points à 0 : (0,0) et (1,1).", "Ils sont « en croix » : toute droite qui sépare (0,1) de (0,0) laisse (1,1) du mauvais côté.",
                 "Un seul neurone = une seule droite : impossible. Il faut une couche cachée.")
    ex = Exercise("Un SEUL neurone à seuil (une droite de séparation) peut-il réaliser la porte XOR ? (oui / non)",
                    Choice("non", ["oui", "non"]),
                    "XOR vaut 1 pour (0,1) et (1,0), 0 pour (0,0) et (1,1). Essaie de séparer ces points par une seule droite.",
                    lambda ax: (ax.scatter([0, 1], [1, 0], s=150, color="#2f855a", label="1"),
                                ax.scatter([0, 1], [0, 1], s=150, color="#c53030", label="0"), ax.legend(), ax.set_aspect("equal"),
                                ax.set_xlim(-0.5, 1.5), ax.set_ylim(-0.5, 1.5), ax.grid(True, alpha=0.3)))
    ex.solution = _sol
    return ex


def l2_bias_shift(r):
    b = r.randint(-5, 5)
    _sol = lines("σ(z) = 0.5 ⇔ z = 0", f"x {'+' if b >= 0 else '−'} {abs(b)} = 0 ⇔ x = {-b}")
    ex = Exercise(f"Neurone sigmoïde à une entrée : a = σ(x + b) avec b = {b}. Pour quelle valeur de x a-t-on a = 0.5 ?", Number(-b),
                    "σ(z) = 0.5 ⇔ z = 0. Le biais décale la courbe horizontalement.",
                    curves(lambda t: 1 / (1 + np.exp(-t)), lambda t: 1 / (1 + np.exp(-(t + b))), xlim=(-10, 10), labels=["σ(x)", "σ(x + b)"]))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 3

def l3_mse(r):
    y, p = [r.randint(0, 4) for _ in range(3)], [r.randint(0, 4) for _ in range(3)]
    _e = [a - b for a, b in zip(p, y)]
    _sol = lines(f"Erreurs : {_e} ; carrés : {[e * e for e in _e]}", f"MSE = {sum(e * e for e in _e)} / 3 = {fmt(F(sum(e * e for e in _e), 3))}")
    ex = Exercise(f"MSE entre ŷ = {vec(p)} et y = {vec(y)} ?", Number(F(sum((a - b) ** 2 for a, b in zip(p, y)), 3)),
                    "MSE = moyenne des carrés des erreurs.", None)
    ex.solution = _sol
    return ex


def l3_bce(r):
    y, p = r.choice([0, 1]), r.choice([0.9, 0.6, 0.2, 0.99, 0.5])
    _sol = lines(f"y = {y} : L = −ln({'p' if y == 1 else '1 − p'})", f"= −ln({p if y == 1 else round(1 - p, 4):g}) ≈ {-(y * math.log(p) + (1 - y) * math.log(1 - p)):.4f}")
    ex = Exercise(f"Entropie croisée binaire (en ln) : y = {y}, p = {p}. L = −[y·ln p + (1 − y)·ln(1 − p)] ?" + CALC,
                    Number(-(y * math.log(p) + (1 - y) * math.log(1 - p)), 5e-3),
                    "Un seul des deux termes survit selon y.", curves(lambda t: -np.log(t), xlim=(0.01, 1), ylim=(0, 5)))
    ex.solution = _sol
    return ex


def l3_ce(r):
    p = r.choice([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.25, 0.25, 0.5]])
    k = r.randint(0, 2)
    _sol = lines(f"Probabilité de la vraie classe {k + 1} : {p[k]}", f"L = −ln({p[k]}) ≈ {-math.log(p[k]):.4f}")
    ex = Exercise(f"Probas softmax {p}, vraie classe {k + 1}. Entropie croisée (en ln) ?" + CALC, Number(-math.log(p[k]), 5e-3),
                    "Avec une cible one-hot : L = −ln(probabilité donnée à la vraie classe).", bars(["1", "2", "3"], p))
    ex.solution = _sol
    return ex


def l3_accuracy(r):
    yt = [r.randint(0, 2) for _ in range(8)]
    yp = [v if r.random() < 0.7 else (v + 1) % 3 for v in yt]
    _ok = sum(a == b for a, b in zip(yt, yp))
    _sol = lines("Comparaison position par position : " + " ".join("✔" if a == b else "✘" for a, b in zip(yt, yp)), f"{_ok} bonnes sur 8 = {fmt(F(_ok, 8))}")
    ex = Exercise(f"Vraies classes {yt}, prédictions {yp}. Exactitude (proportion de bonnes réponses) ?",
                    Number(F(sum(a == b for a, b in zip(yt, yp)), 8)), "Compte les positions où prédiction = vérité, divise par le total.",
                    None)
    ex.solution = _sol
    return ex


def l3_dmse(r):
    yh, y = r.randint(-3, 5), r.randint(-3, 5)
    _sol = lines("∂L/∂ŷ = 2(ŷ − y)", f"= 2 × ({yh} − {par(y)}) = 2 × {par(yh - y)} = {2 * (yh - y)}")
    ex = Exercise(f"Perte L = (ŷ − y)² pour un exemple. ŷ = {yh}, y = {y}. ∂L/∂ŷ ?", Number(2 * (yh - y)),
                    "Dérive le carré : 2(ŷ − y). C'est le signal d'erreur qui démarre la rétropropagation.", None)
    ex.solution = _sol
    return ex


def l3_dbce_logit(r):
    z, y = r.choice([0, 1, -1, 2]), r.choice([0, 1])
    _sol = lines(f"p = σ({z}) ≈ {sig(z):.4f}", f"∂L/∂z = p − y ≈ {sig(z):.4f} − {y} = {sig(z) - y:.4f}")
    ex = Exercise(f"Sortie p = σ(z) avec perte entropie croisée binaire. z = {z}, y = {y}. ∂L/∂z = p − y ?" + CALC,
                    Number(sig(z) - y, 5e-3), "Résultat classique : la dérivée par rapport au logit est simplement p − y.", None)
    ex.solution = _sol
    return ex


def l3_l2pen(r):
    w = [r.randint(-3, 3) for _ in range(4)]
    lam = r.choice([0.01, 0.1, 0.5])
    _sol = lines(f"Σw² = {' + '.join(f'{par(v)}²' for v in w)} = {sum(v * v for v in w)}", f"λ·Σw² = {lam} × {sum(v * v for v in w)} = {lam * sum(v * v for v in w):g}")
    ex = Exercise(f"Pénalité L2 = λ·Σwᵢ² avec λ = {lam}, w = {vec(w)} ?", Number(lam * sum(v * v for v in w)), "Somme des carrés × λ.",
                    bars(["1", "2", "3", "4"], w))
    ex.solution = _sol
    return ex


def l3_total(r):
    data, lam = round(r.uniform(0.1, 2), 2), r.choice([0.01, 0.1])
    w = [r.randint(-3, 3) for _ in range(3)]
    _sol = lines(f"‖w‖² = {sum(v * v for v in w)}", f"Total = {data} + {lam} × {sum(v * v for v in w)} = {data + lam * sum(v * v for v in w):g}")
    ex = Exercise(f"Perte données = {data}, λ = {lam}, w = {vec(w)}. Perte totale = perte données + λ·‖w‖² ?",
                    Number(data + lam * sum(v * v for v in w)), "Additionne la perte et la pénalité.", None)
    ex.solution = _sol
    return ex


def l3_hinge(r):
    y, s = r.choice([1, -1]), r.choice([-2, -0.5, 0, 0.5, 1, 2.5])
    _sol = lines(f"y·s = {y} × {par(s)} = {y * s:g}", f"max(0, 1 − {par(y * s)}) = max(0, {1 - y * s:g}) = {max(0, 1 - y * s):g}")
    ex = Exercise(f"Perte hinge (SVM) : max(0, 1 − y·s) avec y = {y}, score s = {s} ?", Number(max(0, 1 - y * s)),
                    "Nulle si le point est bien classé avec une marge ≥ 1 ; sinon elle croît linéairement.", None)
    ex.solution = _sol
    return ex


def l3_mae(r):
    y, p = [r.randint(0, 6) for _ in range(4)], [r.randint(0, 6) for _ in range(4)]
    _e = [abs(a - b) for a, b in zip(p, y)]
    _sol = lines(f"|erreurs| : {_e}", f"MAE = {sum(_e)} / 4 = {fmt(F(sum(_e), 4))}")
    ex = Exercise(f"MAE entre ŷ = {vec(p)} et y = {vec(y)} ?", Number(F(sum(abs(a - b) for a, b in zip(p, y)), 4)),
                    "Moyenne des valeurs absolues des erreurs. Moins sensible aux valeurs aberrantes que la MSE.", None)
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 4

def l4_dw(r):
    w, b, x, y = r.randint(-2, 2), r.randint(-2, 2), nz(r, -3, 3), r.randint(-4, 4)
    _e = w * x + b - y
    _sol = lines(f"ŷ = {w} × {par(x)} {'+' if b >= 0 else '−'} {abs(b)} = {w * x + b} ; ŷ − y = {_e}", f"∂L/∂ŷ = 2(ŷ − y) = {2 * _e}", f"∂L/∂w = {2 * _e} × x = {2 * _e} × {par(x)} = {2 * _e * x}")
    ex = Exercise(f"ŷ = w·x + b, L = (ŷ − y)². w = {w}, b = {b}, x = {x}, y = {y}. ∂L/∂w ?", Number(2 * (w * x + b - y) * x), M_BACK,
                    network([1, 1], labels=["x", "ŷ"]))
    ex.solution = _sol
    return ex


def l4_db(r):
    w, b, x, y = r.randint(-2, 2), r.randint(-2, 2), nz(r, -3, 3), r.randint(-4, 4)
    _e = w * x + b - y
    _sol = lines(f"ŷ = {w * x + b} ; ŷ − y = {_e}", f"∂L/∂b = 2(ŷ − y) × 1 = {2 * _e}")
    ex = Exercise(f"ŷ = w·x + b, L = (ŷ − y)². w = {w}, b = {b}, x = {x}, y = {y}. ∂L/∂b ?", Number(2 * (w * x + b - y)), M_BACK, None)
    ex.solution = _sol
    return ex


def l4_sigmoid_bce(r):
    w, x, y = r.choice([0, 1, -1, 0.5]), r.choice([1, 2, -1]), r.choice([0, 1])
    _p = sig(w * x)
    _sol = lines(f"z = {w} × {par(x)} = {w * x:g} ; p = σ(z) ≈ {_p:.4f}", f"∂L/∂z = p − y ≈ {_p - y:.4f}", f"∂L/∂w = (p − y)·x ≈ {_p - y:.4f} × {par(x)} = {(_p - y) * x:.4f}")
    ex = Exercise(f"p = σ(w·x), entropie croisée binaire. w = {w}, x = {x}, y = {y}. ∂L/∂w ?" + CALC,
                    Number((sig(w * x) - y) * x, 5e-3), M_BACK + "\n∂L/∂z = p − y, puis ∂z/∂w = x.", None)
    ex.solution = _sol
    return ex


def l4_relu_gate(r):
    z, up = r.choice([-2, -0.5, 1, 3]), r.randint(-5, 5)
    _sol = lines(f"z = {z} {'> 0 : ReLU laisse passer (dérivée 1)' if z > 0 else '< 0 : ReLU bloque (dérivée 0)'}", f"Gradient transmis = {up} × {1 if z > 0 else 0} = {up if z > 0 else 0}")
    ex = Exercise(f"Le gradient qui arrive sur la sortie d'un ReLU vaut {up}. L'entrée du ReLU était z = {z}. Gradient transmis à z ?",
                    Number(up if z > 0 else 0), M_BACK, curves(lambda t: np.maximum(0, t), xlim=(-4, 4), points=[(z, max(0, z))]))
    ex.solution = _sol
    return ex


def l4_hidden_delta(r):
    w2, d_out, z = nz(r, -3, 3), r.randint(-4, 4), r.choice([-1, 2, 0.5, -3])
    _sol = lines(f"ReLU'({z}) = {1 if z > 0 else 0}", f"δ_h = w₂ × δ_out × ReLU'(z) = {w2} × {par(d_out)} × {1 if z > 0 else 0} = {w2 * d_out * (1 if z > 0 else 0)}")
    ex = Exercise(f"Neurone caché ReLU de pré-activation z = {z}, relié à la sortie par le poids w₂ = {w2}. Le gradient en sortie vaut "
                    f"δ_out = {d_out}. Gradient δ_h = ∂L/∂z du neurone caché ?", Number(w2 * d_out * (1 if z > 0 else 0)),
                    M_BACK + "\nδ_h = w₂·δ_out·ReLU'(z).", network([1, 1, 1], labels=["x", "h", "ŷ"]))
    ex.solution = _sol
    return ex


def l4_dW1(r):
    dh, x = r.randint(-4, 4), r.randint(-3, 3)
    _sol = lines("∂z_h/∂w₁ = x", f"∂L/∂w₁ = δ_h × x = {dh} × {par(x)} = {dh * x}")
    ex = Exercise(f"Pour le neurone caché, δ_h = ∂L/∂z_h = {dh} et z_h = w₁·x + b₁ avec x = {x}. ∂L/∂w₁ ?", Number(dh * x),
                    M_BACK, None)
    ex.solution = _sol
    return ex


def l4_update(r):
    w, g, al = r.randint(-5, 5), r.randint(-6, 6), r.choice([0.1, 0.01, 0.5])
    _sol = lines("w ← w − α·∂L/∂w", f"= {w} − {al} × {par(g)} = {w - al * g:g}")
    ex = Exercise(f"w = {w}, ∂L/∂w = {g}, learning rate {al}. Nouveau w ?", Number(w - al * g), "w ← w − α·∂L/∂w.", None)
    ex.solution = _sol
    return ex


def l4_sum_node(r):
    up = r.randint(-6, 6)
    _sol = lines("∂(a + b)/∂a = 1", f"Gradient transmis à a = {up} × 1 = {up}")
    ex = Exercise(f"Nœud d'addition z = a + b. Le gradient qui arrive sur z vaut {up}. Gradient transmis à a ?", Number(up),
                    "∂z/∂a = 1 : une addition « distribue » le gradient tel quel à toutes ses entrées.", None)
    ex.solution = _sol
    return ex


def l4_mul_node(r):
    a, b, up = r.randint(-4, 4), r.randint(-4, 4), r.randint(-3, 3)
    _sol = lines("∂(a·b)/∂a = b", f"Gradient transmis à a = {up} × {par(b)} = {up * b}")
    ex = Exercise(f"Nœud de multiplication z = a·b avec a = {a}, b = {b}. Gradient arrivant sur z : {up}. Gradient transmis à a ?",
                    Number(up * b), "∂z/∂a = b : une multiplication « échange » les valeurs. Multiplie par le gradient d'en haut.", None)
    ex.solution = _sol
    return ex


def l4_sig_deriv(r):
    z = r.choice([0, 1, -1, 2, -3])
    s = sig(z)
    _sol = lines(f"σ({z}) ≈ {s:.4f}", f"σ'({z}) = {s:.4f} × (1 − {s:.4f}) ≈ {s * (1 - s):.4f}")
    ex = Exercise(f"σ'(z) = σ(z)·(1 − σ(z)). Calcule σ'({z})." + CALC, Number(s * (1 - s), 5e-3),
                    "Calcule σ(z) puis le produit. Maximum en z = 0 : 0.25.", curves(lambda t: 1 / (1 + np.exp(-t)) * (1 - 1 / (1 + np.exp(-t))), xlim=(-6, 6)))
    ex.solution = _sol
    return ex


# ================================================================ NIVEAU 5

def l5_full(r):
    x, w1, w2, y = r.choice([1, 2]), r.choice([1, -1, 2]), r.choice([1, 2, -1]), r.randint(-2, 4)
    h = max(0, w1 * x)
    yh = w2 * h
    g = 2 * (yh - y) * w2 * (1 if w1 * x > 0 else 0) * x
    _z = w1 * x
    _sol = lines(f"Passe avant : z = {w1} × {x} = {_z}, h = ReLU(z) = {h}, ŷ = {w2} × {h} = {yh}",
                 f"∂L/∂ŷ = 2(ŷ − y) = 2 × ({yh} − {par(y)}) = {2 * (yh - y)}", f"∂L/∂h = ∂L/∂ŷ × w₂ = {2 * (yh - y)} × {par(w2)} = {2 * (yh - y) * w2}",
                 f"∂L/∂z = ∂L/∂h × ReLU'(z) = {2 * (yh - y) * w2} × {1 if _z > 0 else 0} = {2 * (yh - y) * w2 * (1 if _z > 0 else 0)}",
                 f"∂L/∂w₁ = ∂L/∂z × x = {2 * (yh - y) * w2 * (1 if _z > 0 else 0)} × {x} = {g}")
    ex = Exercise(f"Réseau x → h = ReLU(w₁·x) → ŷ = w₂·h, perte (ŷ − y)². x = {x}, w₁ = {w1}, w₂ = {w2}, y = {y}. ∂L/∂w₁ ?",
                    Number(g), M_BACK + "\nPasse avant (h, ŷ), puis ∂L/∂ŷ → ∂L/∂h = ∂L/∂ŷ·w₂ → ∂L/∂z = ·ReLU' → ∂L/∂w₁ = ·x.",
                    network([1, 1, 1], labels=["x", "h", "ŷ"]))
    ex.solution = _sol
    return ex


def l5_vanishing(r):
    n = r.randint(3, 10)
    _sol = lines(f"0.25^{n} = (2⁻²)^{n} = 2^(−2 × {n})", f"k = {2 * n}  (soit environ {0.25 ** n:.2e} : le gradient a presque disparu)")
    ex = Exercise(f"La dérivée de la sigmoïde vaut au plus 0.25 = 2⁻². Après {n} couches sigmoïdes (poids égaux à 1), le gradient "
                    f"est multiplié au plus par 0.25^{n}. Écris ce nombre sous la forme 2⁻ᵏ : que vaut k ?", Number(2 * n),
                    "Le gradient est un produit de dérivées : il rétrécit exponentiellement avec la profondeur (gradient qui disparaît). "
                    "D'où ReLU, les connexions résiduelles, la normalisation...", None)
    ex.solution = _sol
    return ex


def l5_softmax_ce_grad(r):
    z = r.choice([[0, 0, 0], [1, 0, 0], [0, 0, math.log(2)]])
    k = r.randint(0, 2)
    s = sum(math.exp(v) for v in z)
    p = [math.exp(v) / s for v in z]
    _sol = lines("p = softmax(z) : " + ", ".join(f"{v:.4f}" for v in p), f"On retire 1 à la composante {k + 1} (vraie classe)",
                 "p − y = " + ", ".join(f"{v - (1 if i == k else 0):.4f}" for i, v in enumerate(p)))
    ex = Exercise(f"Logits z = {[round(v, 4) for v in z]}, vraie classe {k + 1}. ∂L/∂z (entropie croisée) = p − y ?\n"
                    "Réponse : 3 composantes (calculatrice ; 3 décimales).",
                    Numbers([pi - (1 if i == k else 0) for i, pi in enumerate(p)], ordered=True, tol=5e-3),
                    "p = softmax(z) ; y = one-hot de la vraie classe.", None)
    ex.solution = _sol
    return ex


def l5_dropout(r):
    p, a = r.choice([0.5, 0.2, 0.25, 0.1]), r.randint(1, 8)
    _sol = lines(f"Facteur = 1/(1 − {p}) = 1/{1 - p:g} = {1 / (1 - p):.4g}", f"{a} × {1 / (1 - p):.4g} ≈ {a / (1 - p):.4f}")
    ex = Exercise(f"Dropout « inversé » avec probabilité d'éteindre p = {p}. Un neurone gardé de valeur {a} est multiplié par "
                    "1/(1 − p) pendant l'entraînement. Nouvelle valeur ?", Number(a / (1 - p)),
                    "On compense les neurones éteints pour garder la même espérance qu'au test.", None)
    ex.solution = _sol
    return ex


def l5_bn(r):
    xs = r.choice([[1, 3, 5, 7], [2, 2, 6, 6], [0, 4, 4, 8]])
    i = r.randint(0, 3)
    gamma, beta = r.choice([1, 2]), r.choice([0, 1])
    m = sum(xs) / 4
    v = sum((x - m) ** 2 for x in xs) / 4
    _sol = lines(f"μ = {m:g} ; σ² = {v:g} ; σ = {math.sqrt(v):.4f}", f"x̂ = ({xs[i]} − {m:g}) / {math.sqrt(v):.4f} ≈ {(xs[i] - m) / math.sqrt(v):.4f}",
                 f"γ·x̂ + β = {gamma} × {(xs[i] - m) / math.sqrt(v):.4f} + {beta} ≈ {gamma * (xs[i] - m) / math.sqrt(v) + beta:.4f}")
    ex = Exercise(f"Batch norm sur {xs} (ε négligé), γ = {gamma}, β = {beta}. Sortie pour la valeur {xs[i]} : γ·x̂ + β ?" + CALC,
                    Number(gamma * (xs[i] - m) / math.sqrt(v) + beta, 5e-3),
                    "x̂ = (x − moyenne du batch) / écart-type du batch, puis applique γ et β (appris).", bars(["1", "2", "3", "4"], xs))
    ex.solution = _sol
    return ex


def l5_conv_len(r):
    n, k, p, s = r.choice([28, 32, 10, 64]), r.choice([3, 5]), r.choice([0, 1, 2]), r.choice([1, 2])
    _sol = lines(f"(n + 2p − k) / s + 1 = ({n} + 2 × {p} − {k}) / {s} + 1", f"= {n + 2 * p - k} / {s} + 1 → ⌊{(n + 2 * p - k) / s:g}⌋ + 1 = {(n + 2 * p - k) // s + 1}")
    ex = Exercise(f"Convolution 1D : entrée de longueur {n}, noyau {k}, padding {p}, stride {s}. Longueur de sortie ?",
                    Number((n + 2 * p - k) // s + 1), "Sortie = ⌊(n + 2p − k) / s⌋ + 1.", None)
    ex.solution = _sol
    return ex


def l5_conv_value(r):
    x = [r.randint(-2, 3) for _ in range(5)]
    k = [r.randint(-1, 1) for _ in range(3)]
    i = r.randint(0, 2)
    _win = x[i:i + 3]
    _sol = lines(f"Fenêtre n°{i + 1} (positions {i + 1} à {i + 3}) : {_win}", f"Produit avec le noyau {k} : {dstr(_win, k)} = {dot(_win, k)}")
    ex = Exercise(f"Convolution 1D (sans retournement, comme en deep learning) : entrée {x}, noyau {k}, stride 1, pas de padding. "
                    f"Valeur de la sortie n°{i + 1} ?", Number(dot(x[i:i + 3], k)),
                    "La sortie i est le produit scalaire du noyau avec la fenêtre de l'entrée qui commence en position i.",
                    bars([str(j + 1) for j in range(5)], x, "entrée"))
    ex.solution = _sol
    return ex


def l5_maxpool(r):
    x = [r.randint(-5, 9) for _ in range(6)]
    _sol = lines(*[f"bloc {j // 2 + 1} : max({x[j]}, {x[j + 1]}) = {max(x[j], x[j + 1])}" for j in range(0, 6, 2)])
    ex = Exercise(f"Max pooling 1D, fenêtre 2, stride 2, sur {x}. Sortie ?\nRéponse : 3 composantes.",
                    Numbers([max(x[i], x[i + 1]) for i in range(0, 6, 2)], ordered=True),
                    "Découpe en blocs de 2 sans chevauchement, garde le maximum de chaque bloc.", bars([str(j + 1) for j in range(6)], x))
    ex.solution = _sol
    return ex


def l5_conv_params(r):
    k, cin, cout = r.choice([3, 5, 1]), r.choice([1, 3, 16, 32]), r.choice([8, 16, 32, 64])
    _sol = lines(f"Un filtre : {k} × {k} × {cin} = {k * k * cin} poids + 1 biais", f"{cout} filtres : {cout} × {k * k * cin + 1} = {cout * (k * k * cin + 1)}")
    ex = Exercise(f"Couche Conv2D : noyau {k}×{k}, {cin} canaux en entrée, {cout} filtres. Nombre de paramètres (avec biais) ?",
                    Number(k * k * cin * cout + cout), "Chaque filtre a k·k·C_in poids + 1 biais. Multiplie par le nombre de filtres.", None)
    ex.solution = _sol
    return ex


def l5_attention(r):
    q = [r.randint(-1, 2) for _ in range(2)]
    k1, k2 = [r.randint(-1, 2) for _ in range(2)], [r.randint(-1, 2) for _ in range(2)]
    s1, s2 = dot(q, k1) / math.sqrt(2), dot(q, k2) / math.sqrt(2)
    w1 = math.exp(s1) / (math.exp(s1) + math.exp(s2))
    _sol = lines(f"q·k₁ = {dstr(q, k1)} = {dot(q, k1)} ; q·k₂ = {dstr(q, k2)} = {dot(q, k2)}",
                 f"Scores / √2 : s₁ ≈ {s1:.4f}, s₂ ≈ {s2:.4f}", f"Poids sur k₁ = e^s₁ / (e^s₁ + e^s₂) ≈ {math.exp(s1):.4f} / ({math.exp(s1):.4f} + {math.exp(s2):.4f}) ≈ {w1:.4f}")
    ex = Exercise(f"Attention : q = {vec(q)}, clés k₁ = {vec(k1)}, k₂ = {vec(k2)}, d = 2. Poids d'attention sur k₁ = "
                    "softmax(q·k / √d)₁ ?" + CALC, Number(w1, 5e-3),
                    "Scores sᵢ = q·kᵢ / √d, puis softmax sur les deux scores. Le √d évite que les scores deviennent trop grands.",
                    None)
    ex.solution = _sol
    return ex


LEVELS = {
    1: [l1_weighted, l1_relu, l1_sigmoid, l1_step, l1_neuron_relu, l1_n_params, l1_and, l1_classify, l1_tanh, l1_leaky],
    2: [l2_layer, l2_params_layer, l2_params_net, l2_shape, l2_softmax, l2_argmax, l2_mlp, l2_linear_collapse, l2_xor,
        l2_bias_shift],
    3: [l3_mse, l3_bce, l3_ce, l3_accuracy, l3_dmse, l3_dbce_logit, l3_l2pen, l3_total, l3_hinge, l3_mae],
    4: [l4_dw, l4_db, l4_sigmoid_bce, l4_relu_gate, l4_hidden_delta, l4_dW1, l4_update, l4_sum_node, l4_mul_node,
        l4_sig_deriv],
    5: [l5_full, l5_vanishing, l5_softmax_ce_grad, l5_dropout, l5_bn, l5_conv_len, l5_conv_value, l5_maxpool,
        l5_conv_params, l5_attention],
}
