"""Matrices : opérations, produit, déterminant, inverse, transformations, valeurs propres, couches de réseau."""

from __future__ import annotations

from fractions import Fraction as F

import numpy as np
import sympy as sp

from ..answers import Choice, Expr, Number, Numbers
from ..core import Exercise, mat, matrices, nz, transform_square, vec

DESCRIPTIONS = {
    1: "Notations, addition, transposée, matrice × vecteur",
    2: "Produit de matrices, déterminant, inverse 2×2",
    3: "Matrices comme transformations du plan, déterminant 3×3",
    4: "Valeurs propres, rang, systèmes linéaires",
    5: "Matrices en IA : couches denses, dimensions, gradients, ACP",
}

FLAT = "\nRéponse : les coefficients ligne par ligne, séparés par des virgules (ex : 1, 2, 3, 4)."
M_PROD = ("Produit AB : le coefficient (i, j) est le produit scalaire de la ligne i de A et de la colonne j de B.\n"
          "Possible seulement si (nb de colonnes de A) = (nb de lignes de B). Dimensions : (m×n)·(n×p) = m×p.")
M_DET = "det [[a, b], [c, d]] = ad − bc.  Inverse : A⁻¹ = (1/det)·[[d, −b], [−c, a]] (si det ≠ 0)."


def rm(r, rows, cols, lo=-4, hi=5):
    return [[r.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def flat(M):
    return [c for row in M for c in row]


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def mv(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def T(A):
    return [list(c) for c in zip(*A)]


def det2(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def show(*items, ops=None):
    return matrices(list(items), ops)


def col(v):
    return [[c] for c in v]


# ================================================================ NIVEAU 1

def l1_add(r):
    A, B = rm(r, 2, 2), rm(r, 2, 2)
    return Exercise(f"A = {mat(A)}, B = {mat(B)}. Calcule A + B." + FLAT,
                    Numbers(flat([[a + b for a, b in zip(x, y)] for x, y in zip(A, B)]), ordered=True),
                    "Addition coefficient par coefficient (les matrices doivent avoir la même taille).",
                    show(("A", A), ("B", B), ops=["+"]))


def l1_scale(r):
    A, k = rm(r, 2, 3), nz(r, -3, 3)
    return Exercise(f"A = {mat(A)}. Calcule {k}·A." + FLAT, Numbers([k * c for c in flat(A)], ordered=True),
                    "Chaque coefficient est multiplié par le nombre.", show(("A", A)))


def l1_transpose(r):
    A = rm(r, 2, 3)
    return Exercise(f"A = {mat(A)} (2 lignes, 3 colonnes). Donne Aᵀ (3 lignes, 2 colonnes)." + FLAT,
                    Numbers(flat(T(A)), ordered=True), "Transposer : les lignes deviennent les colonnes. (Aᵀ)ᵢⱼ = Aⱼᵢ.",
                    show(("A", A)))


def l1_entry(r):
    A = rm(r, 3, 3, -9, 9)
    i, j = r.randint(1, 3), r.randint(1, 3)
    return Exercise(f"A = {mat(A)}. Que vaut le coefficient a{i}{j} (ligne {i}, colonne {j}) ?", Number(A[i - 1][j - 1]),
                    "aᵢⱼ : le premier indice est la LIGNE, le second la COLONNE. On compte à partir de 1.", show(("A", A)))


def l1_dims(r):
    m, n, p = r.randint(1, 5), r.randint(1, 5), r.randint(1, 5)
    return Exercise(f"A est de taille {m}×{n} et B de taille {n}×{p}. Quelle est la taille de AB ?\nRéponse : lignes, colonnes",
                    Numbers([m, p], ordered=True), M_PROD, None)


def l1_matvec(r):
    A, x = rm(r, 2, 2), [r.randint(-3, 3) for _ in range(2)]
    return Exercise(f"A = {mat(A)}, x = {vec(x)}. Calcule A·x.\nRéponse : 2 composantes.", Numbers(mv(A, x), ordered=True),
                    "Chaque composante du résultat = produit scalaire d'une LIGNE de A avec x.",
                    show(("A", A), ("x", col(x)), ops=["·"]))


def l1_identity(r):
    x = [r.randint(-9, 9) for _ in range(3)]
    return Exercise(f"I est la matrice identité 3×3. Calcule I·x pour x = {vec(x)}.\nRéponse : 3 composantes.",
                    Numbers(x, ordered=True), "L'identité a des 1 sur la diagonale et des 0 ailleurs : elle ne change rien.",
                    show(("I", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), ("x", col(x)), ops=["·"]))


def l1_sub(r):
    A, B = rm(r, 2, 2), rm(r, 2, 2)
    return Exercise(f"A = {mat(A)}, B = {mat(B)}. Calcule A − B." + FLAT,
                    Numbers(flat([[a - b for a, b in zip(x, y)] for x, y in zip(A, B)]), ordered=True),
                    "Soustraction coefficient par coefficient.", show(("A", A), ("B", B), ops=["−"]))


def l1_trace(r):
    A = rm(r, 3, 3, -9, 9)
    return Exercise(f"A = {mat(A)}. Calcule la trace de A.", Number(sum(A[i][i] for i in range(3))),
                    "Trace = somme des coefficients de la diagonale principale (a₁₁ + a₂₂ + a₃₃).", show(("A", A)))


def l1_matvec3(r):
    A, x = rm(r, 2, 3, -3, 3), [r.randint(-2, 3) for _ in range(3)]
    return Exercise(f"A = {mat(A)} (2×3), x = {vec(x)}. Calcule A·x.\nRéponse : 2 composantes.",
                    Numbers(mv(A, x), ordered=True), "Ligne de A · x pour chaque ligne. Une matrice 2×3 envoie R³ dans R².",
                    show(("A", A), ("x", col(x)), ops=["·"]))


# ================================================================ NIVEAU 2

def l2_prod(r):
    A, B = rm(r, 2, 2), rm(r, 2, 2)
    return Exercise(f"A = {mat(A)}, B = {mat(B)}. Calcule AB." + FLAT, Numbers(flat(mul(A, B)), ordered=True), M_PROD,
                    show(("A", A), ("B", B), ops=["·"]))


def l2_prod_rect(r):
    A, B = rm(r, 2, 3, -3, 3), rm(r, 3, 2, -3, 3)
    return Exercise(f"A = {mat(A)} (2×3), B = {mat(B)} (3×2). Calcule AB (2×2)." + FLAT,
                    Numbers(flat(mul(A, B)), ordered=True), M_PROD, show(("A", A), ("B", B), ops=["·"]))


def l2_matvec3(r):
    A, x = rm(r, 3, 3, -3, 3), [r.randint(-3, 3) for _ in range(3)]
    return Exercise(f"A = {mat(A)}, x = {vec(x)}. Calcule A·x.\nRéponse : 3 composantes.", Numbers(mv(A, x), ordered=True),
                    "Chaque composante = ligne de A · x.", show(("A", A), ("x", col(x)), ops=["·"]))


def l2_det(r):
    A = rm(r, 2, 2, -6, 7)
    return Exercise(f"Calcule le déterminant de A = {mat(A)}.", Number(det2(A)), M_DET, show(("A", A)))


def l2_inverse(r):
    while True:
        A = rm(r, 2, 2, -4, 5)
        d = det2(A)
        if d in (1, -1, 2, -2):
            break
    inv = [[F(A[1][1], d), F(-A[0][1], d)], [F(-A[1][0], d), F(A[0][0], d)]]
    return Exercise(f"Calcule l'inverse de A = {mat(A)}." + FLAT, Numbers(flat(inv), ordered=True),
                    M_DET + "\nÉchange a et d, change le signe de b et c, divise tout par det.\nVérifie : A·A⁻¹ = I.",
                    show(("A", A)))


def l2_invertible(r):
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    k = r.choice([2, -1, 3])
    A = [[a, b], [k * a, k * b]] if r.random() < 0.5 else rm(r, 2, 2, -4, 5)
    return Exercise(f"A = {mat(A)} est-elle inversible ? (oui / non)", Choice("oui" if det2(A) != 0 else "non", ["oui", "non"]),
                    "Inversible ⇔ det ≠ 0. (det = 0 : les lignes sont proportionnelles, la matrice « écrase » le plan.)",
                    transform_square(A))


def l2_square(r):
    A = rm(r, 2, 2, -3, 4)
    return Exercise(f"A = {mat(A)}. Calcule A² = A·A." + FLAT, Numbers(flat(mul(A, A)), ordered=True),
                    M_PROD + "\nAttention : A² n'est PAS la matrice des carrés des coefficients.", show(("A", A)))


def l2_transpose_prod(r):
    A, B = rm(r, 2, 2, -3, 4), rm(r, 2, 2, -3, 4)
    return Exercise(f"A = {mat(A)}, B = {mat(B)}. Calcule (AB)ᵀ." + FLAT, Numbers(flat(T(mul(A, B))), ordered=True),
                    "Calcule AB puis transpose. (Propriété utile : (AB)ᵀ = BᵀAᵀ — l'ordre s'inverse.)",
                    show(("A", A), ("B", B)))


def l2_solve(r):
    while True:
        A = rm(r, 2, 2, -4, 5)
        if det2(A) in (1, -1):
            break
    x = [r.randint(-5, 5) for _ in range(2)]
    b = mv(A, x)
    return Exercise(f"Résous A·x = b avec A = {mat(A)} et b = {vec(b)} (det A = ±1).\nRéponse : x₁, x₂",
                    Numbers(x, ordered=True), "x = A⁻¹·b. Calcule l'inverse (" + M_DET + ") puis le produit.",
                    show(("A", A), ("x", [["?"], ["?"]]), ("b", col(b)), ops=["·", "="]))


def l2_det_diag(r):
    d = [nz(r, -5, 5) for _ in range(3)]
    A = [[d[0], r.randint(-9, 9), r.randint(-9, 9)], [0, d[1], r.randint(-9, 9)], [0, 0, d[2]]]
    return Exercise(f"Calcule le déterminant de la matrice triangulaire A = {mat(A)}.", Number(d[0] * d[1] * d[2]),
                    "Pour une matrice triangulaire (ou diagonale), le déterminant est le produit des coefficients diagonaux.",
                    show(("A", A)))


# ================================================================ NIVEAU 3

TRANSFORMS = {
    "rotation de 90°": [[0, -1], [1, 0]],
    "rotation de 180°": [[-1, 0], [0, -1]],
    "symétrie par rapport à l'axe des x": [[1, 0], [0, -1]],
    "symétrie par rapport à la droite y = x": [[0, 1], [1, 0]],
    "étirement ×2 horizontal": [[2, 0], [0, 1]],
    "cisaillement horizontal de 1": [[1, 1], [0, 1]],
}
M_TRANSF = ("Une matrice 2×2 est une transformation du plan. Ses COLONNES sont les images des vecteurs de base : "
            "colonne 1 = image de (1, 0), colonne 2 = image de (0, 1).")


def l3_image_point(r):
    name = r.choice(list(TRANSFORMS))
    M = TRANSFORMS[name]
    p = [nz(r, -4, 4), nz(r, -4, 4)]
    return Exercise(f"M = {mat(M)} ({name}). Image du point {vec(p)} ?\nRéponse : x, y", Numbers(mv(M, p), ordered=True),
                    M_TRANSF + "\nCalcule M·p.", transform_square(M))


def l3_which_matrix(r):
    name = r.choice(list(TRANSFORMS))
    M = TRANSFORMS[name]
    return Exercise(f"Donne la matrice 2×2 de la transformation : {name}." + FLAT, Numbers(flat(M), ordered=True),
                    M_TRANSF + "\nCherche où vont (1, 0) et (0, 1), et écris-les en colonnes.", None)


def l3_rotation(r):
    th = r.choice([30, 45, 60, 90, 120, 180])
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    return Exercise(f"Donne la matrice de rotation d'angle {th}° (valeurs exactes avec sqrt ou décimales précises)." + FLAT,
                    Numbers([c, -s, s, c], ordered=True),
                    "R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]].", transform_square([[c, -s], [s, c]]))


def l3_area_factor(r):
    M = rm(r, 2, 2, -3, 4)
    return Exercise(f"La matrice M = {mat(M)} multiplie les aires par quel facteur ?", Number(abs(det2(M))),
                    "Le facteur d'aire est |det M|. (Le signe de det indique si l'orientation est inversée.)",
                    transform_square(M))


def l3_compose(r):
    names = r.sample(list(TRANSFORMS), 2)
    A, B = TRANSFORMS[names[0]], TRANSFORMS[names[1]]
    return Exercise(f"On applique d'abord « {names[0]} » (A = {mat(A)}), puis « {names[1]} » (B = {mat(B)}). "
                    "Matrice de la transformation totale ?" + FLAT, Numbers(flat(mul(B, A)), ordered=True),
                    "Composer = multiplier. Attention à l'ordre : appliquer A puis B, c'est B·A (on lit de droite à gauche, "
                    "comme f(g(x))).", show(("B", B), ("A", A), ops=["·"]))


def l3_det3(r):
    A = rm(r, 3, 3, -3, 4)
    return Exercise(f"Calcule le déterminant de A = {mat(A)}.", Number(round(np.linalg.det(np.array(A, dtype=float)))),
                    "Développe selon la 1re ligne : det = a·det(M₁₁) − b·det(M₁₂) + c·det(M₁₃), où Mᵢⱼ est la matrice 2×2 "
                    "obtenue en supprimant la ligne i et la colonne j. (Ou règle de Sarrus.)", show(("A", A)))


def l3_inv_diag(r):
    d = [r.choice([1, 2, 4, 5, -2, -1]) for _ in range(3)]
    return Exercise(f"Inverse de la matrice diagonale D = diag{vec(d)} ? Donne les 3 coefficients diagonaux de D⁻¹.",
                    Numbers([F(1, x) for x in d], ordered=True), "L'inverse d'une diagonale : on inverse chaque coefficient diagonal.",
                    show(("D", [[d[0], 0, 0], [0, d[1], 0], [0, 0, d[2]]])))


def l3_unit_square(r):
    M = rm(r, 2, 2, -3, 4)
    return Exercise(f"M = {mat(M)}. Image du coin (1, 1) du carré unité ?\nRéponse : x, y", Numbers(mv(M, [1, 1]), ordered=True),
                    M_TRANSF + "\nM·(1, 1) = somme des deux colonnes.", transform_square(M))


def l3_scale_matrix(r):
    a, b = nz(r, -4, 4), nz(r, -4, 4)
    p = [nz(r, -3, 3), nz(r, -3, 3)]
    return Exercise(f"La matrice diag({a}, {b}) étire x par {a} et y par {b}. Image de {vec(p)} ?\nRéponse : x, y",
                    Numbers([a * p[0], b * p[1]], ordered=True), "Une matrice diagonale agit indépendamment sur chaque axe.",
                    transform_square([[a, 0], [0, b]]))


def l3_det_product(r):
    A, B = rm(r, 2, 2, -3, 4), rm(r, 2, 2, -3, 4)
    return Exercise(f"det A = {det2(A)} et det B = {det2(B)}. Que vaut det(AB) ?", Number(det2(A) * det2(B)),
                    "det(AB) = det(A)·det(B) : les facteurs d'aire se multiplient.", None)


# ================================================================ NIVEAU 4

M_EIG = ("Valeurs propres : les λ tels que A·v = λ·v pour un vecteur v ≠ 0. On les trouve en résolvant det(A − λI) = 0.\n"
         "Pour 2×2 : λ² − (trace)·λ + det = 0.  Somme des λ = trace, produit des λ = det.")


def _eig_matrix(r, with_vectors=False):
    """A = P·diag(l1, l2)·P⁻¹ avec P entière de déterminant ±1 : A entière, valeurs propres l1, l2, vecteurs propres = colonnes de P."""
    l1, l2 = r.sample(range(-4, 6), 2)
    P = r.choice([[[1, 1], [0, 1]], [[1, 0], [1, 1]], [[2, 1], [1, 1]], [[1, 1], [1, 2]], [[1, -1], [1, 0]]])
    Pi = np.linalg.inv(np.array(P, dtype=float))
    A = np.rint(np.array(P) @ np.diag([l1, l2]) @ Pi).astype(int).tolist()
    if with_vectors:
        return A, (l1, [P[0][0], P[1][0]]), (l2, [P[0][1], P[1][1]])
    return A, l1, l2


def l4_eigvals(r):
    A, l1, l2 = _eig_matrix(r)
    return Exercise(f"Trouve les valeurs propres de A = {mat(A)}.", Numbers([l1, l2]), M_EIG, show(("A", A)))


def l4_eig_tri(r):
    A = [[nz(r, -6, 6), r.randint(-9, 9)], [0, nz(r, -6, 6)]]
    return Exercise(f"Valeurs propres de la matrice triangulaire A = {mat(A)} ?", Numbers([A[0][0], A[1][1]]),
                    M_EIG + "\nPour une matrice triangulaire, ce sont les coefficients de la diagonale.", show(("A", A)))


def l4_eigvec_check(r):
    A, *pairs = _eig_matrix(r, with_vectors=True)
    lam, v = r.choice(pairs)
    return Exercise(f"A = {mat(A)} et v = {vec(v)} est un vecteur propre. Quelle est sa valeur propre λ ?",
                    Number(lam), "Calcule A·v et compare à v : A·v = λ·v, donc λ = (A·v)ᵢ / vᵢ pour une composante non nulle.",
                    show(("A", A), ("v", col(v)), ops=["·"]))


def l4_trace_sum(r):
    A, l1, l2 = _eig_matrix(r)
    return Exercise(f"A = {mat(A)} a une valeur propre égale à {l1}. Quelle est l'autre ?", Number(l2),
                    M_EIG + "\nUtilise : somme des valeurs propres = trace.", show(("A", A)))


def l4_charpoly(r):
    A = rm(r, 2, 2, -4, 5)
    x = sp.Symbol("x")
    p = sp.expand((A[0][0] - x) * (A[1][1] - x) - A[0][1] * A[1][0])
    return Exercise(f"Polynôme caractéristique de A = {mat(A)} : det(A − x·I) (une expression en x).", Expr(p),
                    "A − xI = [[a − x, b], [c, d − x]]. Son déterminant : (a − x)(d − x) − bc.", show(("A", A)))


def l4_rank(r):
    kind = r.choice([1, 2, 3])
    u = [nz(r, -3, 3) for _ in range(3)]
    v = [r.randint(-3, 3) for _ in range(3)]
    if kind == 1:
        A = [u, [2 * c for c in u], [-c for c in u]]
    elif kind == 2:
        A = [u, v, [a + b for a, b in zip(u, v)]]
    else:
        A = rm(r, 3, 3, -3, 4)
    rank = int(np.linalg.matrix_rank(np.array(A, dtype=float)))
    return Exercise(f"Quel est le rang de A = {mat(A)} ?", Number(rank),
                    "Rang = nombre de lignes (ou colonnes) linéairement indépendantes. Cherche des lignes multiples ou "
                    "sommes d'autres lignes ; si det ≠ 0 (3×3), le rang vaut 3.", show(("A", A)))


def l4_gauss(r):
    while True:
        A = rm(r, 3, 3, -2, 3)
        if abs(round(np.linalg.det(np.array(A, dtype=float)))) == 1:
            break
    x = [r.randint(-3, 3) for _ in range(3)]
    b = mv(A, x)
    return Exercise(f"Résous A·x = b avec A = {mat(A)}, b = {vec(b)}.\nRéponse : x₁, x₂, x₃", Numbers(x, ordered=True),
                    "Pivot de Gauss : combine les lignes pour éliminer x₁ des lignes 2 et 3, puis x₂ de la ligne 3. "
                    "Remonte ensuite (substitution).", show(("A", A), ("b", col(b))))


def l4_kernel_dim(r):
    A = [[1, 2, 3], [2, 4, 6]] if r.random() < 0.5 else rm(r, 2, 3, -3, 4)
    rank = int(np.linalg.matrix_rank(np.array(A, dtype=float)))
    return Exercise(f"A = {mat(A)} (2×3). Dimension du noyau (ensemble des x tels que A·x = 0) ?", Number(3 - rank),
                    "Théorème du rang : dim(noyau) = nombre de colonnes − rang.", show(("A", A)))


def l4_symmetric(r):
    a, b, c = r.randint(-5, 5), r.randint(-5, 5), r.randint(-5, 5)
    sym = r.random() < 0.5
    A = [[a, b], [b if sym else b + nz(r, -3, 3), c]]
    return Exercise(f"A = {mat(A)} est-elle symétrique ? (oui / non)", Choice("oui" if A[0][1] == A[1][0] else "non", ["oui", "non"]),
                    "Symétrique ⇔ A = Aᵀ ⇔ aᵢⱼ = aⱼᵢ. (Les matrices de covariance et les Hessiennes sont symétriques.)",
                    show(("A", A)))


def l4_det_eig(r):
    A, l1, l2 = _eig_matrix(r)
    return Exercise(f"Les valeurs propres de A sont {l1} et {l2}. Que vaut det A ?", Number(l1 * l2),
                    "Le déterminant est le produit des valeurs propres.", None)


# ================================================================ NIVEAU 5

def l5_dense(r):
    W, x = rm(r, 2, 3, -2, 3), [r.randint(-2, 2) for _ in range(3)]
    b = [r.randint(-3, 3) for _ in range(2)]
    z = [a + c for a, c in zip(mv(W, x), b)]
    return Exercise(f"Couche dense : z = W·x + b avec W = {mat(W)}, x = {vec(x)}, b = {vec(b)}. Calcule z.\nRéponse : 2 composantes.",
                    Numbers(z, ordered=True), "Produit matrice-vecteur, puis ajoute le biais composante par composante.",
                    show(("W", W), ("x", col(x)), ("b", col(b)), ops=["·", "+"]))


def l5_dense_relu(r):
    W, x = rm(r, 3, 2, -3, 3), [r.randint(-2, 2) for _ in range(2)]
    z = mv(W, x)
    return Exercise(f"a = ReLU(W·x) avec W = {mat(W)}, x = {vec(x)}. Calcule a.\nRéponse : 3 composantes.",
                    Numbers([max(0, v) for v in z], ordered=True), "Calcule W·x, puis applique ReLU(u) = max(0, u) à chaque composante.",
                    show(("W", W), ("x", col(x)), ops=["·"]))


def l5_batch_shape(r):
    n, d, h = r.choice([32, 64, 128]), r.choice([10, 784, 20]), r.choice([16, 64, 128])
    return Exercise(f"Un batch X de taille {n}×{d} ({n} exemples, {d} features) passe dans une couche W de taille {d}×{h}. "
                    "Taille de X·W ?\nRéponse : lignes, colonnes", Numbers([n, h], ordered=True), M_PROD, None)


def l5_params(r):
    i, o = r.choice([3, 10, 64, 784]), r.choice([2, 10, 32, 128])
    return Exercise(f"Combien de paramètres (poids + biais) dans une couche dense de {i} entrées vers {o} sorties ?",
                    Number(i * o + o), "W a une taille sorties × entrées, et il y a un biais par sortie.", None)


def l5_grad_W(r):
    d = [r.randint(-3, 3) for _ in range(2)]
    x = [r.randint(-3, 3) for _ in range(3)]
    G = [[a * b for b in x] for a in d]
    return Exercise(f"Pour z = W·x, si ∂L/∂z = δ = {vec(d)} et x = {vec(x)}, alors ∂L/∂W = δ·xᵀ (produit extérieur). "
                    "Calcule-le (2×3)." + FLAT, Numbers(flat(G), ordered=True),
                    "Produit extérieur : le coefficient (i, j) vaut δᵢ·xⱼ. Chaque poids Wᵢⱼ relie l'entrée j à la sortie i.",
                    show(("δ", col(d)), ("xᵀ", [x]), ops=["·"]))


def l5_backprop_shape(r):
    o, h = r.choice([2, 10, 3]), r.choice([4, 16, 64])
    return Exercise(f"Rétropropagation : W est {o}×{h} et δ (gradient en sortie) a {o} composantes. "
                    "Quelle est la taille de Wᵀ·δ (gradient envoyé à la couche précédente) ?\nRéponse : lignes, colonnes",
                    Numbers([h, 1], ordered=True), "Wᵀ est " + f"{h}×{o}" + " ; δ est un vecteur colonne " + f"{o}×1" + ".", None)


def l5_frobenius(r):
    A = r.choice([[[1, 2], [2, 4]], [[2, 1], [2, 4]], [[3, 0], [0, 4]], [[1, 1], [1, 1]], [[2, 2], [1, 4]]])
    return Exercise(f"Norme de Frobenius de A = {mat(A)} ? (racine de la somme des carrés de tous les coefficients)",
                    Number(float(np.sqrt(sum(c * c for c in flat(A))))), "‖A‖_F = √(Σ aᵢⱼ²).", show(("A", A)))


def l5_power_iter(r):
    A = r.choice([[[2, 1], [1, 2]], [[3, 1], [1, 3]], [[4, 1], [2, 3]]])
    v = [1, 0]
    w = mv(A, v)
    return Exercise(f"Méthode de la puissance : partant de v = (1, 0), calcule A·v pour A = {mat(A)}.\nRéponse : 2 composantes.",
                    Numbers(w, ordered=True),
                    "On multiplie v par A encore et encore (en normalisant) : v s'aligne sur le vecteur propre dominant. "
                    "Ici, une seule multiplication.", transform_square(A))


def l5_pca_var(r):
    l1, l2 = sorted(r.sample([1, 2, 3, 4, 6, 9, 12], 2), reverse=True)
    return Exercise(f"La matrice de covariance de données 2D a pour valeurs propres {l1} et {l2}. "
                    "Quelle proportion de la variance garde-t-on en projetant sur la 1re composante principale ?",
                    Number(F(l1, l1 + l2)), "ACP : chaque valeur propre = variance le long de son axe. Proportion = λ₁ / (λ₁ + λ₂).",
                    None)


def l5_cov(r):
    xs = [r.randint(0, 6) for _ in range(4)]
    ys = [r.randint(0, 6) for _ in range(4)]
    mx, my = F(sum(xs), 4), F(sum(ys), 4)
    vx = sum((a - mx) ** 2 for a in xs) / 4
    vy = sum((b - my) ** 2 for b in ys) / 4
    cxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / 4
    return Exercise(f"Données (x, y) : {list(zip(xs, ys))}. Matrice de covariance (division par n = 4) ?" + FLAT,
                    Numbers([vx, cxy, cxy, vy], ordered=True),
                    "[[Var(x), Cov(x, y)], [Cov(y, x), Var(y)]]. Var(x) = moyenne des (xᵢ − x̄)², Cov = moyenne des "
                    "(xᵢ − x̄)(yᵢ − ȳ).", None)


LEVELS = {
    1: [l1_add, l1_scale, l1_transpose, l1_entry, l1_dims, l1_matvec, l1_identity, l1_sub, l1_trace, l1_matvec3],
    2: [l2_prod, l2_prod_rect, l2_matvec3, l2_det, l2_inverse, l2_invertible, l2_square, l2_transpose_prod, l2_solve,
        l2_det_diag],
    3: [l3_image_point, l3_which_matrix, l3_rotation, l3_area_factor, l3_compose, l3_det3, l3_inv_diag, l3_unit_square,
        l3_scale_matrix, l3_det_product],
    4: [l4_eigvals, l4_eig_tri, l4_eigvec_check, l4_trace_sum, l4_charpoly, l4_rank, l4_gauss, l4_kernel_dim,
        l4_symmetric, l4_det_eig],
    5: [l5_dense, l5_dense_relu, l5_batch_shape, l5_params, l5_grad_W, l5_backprop_shape, l5_frobenius, l5_power_iter,
        l5_pca_var, l5_cov],
}
