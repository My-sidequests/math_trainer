"""Statistiques : indicateurs, dispersion, corrélation et régression, métriques de ML, inférence."""

from __future__ import annotations

import math
from fractions import Fraction as F

import matplotlib.patches

from ..answers import Choice, Number, Numbers
from ..core import Exercise, bars, coef, scatter, signed

DESCRIPTIONS = {
    1: "Moyenne, médiane, mode, étendue",
    2: "Variance, écart-type, quartiles, z-score",
    3: "Covariance, corrélation, régression linéaire (moindres carrés)",
    4: "Normalisation et métriques de ML (MSE, précision, rappel, F1...)",
    5: "Intervalles de confiance, moyennes mobiles, batch norm",
}

M_CENTER = ("Moyenne = somme / nombre de valeurs.\n"
            "Médiane = valeur du milieu une fois les données TRIÉES (si nombre pair : moyenne des deux du milieu).\n"
            "Mode = valeur la plus fréquente. Étendue = max − min.")
M_SPREAD = ("Variance (population) = moyenne des carrés des écarts à la moyenne : Var = (1/n)·Σ(xᵢ − x̄)².\n"
            "Écart-type σ = √Var. Variance d'échantillon (non biaisée) : on divise par n − 1 au lieu de n.")
M_REG = ("Covariance : Cov(x, y) = (1/n)·Σ(xᵢ − x̄)(yᵢ − ȳ).  Corrélation : r = Cov(x, y) / (σₓ·σᵧ), entre −1 et 1.\n"
         "Droite des moindres carrés y = a·x + b : a = Cov(x, y) / Var(x), b = ȳ − a·x̄.")
M_CLS = ("Matrice de confusion : VP (vrais positifs), FP (faux positifs), FN (faux négatifs), VN (vrais négatifs).\n"
         "Exactitude = (VP + VN) / total ; Précision = VP / (VP + FP) ; Rappel = VP / (VP + FN) ;\n"
         "F1 = 2·P·R / (P + R) ; Spécificité = VN / (VN + FP).")
CALC = "\n(calculatrice autorisée ; 3 décimales)"


def data(r, n=None, lo=0, hi=20):
    return [r.randint(lo, hi) for _ in range(n or r.randint(5, 8))]


def dbars(xs, title="données"):
    return bars([str(i + 1) for i in range(len(xs))], xs, title)


def mean(xs):
    return F(sum(xs), len(xs))


def var(xs, ddof=0):
    m = mean(xs)
    return sum((F(x) - m) ** 2 for x in xs) / (len(xs) - ddof)


# ================================================================ NIVEAU 1

def l1_mean(r):
    xs = data(r)
    return Exercise(f"Moyenne de {xs} ?", Number(mean(xs)), M_CENTER, dbars(xs))


def l1_median_odd(r):
    xs = data(r, r.choice([5, 7]))
    return Exercise(f"Médiane de {xs} ?", Number(sorted(xs)[len(xs) // 2]), M_CENTER, dbars(xs))


def l1_median_even(r):
    xs = data(r, r.choice([4, 6, 8]))
    s = sorted(xs)
    n = len(s)
    return Exercise(f"Médiane de {xs} ?", Number(F(s[n // 2 - 1] + s[n // 2], 2)), M_CENTER, dbars(xs))


def l1_mode(r):
    m = r.randint(1, 9)
    xs = [m, m, m] + r.sample([v for v in range(1, 15) if v != m], 4)
    r.shuffle(xs)
    return Exercise(f"Mode de {xs} ?", Number(m), M_CENTER, dbars(xs))


def l1_range(r):
    xs = data(r, lo=-10)
    return Exercise(f"Étendue de {xs} ?", Number(max(xs) - min(xs)), M_CENTER, dbars(xs))


def l1_weighted_table(r):
    vals = sorted(r.sample(range(0, 21), 3))
    cnt = [r.randint(1, 6) for _ in range(3)]
    return Exercise(f"Notes : {vals[0]} (×{cnt[0]} élèves), {vals[1]} (×{cnt[1]}), {vals[2]} (×{cnt[2]}). Moyenne de la classe ?",
                    Number(F(sum(v * c for v, c in zip(vals, cnt)), sum(cnt))),
                    "Chaque valeur compte autant de fois que son effectif : Σ(valeur × effectif) / Σ effectifs.",
                    bars([str(v) for v in vals], cnt, "effectifs"))


def l1_missing(r):
    xs = data(r, 4)
    target = r.randint(-(-sum(xs) // 5), (sum(xs) + 20) // 5)  # garantit une 5e note entre 0 et 20
    x = target * 5 - sum(xs)
    return Exercise(f"Quatre notes : {xs}. Quelle 5e note faut-il pour obtenir une moyenne de {target} ?", Number(x),
                    "Moyenne × nombre = somme totale nécessaire. Soustrais la somme déjà obtenue.", dbars(xs))


def l1_percent(r):
    part, tot = r.randint(1, 50), r.choice([50, 80, 200, 250, 400])
    return Exercise(f"{part} élèves sur {tot} ont réussi. Pourcentage de réussite ?", Number(F(part * 100, tot)),
                    "Pourcentage = part / total × 100.", None)


def l1_minmax(r):
    xs = data(r, lo=-20, hi=40)
    return Exercise(f"Données : {xs}. Donne le minimum puis le maximum.\nRéponse : min, max", Numbers([min(xs), max(xs)], ordered=True),
                    "Parcours les valeurs en gardant la plus petite et la plus grande.", dbars(xs))


def l1_mean_shift(r):
    xs = data(r)
    k = r.randint(-5, 10)
    return Exercise(f"La moyenne de {xs} vaut {float(mean(xs)):g}. Si on ajoute {k} à chaque valeur, que devient la moyenne ?",
                    Number(mean(xs) + k), "Ajouter une constante à toutes les valeurs décale la moyenne de cette constante.", None)


# ================================================================ NIVEAU 2

def _nice(r):
    """Données à moyenne entière."""
    while True:
        xs = data(r, r.choice([4, 5]), 0, 12)
        if sum(xs) % len(xs) == 0:
            return xs


def l2_var(r):
    xs = _nice(r)
    return Exercise(f"Variance (population, division par n) de {xs} ?", Number(var(xs)), M_SPREAD, dbars(xs))


def l2_std(r):
    xs = r.choice([[2, 4, 4, 4, 5, 5, 7, 9], [1, 3, 5, 7], [10, 10, 10, 10], [0, 6], [1, 1, 7, 7], [2, 8, 2, 8]])
    return Exercise(f"Écart-type (population) de {xs} ?", Number(math.sqrt(var(xs))), M_SPREAD, dbars(xs))


def l2_q1(r):
    xs = sorted(data(r, 7))
    r.shuffle(xs)
    s = sorted(xs)
    return Exercise(f"Données (7 valeurs) : {xs}. Premier quartile Q1, défini comme la médiane des 3 plus petites valeurs ?",
                    Number(s[1]), "Trie les données. Q1 = médiane de la moitié basse, Q3 = médiane de la moitié haute.", dbars(xs))


def l2_iqr(r):
    xs = data(r, 7)
    s = sorted(xs)
    return Exercise(f"Données (7 valeurs) : {xs}. Écart interquartile Q3 − Q1 (Q1 = 2e valeur triée, Q3 = 6e valeur triée) ?",
                    Number(s[5] - s[1]), "Trie, repère Q1 et Q3, soustrais. L'IQR mesure la dispersion des 50 % centraux.",
                    dbars(xs))


def l2_z(r):
    mu, s = r.choice([(50, 10), (12, 3), (100, 15), (0, 2)])
    x = mu + s * r.choice([-2, -1, 0.5, 1, 2.5])
    return Exercise(f"Moyenne {mu}, écart-type {s}. z-score de la valeur {x:g} ?", Number((x - mu) / s),
                    "z = (x − moyenne) / écart-type.", None)


def l2_weighted(r):
    notes = [r.randint(5, 20) for _ in range(3)]
    coefs = [r.randint(1, 4) for _ in range(3)]
    return Exercise(f"Notes {notes} avec coefficients {coefs}. Moyenne pondérée ?",
                    Number(F(sum(n * c for n, c in zip(notes, coefs)), sum(coefs))),
                    "Σ(note × coef) / Σ coefs.", bars(["1", "2", "3"], notes))


def l2_scale_std(r):
    s, k = r.randint(1, 9), r.choice([2, 3, -2, 10])
    return Exercise(f"Des données ont un écart-type {s}. On multiplie toutes les valeurs par {k}. Nouvel écart-type ?",
                    Number(abs(k) * s), "Multiplier par k multiplie l'écart-type par |k| (et la variance par k²).", None)


def l2_sample_var(r):
    xs = _nice(r)
    return Exercise(f"Variance d'ÉCHANTILLON (division par n − 1) de {xs} ?", Number(var(xs, 1)),
                    M_SPREAD + "\nMême somme des carrés des écarts, mais divisée par n − 1.", dbars(xs))


def l2_var_shift(r):
    v, k = r.randint(1, 9), r.randint(-10, 10)
    return Exercise(f"Des données ont une variance {v}. On ajoute {k} à chaque valeur. Nouvelle variance ?", Number(v),
                    "Décaler toutes les valeurs ne change pas leur dispersion.", None)


def l2_cv(r):
    m, s = r.choice([(50, 5), (20, 4), (200, 30), (10, 1)])
    return Exercise(f"Moyenne {m}, écart-type {s}. Coefficient de variation σ / moyenne ?", Number(F(s, m)),
                    "CV = σ / moyenne : une dispersion relative, sans unité.", None)


# ================================================================ NIVEAU 3

def _xy(r, n=4):
    xs = r.sample(range(0, 8), n)
    a, b = r.choice([1, 2, -1, 3]), r.randint(-3, 5)
    ys = [a * x + b + r.choice([-1, 0, 0, 1]) for x in xs]
    return xs, ys


def cov(xs, ys):
    mx, my = mean(xs), mean(ys)
    return sum((F(x) - mx) * (F(y) - my) for x, y in zip(xs, ys)) / len(xs)


def l3_cov(r):
    xs, ys = _xy(r)
    return Exercise(f"x = {xs}, y = {ys}. Covariance (division par n) ?", Number(cov(xs, ys)), M_REG, scatter(xs, ys))


def l3_corr(r):
    xs, ys = _xy(r)
    while var(ys) == 0:
        xs, ys = _xy(r)
    return Exercise(f"x = {xs}, y = {ys}. Coefficient de corrélation r ?" + CALC,
                    Number(float(cov(xs, ys)) / math.sqrt(var(xs) * var(ys)), 5e-3), M_REG, scatter(xs, ys))


def l3_slope(r):
    xs, ys = _xy(r)
    return Exercise(f"x = {xs}, y = {ys}. Pente a de la droite des moindres carrés ?", Number(cov(xs, ys) / var(xs)),
                    M_REG, scatter(xs, ys))


def l3_intercept(r):
    xs, ys = _xy(r)
    a = cov(xs, ys) / var(xs)
    return Exercise(f"x = {xs}, y = {ys}. La pente des moindres carrés vaut a = {a}. Ordonnée à l'origine b ?",
                    Number(mean(ys) - a * mean(xs)), M_REG, scatter(xs, ys))


def l3_predict(r):
    a, b = r.choice([1.5, -2, 0.5, 3]), r.randint(-5, 10)
    x = r.randint(0, 10)
    return Exercise(f"Une régression donne ŷ = {coef(a)}x {signed(b)}. Prédiction pour x = {x} ?", Number(a * x + b),
                    "Remplace x dans l'équation de la droite.", None)


def l3_residual(r):
    a, b = r.choice([2, -1, 3]), r.randint(-5, 5)
    x, y = r.randint(0, 8), r.randint(-5, 20)
    return Exercise(f"Modèle ŷ = {coef(a)}x {signed(b)}. Point observé ({x}, {y}). Résidu y − ŷ ?", Number(y - (a * x + b)),
                    "Résidu = valeur observée − valeur prédite.", None)


def l3_r2(r):
    sse, sst = r.randint(5, 40), r.choice([50, 80, 100, 200])
    return Exercise(f"Somme des carrés des résidus SSE = {sse}, variabilité totale SST = {sst}. Coefficient R² = 1 − SSE/SST ?",
                    Number(1 - F(sse, sst)), "R² = part de la variance expliquée par le modèle (1 = parfait, 0 = pas mieux que la moyenne).",
                    None)


def l3_sign(r):
    s = r.choice([1, -1, 0])
    xs = list(range(8))
    if s:
        ys = [s * 2 * x + r.choice([-1, 0, 1]) for x in xs]
    else:
        ys = r.choice([[4, 6, 4, 6, 6, 4, 6, 4], [5, 1, 9, 5, 5, 9, 1, 5], [2, 8, 5, 5, 5, 5, 8, 2]])
    kind = {1: "positive", -1: "negative", 0: "nulle"}[s]
    return Exercise("Le nuage de points montre-t-il une corrélation positive, négative ou nulle (≈ 0) ? "
                    "(positive / négative / nulle)", Choice(kind, ["positive", "négative", "nulle"]),
                    "Positive : y augmente avec x. Négative : y diminue quand x augmente. Nulle : pas de tendance.",
                    scatter(xs, ys))


def l3_standardize(r):
    xs = _nice(r)
    while var(xs) == 0:
        xs = _nice(r)
    i = r.randint(0, len(xs) - 1)
    return Exercise(f"Données {xs}. Standardise la valeur {xs[i]} : (x − moyenne) / écart-type (population)." + CALC,
                    Number((xs[i] - float(mean(xs))) / math.sqrt(var(xs)), 5e-3), M_SPREAD, dbars(xs))


def l3_mean_z(r):
    xs = data(r)
    return Exercise(f"On standardise les données {xs} (z-scores). Quelle est la moyenne des z-scores obtenus ?", Number(0),
                    "Les z-scores sont centrés par construction : on a retiré la moyenne. (Et leur écart-type vaut 1.)", None)


# ================================================================ NIVEAU 4

def l4_minmax_norm(r):
    xs = data(r, 5, 0, 50)
    while max(xs) == min(xs):
        xs = data(r, 5, 0, 50)
    i = r.randint(0, 4)
    return Exercise(f"Normalisation min-max de {xs} vers [0, 1]. Nouvelle valeur de {xs[i]} ?",
                    Number(F(xs[i] - min(xs), max(xs) - min(xs))), "x' = (x − min) / (max − min).", dbars(xs))


def l4_mse(r):
    y, p = data(r, 4, 0, 6), data(r, 4, 0, 6)
    return Exercise(f"Cibles y = {y}, prédictions ŷ = {p}. Erreur quadratique moyenne (MSE) ?",
                    Number(F(sum((a - b) ** 2 for a, b in zip(y, p)), 4)), "MSE = (1/n)·Σ(ŷᵢ − yᵢ)².",
                    bars(["1", "2", "3", "4"], [b - a for a, b in zip(y, p)], "erreurs"))


def l4_mae(r):
    y, p = data(r, 4, 0, 9), data(r, 4, 0, 9)
    return Exercise(f"Cibles y = {y}, prédictions ŷ = {p}. Erreur absolue moyenne (MAE) ?",
                    Number(F(sum(abs(a - b) for a, b in zip(y, p)), 4)), "MAE = (1/n)·Σ|ŷᵢ − yᵢ|.",
                    bars(["1", "2", "3", "4"], [b - a for a, b in zip(y, p)], "erreurs"))


def l4_rmse(r):
    y = data(r, 4, 0, 9)
    p = [a + e for a, e in zip(y, r.choice([[1, -1, 1, -1], [2, 0, 0, 0], [2, 2, -2, 2], [3, 0, -3, 0]]))]
    return Exercise(f"Cibles y = {y}, prédictions ŷ = {p}. RMSE = √MSE ?",
                    Number(math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / 4)), "Calcule la MSE, puis sa racine.", None)


def _conf(r):
    return r.randint(10, 60), r.randint(1, 20), r.randint(1, 20), r.randint(10, 60)


def _conf_draw(vp, fp, fn, vn):
    """Matrice de confusion : lignes = prédiction, colonnes = réalité."""

    def draw(ax):
        cells = [[(vp, "VP", "#c6f6d5"), (fp, "FP", "#fed7d7")], [(fn, "FN", "#fed7d7"), (vn, "VN", "#c6f6d5")]]
        for i, row in enumerate(cells):
            for j, (val, name, col) in enumerate(row):
                ax.add_patch(matplotlib.patches.Rectangle((j, 1 - i), 1, 1, fc=col, ec="white", lw=3))
                ax.text(j + 0.5, 1.5 - i + 0.12, str(val), ha="center", va="center", fontsize=20, fontweight="bold")
                ax.text(j + 0.5, 1.5 - i - 0.2, name, ha="center", va="center", fontsize=12, color="#4a5568")
        for j, t in enumerate(["réel +", "réel −"]):
            ax.text(j + 0.5, 2.15, t, ha="center", fontsize=12)
        for i, t in enumerate(["prédit +", "prédit −"]):
            ax.text(-0.1, 1.5 - i, t, ha="right", va="center", fontsize=12)
        ax.set_xlim(-0.8, 2.2)
        ax.set_ylim(-0.2, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

    return draw


def l4_accuracy(r):
    vp, fp, fn, vn = _conf(r)
    return Exercise(f"VP = {vp}, FP = {fp}, FN = {fn}, VN = {vn}. Exactitude (accuracy) ?", Number(F(vp + vn, vp + fp + fn + vn)),
                    M_CLS, _conf_draw(vp, fp, fn, vn))


def l4_precision(r):
    vp, fp, fn, vn = _conf(r)
    return Exercise(f"VP = {vp}, FP = {fp}, FN = {fn}, VN = {vn}. Précision ?", Number(F(vp, vp + fp)), M_CLS,
                    _conf_draw(vp, fp, fn, vn))


def l4_recall(r):
    vp, fp, fn, vn = _conf(r)
    return Exercise(f"VP = {vp}, FP = {fp}, FN = {fn}, VN = {vn}. Rappel (recall) ?", Number(F(vp, vp + fn)), M_CLS,
                    _conf_draw(vp, fp, fn, vn))


def l4_f1(r):
    p, rc = r.choice([(F(1, 2), F(1, 2)), (F(2, 3), F(1, 3)), (F(4, 5), F(2, 5)), (F(3, 4), F(1, 2))])
    return Exercise(f"Précision = {p}, rappel = {rc}. Score F1 ?", Number(2 * p * rc / (p + rc)), M_CLS + "\nF1 est la moyenne harmonique de P et R.",
                    None)


def l4_specificity(r):
    vp, fp, fn, vn = _conf(r)
    return Exercise(f"VP = {vp}, FP = {fp}, FN = {fn}, VN = {vn}. Spécificité ?", Number(F(vn, vn + fp)), M_CLS,
                    _conf_draw(vp, fp, fn, vn))


def l4_standardize_new(r):
    mu, s = r.choice([(10, 2), (50, 5), (3, 0.5), (100, 20)])
    x = r.randint(int(mu - 3 * s), int(mu + 3 * s))
    return Exercise(f"Un scaler a appris moyenne = {mu} et écart-type = {s} sur l'entraînement. Transforme la nouvelle "
                    f"valeur {x}.", Number((x - mu) / s),
                    "On applique (x − μ)/σ avec les statistiques de l'ENTRAÎNEMENT (jamais recalculées sur le test).", None)


# ================================================================ NIVEAU 5

def l5_ci(r):
    m, s, n = r.choice([(50, 10, 100), (20, 4, 64), (100, 15, 225), (12, 3, 36)])
    h = 1.96 * s / math.sqrt(n)
    return Exercise(f"Moyenne d'échantillon {m}, écart-type {s}, n = {n}. Intervalle de confiance à 95 % : "
                    "x̄ ± 1,96·σ/√n.\nRéponse : borne basse, borne haute", Numbers([m - h, m + h], ordered=True, tol=1e-3),
                    "Calcule l'erreur standard σ/√n, multiplie par 1,96, puis ajoute/retire à la moyenne.", None)


def l5_se(r):
    s, n = r.choice([(10, 25), (6, 36), (20, 100), (9, 9)])
    return Exercise(f"Écart-type {s}, taille d'échantillon n = {n}. Erreur standard de la moyenne σ/√n ?", Number(s / math.sqrt(n)),
                    "L'erreur standard mesure l'incertitude sur la MOYENNE : elle diminue comme 1/√n.", None)


def l5_split(r):
    n, pct = r.choice([1000, 5000, 800, 1200]), r.choice([20, 25, 10, 30])
    return Exercise(f"{n} exemples, on en garde {pct} % pour le test. Combien d'exemples d'entraînement ?",
                    Number(n * (100 - pct) // 100), "Entraînement = total × (1 − part de test).", None)


def l5_kfold(r):
    n, k = r.choice([(100, 5), (1000, 10), (600, 3), (500, 5)])
    return Exercise(f"Validation croisée {k}-fold sur {n} exemples. Combien d'exemples d'entraînement à chaque tour ?",
                    Number(n - n // k), "On découpe en k parts : 1 part pour valider, k − 1 pour entraîner.", None)


def l5_bias_var(r):
    b, v, noise = r.randint(1, 5), r.randint(1, 9), r.randint(0, 3)
    return Exercise(f"Biais = {b}, variance = {v}, bruit irréductible = {noise}. Erreur quadratique attendue = biais² + variance + bruit ?",
                    Number(b * b + v + noise), "Décomposition biais-variance : attention, c'est le biais AU CARRÉ.", None)


def l5_moving_avg(r):
    xs = data(r, 6, 0, 20)
    k = 3
    return Exercise(f"Série {xs}. Moyenne mobile sur {k} points de la dernière fenêtre (les {k} dernières valeurs) ?",
                    Number(F(sum(xs[-k:]), k)), "Moyenne des k dernières valeurs.", dbars(xs))


def l5_ema(r):
    beta = r.choice([0.9, 0.5, 0.8])
    v, g = r.randint(0, 10), r.randint(0, 10)
    return Exercise(f"Moyenne mobile exponentielle (utilisée par le momentum et Adam) : v ← β·v + (1 − β)·g. "
                    f"Avec β = {beta}, v = {v}, g = {g}, nouvelle valeur de v ?", Number(beta * v + (1 - beta) * g),
                    "Remplace dans la formule : l'ancienne moyenne garde un poids β, la nouvelle valeur un poids (1 − β).", None)


def l5_batchnorm(r):
    xs = r.choice([[1, 3, 5, 7], [2, 4, 4, 6], [0, 2, 4, 6], [10, 20, 30, 40]])
    i = r.randint(0, 3)
    eps = 0
    return Exercise(f"Batch norm (sans ε, sans γ ni β) sur le batch {xs} : valeur normalisée de {xs[i]} ?" + CALC,
                    Number((xs[i] - float(mean(xs))) / math.sqrt(var(xs) + eps), 5e-3),
                    "x̂ = (x − μ_B) / √(σ²_B + ε) avec la moyenne et la variance (population) du batch.", dbars(xs))


def l5_percentile(r):
    xs = sorted(r.sample(range(1, 100), 10))
    k = r.choice([10, 20, 50, 90])
    return Exercise(f"Valeurs triées {xs}. Avec la définition « le p-ième percentile est la valeur de rang p·n/100 », "
                    f"que vaut le {k}e percentile ?", Number(xs[k * 10 // 100 - 1]),
                    "Rang = p·n/100 (ici n = 10). Compte à partir de 1 dans la liste triée.", dbars(xs))


def l5_pooled_mean(r):
    n1, m1, n2, m2 = r.randint(10, 50), r.randint(5, 15), r.randint(10, 50), r.randint(5, 15)
    return Exercise(f"Groupe A : {n1} personnes, moyenne {m1}. Groupe B : {n2} personnes, moyenne {m2}. Moyenne globale ?",
                    Number(F(n1 * m1 + n2 * m2, n1 + n2)), "Moyenne pondérée par les effectifs (pas la moyenne des deux moyennes).",
                    None)


LEVELS = {
    1: [l1_mean, l1_median_odd, l1_median_even, l1_mode, l1_range, l1_weighted_table, l1_missing, l1_percent, l1_minmax,
        l1_mean_shift],
    2: [l2_var, l2_std, l2_q1, l2_iqr, l2_z, l2_weighted, l2_scale_std, l2_sample_var, l2_var_shift, l2_cv],
    3: [l3_cov, l3_corr, l3_slope, l3_intercept, l3_predict, l3_residual, l3_r2, l3_sign, l3_standardize, l3_mean_z],
    4: [l4_minmax_norm, l4_mse, l4_mae, l4_rmse, l4_accuracy, l4_precision, l4_recall, l4_f1, l4_specificity,
        l4_standardize_new],
    5: [l5_ci, l5_se, l5_split, l5_kfold, l5_bias_var, l5_moving_avg, l5_ema, l5_batchnorm, l5_percentile, l5_pooled_mean],
}
