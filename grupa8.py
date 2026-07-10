"""Grupa 8 — latentna struktura (Loto 7/39)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4648_k55.csv"
N_COMP = 5

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def presence_matrix(draws: np.ndarray) -> np.ndarray:
    x = np.zeros((len(draws), FRONT_N), dtype=float)
    for i, draw in enumerate(draws):
        for n in draw.tolist():
            x[i, n - 1] = 1.0
    return x


def pca_svd(draws: np.ndarray, n_comp: int = N_COMP) -> dict:
    """PCA preko SVD na centriranoj presence matrici (ceo CSV)."""
    x = presence_matrix(draws)
    xc = x - x.mean(axis=0)
    # economy SVD
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(n_comp, len(s))
    loadings = vt[:k].T  # 39 × k
    explained = (s[:k] ** 2) / (s ** 2).sum()
    # top |loading| po komponenti
    top_load = []
    for c in range(k):
        idx = np.argsort(-np.abs(loadings[:, c]))[:8]
        top_load.append([(int(i + 1), float(loadings[i, c])) for i in idx])
    return {
        "singular_values": [float(v) for v in s[:k]],
        "explained_var_ratio": [float(v) for v in explained],
        "top_loadings": top_load,
        "loadings": loadings,
        "scores": u[:, :k] * s[:k],
    }


def nmf_presence(draws: np.ndarray, n_comp: int = N_COMP, n_iter: int = 80) -> dict:
    """NMF: X ≈ W H (multiplikativno ažuriranje), X = presence."""
    rng = np.random.default_rng(SEED)
    x = presence_matrix(draws)
    t, n = x.shape
    k = min(n_comp, n)
    w = rng.random((t, k)) + 1e-4
    h = rng.random((k, n)) + 1e-4
    for _ in range(n_iter):
        wh = w @ h + 1e-12
        h *= (w.T @ x) / (w.T @ wh + 1e-12)
        wh = w @ h + 1e-12
        w *= (x @ h.T) / (wh @ h.T + 1e-12)
    # top brojevi po faktoru
    top = []
    for c in range(k):
        idx = np.argsort(-h[c])[:8]
        top.append([(int(i + 1), float(h[c, i])) for i in idx])
    recon_err = float(np.linalg.norm(x - w @ h) / np.linalg.norm(x))
    return {"H_top": top, "recon_rel_err": recon_err, "W": w, "H": h}


def ica_fast_proxy(draws: np.ndarray, n_comp: int = N_COMP, n_iter: int = 40) -> dict:
    """
    FastICA-lite na PCA skorovima (whitened):
    jedan non-Gaussian izvor po komponenti (tanh).
    """
    pca = pca_svd(draws, n_comp=n_comp)
    z = pca["scores"]  # T × k already scaled by s
    # whiten columns
    z = z - z.mean(axis=0)
    std = z.std(axis=0) + 1e-12
    z = z / std
    t, k = z.shape
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=(k, k))
    # orthonormalize
    q, _ = np.linalg.qr(w)
    w = q
    for _ in range(n_iter):
        for i in range(k):
            wi = w[:, i]
            wx = z @ wi
            g = np.tanh(wx)
            g_p = 1.0 - g ** 2
            wi_new = (z.T @ g) / t - g_p.mean() * wi
            # decorrelate
            for j in range(i):
                wi_new -= np.dot(wi_new, w[:, j]) * w[:, j]
            nrm = np.linalg.norm(wi_new) + 1e-12
            w[:, i] = wi_new / nrm
    s = z @ w
    # map back toward number space via PCA loadings
    # source → number affinity: |corr(source, presence_n)|
    x = presence_matrix(draws)
    top = []
    for c in range(k):
        corrs = []
        for n in range(FRONT_N):
            a = s[:, c]
            b = x[:, n]
            den = (a.std() * b.std()) + 1e-12
            r = float(np.corrcoef(a, b)[0, 1]) if den > 0 else 0.0
            corrs.append((n + 1, r))
        corrs.sort(key=lambda t: (-abs(t[1]), t[0]))
        top.append(corrs[:8])
    return {"sources_top_numbers": top}


def cca_lag_blocks(draws: np.ndarray, n_comp: int = 3) -> dict:
    """
    CCA između presence_t i presence_{t+1} (blokovi kroz vreme).
    Canonical correlations = latentna veza kolo→kolo.
    """
    x = presence_matrix(draws)
    a = x[:-1]
    b = x[1:]
    # center
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    # covariance
    n = len(a)
    caa = (a.T @ a) / n + np.eye(FRONT_N) * 1e-4
    cbb = (b.T @ b) / n + np.eye(FRONT_N) * 1e-4
    cab = (a.T @ b) / n
    # solve generalized: caa^{-1/2} cab cbb^{-1/2}
    wa, va = np.linalg.eigh(caa)
    wb, vb = np.linalg.eigh(cbb)
    caa_inv_sqrt = va @ np.diag(1.0 / np.sqrt(np.maximum(wa, 1e-12))) @ va.T
    cbb_inv_sqrt = vb @ np.diag(1.0 / np.sqrt(np.maximum(wb, 1e-12))) @ vb.T
    m = caa_inv_sqrt @ cab @ cbb_inv_sqrt
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    k = min(n_comp, len(s))
    # A loadings in original space
    xa = caa_inv_sqrt @ u[:, :k]
    yb = cbb_inv_sqrt @ vt[:k].T
    top_a, top_b = [], []
    for c in range(k):
        ia = np.argsort(-np.abs(xa[:, c]))[:8]
        ib = np.argsort(-np.abs(yb[:, c]))[:8]
        top_a.append([(int(i + 1), float(xa[i, c])) for i in ia])
        top_b.append([(int(i + 1), float(yb[i, c])) for i in ib])
    return {
        "canonical_corr": [float(v) for v in s[:k]],
        "top_from_t": top_a,
        "top_to_t1": top_b,
        "xa": xa,
        "yb": yb,
    }


def lsa_cooccurrence(draws: np.ndarray, n_comp: int = N_COMP) -> dict:
    """LSA: SVD na broj×broj co-occurrence matrici (ceo CSV)."""
    co = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for draw in draws:
        nums = draw.tolist()
        for i, a in enumerate(nums):
            for b in nums[i + 1 :]:
                co[a - 1, b - 1] += 1.0
                co[b - 1, a - 1] += 1.0
    # PMI-ish log1p
    m = np.log1p(co)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    k = min(n_comp, len(s))
    emb = u[:, :k] * np.sqrt(s[:k])
    top = []
    for c in range(k):
        idx = np.argsort(-np.abs(emb[:, c]))[:8]
        top.append([(int(i + 1), float(emb[i, c])) for i in idx])
    return {"singular_values": [float(v) for v in s[:k]], "topic_top": top, "embedding": emb}


def learn_next_rule(draws: np.ndarray) -> dict:
    """
    Pravilo next iz grupe 8:
    skor(y) = sličnost u LSA/PCA prostoru sa last draw + NMF težina poslednjeg kola.
    """
    lsa = lsa_cooccurrence(draws)
    pca = pca_svd(draws)
    nmf = nmf_presence(draws)
    emb = lsa["embedding"]
    load = pca["loadings"]

    last = [int(v) for v in draws[-1].tolist()]
    last_vec_lsa = emb[[n - 1 for n in last]].mean(axis=0)
    last_vec_pca = load[[n - 1 for n in last]].mean(axis=0)

    # NMF: poslednje kolo → latent W[-1], pa H.T @ w
    w_last = nmf["W"][-1]
    nmf_score = nmf["H"].T @ w_last

    freq = Counter(draws.reshape(-1).tolist())
    max_f = max(freq.values()) if freq else 1

    number_score = {}
    for y in range(1, FRONT_N + 1):
        ey = emb[y - 1]
        py = load[y - 1]
        sim_lsa = float(np.dot(ey, last_vec_lsa) / ((np.linalg.norm(ey) * np.linalg.norm(last_vec_lsa)) + 1e-12))
        sim_pca = float(np.dot(py, last_vec_pca) / ((np.linalg.norm(py) * np.linalg.norm(last_vec_pca)) + 1e-12))
        number_score[y] = (
            0.45 * sim_lsa
            + 0.35 * sim_pca
            + 0.35 * float(nmf_score[y - 1] / (nmf_score.max() + 1e-12))
            + 0.1 * (freq.get(y, 0) / max_f)
        )

    return {
        "number_score": number_score,
        "last_draw": last,
        "target_sum": float(draws.sum(axis=1).mean()),
        "pca_explained": pca["explained_var_ratio"],
    }


def _combo_fit(combo: list[int], rule: dict) -> float:
    score = sum(rule["number_score"][x] for x in combo)
    score -= 0.015 * abs(sum(combo) - rule["target_sum"])
    return score


def predict_next_from_rule(draws: np.ndarray, rule: dict | None = None) -> list[int]:
    if rule is None:
        rule = learn_next_rule(draws)
    ranked = sorted(rule["number_score"], key=lambda n: (-rule["number_score"][n], n))
    best = None
    best_fit = -1e18
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        base = sorted(ranked[start : start + FRONT_SELECT])
        for repl in ranked[:28]:
            cand = sorted(set(base[1:] + [repl]))
            if len(cand) != FRONT_SELECT:
                continue
            fit = _combo_fit(cand, rule)
            if fit > best_fit:
                best_fit = fit
                best = cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_grupa8(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | 7/39 | grupa8")
    print()

    pca = pca_svd(draws)
    print("=== PCA/SVD explained_var ===")
    print(pca["explained_var_ratio"])
    print("top loadings PC1:", pca["top_loadings"][0])
    print()

    print("=== NMF factors (top brojevi) ===")
    nmf = nmf_presence(draws)
    print({"recon_rel_err": nmf["recon_rel_err"], "H_top": nmf["H_top"]})
    print()

    print("=== ICA sources → top |corr| brojevi ===")
    print(ica_fast_proxy(draws)["sources_top_numbers"])
    print()

    print("=== CCA presence_t ↔ presence_{t+1} ===")
    cca = cca_lag_blocks(draws)
    print({"canonical_corr": cca["canonical_corr"], "top_from_t0": cca["top_from_t"][0], "top_to_t1": cca["top_to_t1"][0]})
    print()

    print("=== LSA co-occurrence topics ===")
    lsa = lsa_cooccurrence(draws)
    print({"sv": lsa["singular_values"], "topic0": lsa["topic_top"][0]})
    print()

    print("=== pravilo → next (grupa 8) ===")
    rule = learn_next_rule(draws)
    combo = predict_next_from_rule(draws, rule)
    print(
        "rule:",
        {
            "last_draw": rule["last_draw"],
            "target_sum": round(rule["target_sum"], 2),
            "pca_explained": [round(v, 4) for v in rule["pca_explained"]],
        },
    )
    print("next:", combo)


if __name__ == "__main__":
    run_grupa8()


"""
8. Latentna struktura
PCA, sparse PCA, ICA, NMF, FA (factor analysis), SVD, truncated SVD, CCA, PLS
(kao veza blokova), CP/PARAFAC, Tucker, tensor train, LDA/topic models, LSA,
embedding struktura (samo kao mapa sličnosti)
"""



"""
CSV: loto7_4648_k55.csv
Kola: 4648 | seed=39 | 7/39 | grupa8

=== PCA/SVD explained_var ===
[0.03094479104664385, 0.030268039323091946, 0.03004008880569167,0.02976689014978091, 0.029252189084418977]
top loadings PC1: [(8, -0.4781092659770914), (28, -0.301468818366985), (33, -0.2790757935671283), (39, 0.2563280420063291), (6, 0.2474016499612076), (23, 0.24605615944648293), (35, 0.23051082250357238), (32, 0.22496078158395444)]

=== NMF factors (top brojevi) ===
{'recon_rel_err': 0.8541078017509212, 'H_top': [[(34, 0.3608491981642529), (2, 0.3250720137458874), (37, 0.3163806946626482), (13, 0.22832730323227374), (1, 0.19530494025149328), (19, 0.15778582355544613), (8, 0.152064604470614), (17, 0.12854352423266038)], [(29, 0.6193599093344946), (4, 0.45645515829631245), (18, 0.05443650892291596), (11, 0.04773865756227184), (36, 0.04746643191057517), (19, 0.043663771283363), (39, 0.04311253858017997), (6, 0.042153951658171734)], [(38, 0.4044244815773116), (5, 0.3648102532779369), (27, 0.34011297465403084), (9, 0.23641630001918304), (14, 0.18722531544148158), (21, 0.1384054354397639), (24, 0.1084632644231203), (13, 0.08346145924621902)], [(32, 0.24004838521894872), (25, 0.23753971911357927), (35, 0.22927664288598432), (39, 0.21853136580320973), (16, 0.21519403604926512), (36, 0.1932762706997101), (12, 0.1740519648763057), (31, 0.17112459084613077)], [(23, 0.3336676972244023), (33, 0.3244768121277207), (11, 0.28026817365402595), (3, 0.2621852692809767), (15, 0.2600979379134982), (18, 0.2584487295098125), (22, 0.23768622179141677), (8, 0.1837103870912325)]]}

=== ICA sources → top |corr| brojevi ===
[[(28, 0.39046100734541156), (34, -0.3618038536539897), (29, -0.3576744578283231), (5, 0.2589961562282116), (21, 0.23912844757332835), (24, 0.2311284385346832), (19, -0.2305795707585603), (3, 0.22061964231178846)], [(34, 0.4381806887568236), (33, 0.38390084036132294), (23, -0.30086949685590575), (31, -0.2884241572997914), (37, -0.2785388166894016), (6, -0.2202705718516423), (21, -0.21669429427742312), (9, -0.21666084663045185)], [(26, 0.38755850507507206), (23, 0.32979067512918175), (38, 0.3291547227344132), (29, -0.2957085184817554), (16, -0.28955286334801783), (35, 0.2726496925532568), (37, -0.2471941679838284), (22, -0.24063937872417826)], [(39, 0.501010383108813), (32, 0.38891809945070216), (8, -0.3217535379021562), (37, -0.2624413830837046), (12, 0.25234309953629347), (13, -0.23698416901665187), (38, -0.20639710257487826), (6, 0.20424913621512839)], [(23, -0.5643211238302968), (8, -0.37961765465022945), (24, 0.33308657284551263), (11, -0.28025839860418716), (9, 0.19766820024546294), (13, 0.19409026269425236), (10, 0.19136169441079398), (22, -0.16518855821750345)]]

=== CCA presence_t ↔ presence_{t+1} ===
{'canonical_corr': [0.17025760807329515, 0.1619308292091662, 0.15340508962554641], 'top_from_t0': [(9, -0.8378424459466299), (12,-0.7626291804267682), (30, -0.7240525412897697), (14, -0.7169505999558219), (38, -0.7040198791702704), (24, 0.5765941524108625), (39, 0.5762229856795928), (6, -0.5714324722395311)], 'top_to_t1':[(7, 1.088202018150946), (28, -0.8458936339966112), (33, 0.7171254401914762), (23, -0.6577514584183991), (12, -0.6500214681881843), (21, -0.5756288026960518), (2, 0.571341076858796), (1, -0.5506392971119615)]}

=== LSA co-occurrence topics ===
{'sv': [185.6163440617029, 5.670306558847146, 5.564615801065318,5.5107695103170045, 5.483304998622095], 'topic0': [(8, -2.221848044733184), (23, -2.218448579474425), (34, -2.2029162049507574), (26, -2.200473751642215), (37, -2.196523270081956), (11, -2.195901764326212), (32, -2.195173668790193), (33, -2.1925523412675614)]}

=== pravilo → next (grupa 8) ===
rule: {'last_draw': [3, 7, 12, 13, 18, 24, 29], 'target_sum': 140.43, 'pca_explained': [0.0309, 0.0303, 0.03, 0.0298, 0.0293]}
next: [4, 9, 13, 18, 24, 29, 38]
"""
