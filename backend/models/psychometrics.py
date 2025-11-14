import numpy as np
from typing import List, Dict
import math
try:
    from factor_analyzer import FactorAnalyzer
    _HAS_FACTOR = True
except Exception:
    _HAS_FACTOR = False


class PsychometricEngine:

    def cronbach_alpha(self, X: np.ndarray) -> float:
        # Cronbach's alpha requires at least 2 items (k>=2)
        k = X.shape[1]
        if k < 2:
            return float('nan')

        var_items = X.var(axis=0, ddof=1)
        total_var = X.sum(axis=1).var(ddof=1)
        if total_var == 0:
            return float('nan')
        return (k / (k - 1)) * (1 - var_items.sum() / total_var)

    def guttman_lambda6(self, X: np.ndarray) -> float:
        k = X.shape[1]
        if k < 2:
            return float('nan')
        item_vars = X.var(axis=0, ddof=1)
        total_var = X.sum(axis=1).var(ddof=1)
        if total_var == 0:
            return float('nan')
        return 1 - (item_vars.sum() / total_var)

    def mcdonald_omega(self, X: np.ndarray) -> float:
        # Use factor analyzer if available and there are at least 2 items
        if not _HAS_FACTOR or X.shape[1] < 2:
            return float('nan')

        try:
            fa = FactorAnalyzer(n_factors=1)
            fa.fit(X)
            loadings = fa.loadings_.flatten()
            denom = (loadings.sum() ** 2) + fa.get_uniquenesses().sum()
            if denom == 0:
                return float('nan')
            return (loadings.sum() ** 2) / denom
        except Exception:
            return float('nan')

    def compute_validity(self, all_scores: List[Dict]) -> Dict:
        # Convert dicts into domain matrices. We expect many items per domain;
        # if only one item exists, psychometrics will return NaNs rather than crash.
        domains = {}
        for entry in all_scores:
            d = entry.get("domain")
            if d is None:
                continue
            domains.setdefault(d, []).append(int(entry.get("score", 0)))

        stats = {}
        for d, vals in domains.items():
            # arrange as rows=observations, cols=items; we only have a 1D list
            # which represents repeated measurements; try to coerce into 2D
            X = np.array(vals)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            alpha = self.cronbach_alpha(X)
            lam6 = self.guttman_lambda6(X)
            omega = self.mcdonald_omega(X)

            stats[d] = {
                "alpha": (None if math.isnan(alpha) else float(alpha)),
                "lambda6": (None if math.isnan(lam6) else float(lam6)),
                "omega": (None if math.isnan(omega) else float(omega)),
            }

        return stats

