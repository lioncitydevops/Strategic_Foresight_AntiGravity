"""
Symbolic Snapping for KAN Edges (Section 9.4).
Fits surviving univariate spline edges against a library of simple functions:
(polynomial, exponential, logarithm, sigmoid, step).
Snaps to closed-form algebraic formula if R^2 exceeds threshold.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from src.core.actions import Action
from src.surrogate.kan import KolmogorovArnoldNetwork, KANSplineEdge


@dataclass
class SnappedEdge:
    layer_index: int
    input_index: int
    output_index: int
    is_active: bool
    is_snapped: bool
    formula: str
    r2: float
    raw_spline_points: Optional[List[Tuple[float, float]]] = None


class SymbolicSnapper:
    """Snaps surviving KAN edges to human-auditable algebraic formulas."""

    def __init__(self, r2_threshold: float = 0.70):
        self.r2_threshold = r2_threshold

    def _fit_candidate_functions(self, x: np.ndarray, y: np.ndarray) -> Tuple[str, float]:
        """Tests library of functions against spline curve and returns best (formula, R^2)."""
        best_r2 = -1.0
        best_formula = "spline(x)"

        mean_y = float(np.mean(y))
        std_y = float(np.std(y))

        # 1. Flat / Constant: c
        if std_y < 0.04:
            return f"{mean_y:.2f}", 0.98

        # 2. Linear: a * x + b via np.polyfit
        try:
            poly1 = np.polyfit(x, y, deg=1)
            pred1 = np.polyval(poly1, x)
            r2_1 = float(r2_score(y, pred1))
            if r2_1 > best_r2:
                best_r2 = r2_1
                sign = "+" if poly1[1] >= 0 else "-"
                best_formula = f"{poly1[0]:.2f}x {sign} {abs(poly1[1]):.2f}"
        except Exception:
            pass

        # 3. Quadratic: a * x^2 + b * x + c via np.polyfit
        try:
            poly2 = np.polyfit(x, y, deg=2)
            pred2 = np.polyval(poly2, x)
            r2_2 = float(r2_score(y, pred2))
            if r2_2 > best_r2:
                best_r2 = r2_2
                sign1 = "+" if poly2[1] >= 0 else "-"
                sign2 = "+" if poly2[2] >= 0 else "-"
                best_formula = f"{poly2[0]:.2f}x² {sign1} {abs(poly2[1]):.2f}x {sign2} {abs(poly2[2]):.2f}"
        except Exception:
            pass

        # 4. Sigmoid: a / (1 + exp(-b*(x - c))) + d
        try:
            popt, _ = curve_fit(
                lambda t, a, b, c, d: a / (1.0 + np.exp(-np.clip(b * (t - c), -10, 10))) + d,
                x, y, p0=[1.0, 1.0, 0.0, 0.0], maxfev=500
            )
            pred = popt[0] / (1.0 + np.exp(-np.clip(popt[1] * (x - popt[2]), -10, 10))) + popt[3]
            r2 = float(r2_score(y, pred))
            if r2 > best_r2:
                best_r2 = r2
                best_formula = f"{popt[0]:.2f}·σ({popt[1]:.2f}(x - {popt[2]:.2f})) + {popt[3]:.2f}"
        except Exception:
            pass

        # 5. Exponential: a * exp(b * x)
        try:
            popt, _ = curve_fit(lambda t, a, b: a * np.exp(np.clip(b * t, -5, 5)), x, y, p0=[0.1, 0.5], maxfev=500)
            pred = popt[0] * np.exp(np.clip(popt[1] * x, -5, 5))
            r2 = float(r2_score(y, pred))
            if r2 > best_r2:
                best_r2 = r2
                best_formula = f"{popt[0]:.2f}·exp({popt[1]:.2f}x)"
        except Exception:
            pass

        return best_formula, float(best_r2)

    def snap_network(
        self,
        kan: KolmogorovArnoldNetwork,
        input_feature_names: List[str],
        output_action_names: List[str]
    ) -> List[SnappedEdge]:
        """Snaps all surviving active edges in the KAN model."""
        snapped_edges = []
        x_eval = torch.linspace(-1.5, 1.5, 60).unsqueeze(1)
        x_np = x_eval.squeeze(1).numpy()

        # Layer 1
        for j in range(kan.hidden_dim):
            for i in range(kan.in_dim):
                edge = kan.layer1.edges[j][i]
                if not edge.is_active:
                    snapped_edges.append(SnappedEdge(
                        layer_index=1, input_index=i, output_index=j,
                        is_active=False, is_snapped=False, formula="0 (pruned)", r2=1.0
                    ))
                    continue

                with torch.no_grad():
                    y_eval = edge(x_eval).numpy()

                formula, r2 = self._fit_candidate_functions(x_np, y_eval)
                is_snapped = (r2 >= self.r2_threshold)
                edge.snapped_formula = formula if is_snapped else "spline(x)"
                edge.r2_score = r2

                raw_pts = [(float(x_np[idx]), float(y_eval[idx])) for idx in range(0, len(x_np), 5)]

                snapped_edges.append(SnappedEdge(
                    layer_index=1,
                    input_index=i,
                    output_index=j,
                    is_active=True,
                    is_snapped=is_snapped,
                    formula=formula if is_snapped else f"B-Spline (R²={r2:.2f})",
                    r2=round(r2, 3),
                    raw_spline_points=raw_pts
                ))

        # Layer 2
        for j in range(kan.out_dim):
            for i in range(kan.hidden_dim):
                edge = kan.layer2.edges[j][i]
                if not edge.is_active:
                    snapped_edges.append(SnappedEdge(
                        layer_index=2, input_index=i, output_index=j,
                        is_active=False, is_snapped=False, formula="0 (pruned)", r2=1.0
                    ))
                    continue

                with torch.no_grad():
                    y_eval = edge(x_eval).numpy()

                formula, r2 = self._fit_candidate_functions(x_np, y_eval)
                is_snapped = (r2 >= self.r2_threshold)
                edge.snapped_formula = formula if is_snapped else "spline(h)"
                edge.r2_score = r2

                raw_pts = [(float(x_np[idx]), float(y_eval[idx])) for idx in range(0, len(x_np), 5)]

                snapped_edges.append(SnappedEdge(
                    layer_index=2,
                    input_index=i,
                    output_index=j,
                    is_active=True,
                    is_snapped=is_snapped,
                    formula=formula if is_snapped else f"B-Spline (R²={r2:.2f})",
                    r2=round(r2, 3),
                    raw_spline_points=raw_pts
                ))

        return snapped_edges

    def extract_full_policy_formula(
        self,
        kan: KolmogorovArnoldNetwork,
        concept_names: List[str],
        action_names: List[Action]
    ) -> List[Dict[str, str]]:
        """Extracts complete explicit algebraic formulas for each action logit."""
        formulas = []
        for j, act in enumerate(action_names):
            # Logit_j = sum_i phi_2,j,i(h_i)
            terms = []
            for i in range(kan.hidden_dim):
                edge2 = kan.layer2.edges[j][i]
                if edge2.is_active and edge2.snapped_formula:
                    terms.append(f"({edge2.snapped_formula.replace('x', f'h_{i}')})")
            body = " + ".join(terms) if terms else "0.0"
            formulas.append({
                "action": act.name(),
                "logit_formula": f"Logit({act.name()}) = {body}",
                "description": f"Auditable algebraic policy decision rule for action {act.action_type} on {act.target}"
            })
        return formulas
