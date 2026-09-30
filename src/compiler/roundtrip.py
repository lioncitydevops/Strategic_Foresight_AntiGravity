"""
Round-trip narrative synthesis from formal scenario specification Sigma_n.
Renders compiled specification into readable prose so the scenario owner can sign off.
"""
from src.compiler.compiler import ScenarioSpec


class ScenarioRoundTrip:
    """Renders formal specification Sigma_n back into prose for owner review."""

    @staticmethod
    def render_to_prose(spec: ScenarioSpec) -> str:
        paragraphs = []
        paragraphs.append(f"### Formalized Scenario: {spec.title} (ID: {spec.scenario_id})")

        # 1. Premises (Causal interventions)
        if spec.interventions:
            premise_lines = []
            for t, act in spec.interventions:
                premise_lines.append(f"- At time t={t}, intervention do({act.name()}) is imposed.")
            paragraphs.append("**Stipulated Premises (Interventions):**\n" + "\n".join(premise_lines))

        # 2. Driver constraints
        if spec.driver_constraints:
            driver_lines = []
            for d in spec.driver_constraints:
                cond = []
                if d.min_value is not None:
                    cond.append(f">= {d.min_value}")
                if d.max_value is not None:
                    cond.append(f"<= {d.max_value}")
                driver_lines.append(f"- Exogenous driver '{d.indicator}' constrained to {' and '.join(cond)} across t in {d.time_window}.")
            paragraphs.append("**Exogenous Driver Constraints:**\n" + "\n".join(driver_lines))

        # 3. Actor dispositions
        if spec.actor_dispositions:
            disp_lines = []
            for disp in spec.actor_dispositions:
                bounds_str = ", ".join([f"{k} in [{v[0]:.2f}, {v[1]:.2f}]" for k, v in disp.weight_bounds.items()])
                disp_lines.append(f"- Actor {disp.actor}'s preference type restricted to: {bounds_str}.")
            paragraphs.append("**Actor Dispositions:**\n" + "\n".join(disp_lines))

        # 4. Milestone / End state
        if spec.conditioning_events:
            cond_lines = []
            for c in spec.conditioning_events:
                kind = "Hard condition" if c.is_hard else f"Soft condition (kappa={c.penalty_kappa})"
                cond_lines.append(f"- {kind}: Requires '{c.target_variable}' {c.comparison_op} {c.target_value} by t={c.time_step}.")
            paragraphs.append("**Conditioning Milestones & End States (Backcasting Targets):**\n" + "\n".join(cond_lines))

        # 5. Causal claims
        if spec.causal_hypotheses:
            hypo_lines = []
            for h in spec.causal_hypotheses:
                hypo_lines.append(f"- Hypothesis: {h.cause_action.name()} causes '{h.expected_effect}' with impact on {h.target_metric}.")
            paragraphs.append("**Empirical Causal Hypotheses (To be tested, not imposed):**\n" + "\n".join(hypo_lines))

        # 6. Flagged unmapped
        if spec.unmapped_sentences:
            unmapped = "\n".join([f"- \"{s}\"" for s in spec.unmapped_sentences])
            paragraphs.append("**Sentences Flagged Outside Model Ontology:**\n" + unmapped)

        return "\n\n".join(paragraphs)
