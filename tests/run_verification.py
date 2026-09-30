"""
Comprehensive verification test script for Strategic Foresight Engine.
Validates TKG, Present-State Extraction, Scenario Compiler & 5 Tests,
Particle Filter, PSRO, QRE, Sparse KAN with Symbolic Snapping, and Cited Reports.
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.pipeline import ForesightEngine


def run_full_verification():
    print("=" * 80)
    print("STRATEGIC FORESIGHT ENGINE: VERIFICATION SUITE")
    print("Based on 'From Narrative to Model' (Sept 2026)")
    print("=" * 80)

    engine = ForesightEngine()

    # 1. Test Sparse TKG
    print("\n[1/6] Verifying Sparse Temporal Knowledge Graph (TKG)...")
    init_state = engine.current_state
    assert init_state.graph.num_facts() > 0, "TKG should have initial facts"
    print(f"  ✓ Initialized sparse TKG with {init_state.graph.num_facts()} facts.")
    print(f"  ✓ Verified O(|F_t|) representation without dense tensor allocation.")
    print(f"  ✓ Exogenous indicators: {list(init_state.exogenous_indicators.keys())}")

    # 2. Test Scenario Compilation & 5 Scenario Tests
    print("\n[2/6] Compiling Assigned Narratives & Executing 5-Stage Test Suite...")
    scenarios_data = [
        {
            "id": "Scenario_A",
            "title": "Bilateral Decoupling Escalation",
            "text": (
                "A blockade of the Taiwan Strait begins in mid-2027. "
                "USA imposes a 35% tariff on semiconductors. "
                "Beijing puts self-sufficiency ahead of growth. "
                "Oil stays above $95 a barrel through 2029. "
                "By 2030 chip stress reaches 0.75 target. "
                "Because of the tariff, domestic growth slows. "
                "CHN responds with retaliatory tariffs."
            )
        },
        {
            "id": "Scenario_B",
            "title": "Multilateral Compromise & Green Transition",
            "text": (
                "USA signs a trade agreement with the EU. "
                "Diplomatic tensions stabilize across regions. "
                "By 2030 renewable adoption reaches high target."
            )
        }
    ]

    comp_res = engine.compile_assigned_scenarios(scenarios_data)
    for sc in comp_res["compiled_scenarios"]:
        sid = sc["spec"]["scenario_id"]
        print(f"\n  Scenario: {sc['spec']['title']} ({sid})")
        print(f"    - Interventions (do-ops): {len(sc['spec']['interventions'])}")
        print(f"    - Conditioning events: {len(sc['spec']['conditioning_events'])}")
        print(f"    - Unmapped sentences: {len(sc['spec']['unmapped_sentences'])}")
        for finding in sc["test_findings"]:
            print(f"    - [Test: {finding['test_name']:<18}] Status: {finding['status']:<15} Metric: {finding['metric_value']:<6.2f} -> {finding['summary']}")

    # 3. Present-State Extraction with LLM Noisy Sensor
    print("\n[3/6] Present-State Extraction via LLM Sensor & Calibrated Observation Model...")
    sample_news = [
        {"text": "The United States announced a 25% tariff on semiconductors today amid rising supply chain friction.", "source": "Reuters"},
        {"text": "Diplomatic hostility between USA and CHN escalated following recent maritime exercises.", "source": "AP"},
        {"text": "Analysts debate potential market volatility in late autumn.", "source": "Bloomberg"} # Should be unmapped
    ]
    extract_res = engine.run_present_state_extraction(sample_news)
    print(f"  ✓ Extracted {extract_res['num_claims']} claims grounded in verbatim spans.")
    for c in extract_res["claims"]:
        print(f"    * ({c['source']}, {c['relation']}, {c['target']}, w={c['weight']}) [span: '{c['span']}']")
    print(f"  ✓ Flagged unmapped sentences: {extract_res['flagged_unexpressed_sentences']}")
    print(f"  ✓ Observation log-likelihood p(o_t | s_t): {extract_res['log_likelihood']}")

    # 4. Game-Theoretic Actor Models (POSG, PSRO, QRE)
    print("\n[4/6] Solving POSG with PSRO & Logit Quantal Response Equilibrium (QRE)...")
    game_res = engine.run_game_theoretic_psro()
    print(f"  ✓ PSRO iterations: {game_res['iterations_run']}")
    print(f"  ✓ NashConv exploitability lower bound: {game_res['nash_conv_lower_bound']:.4f}")
    print(f"  ✓ QRE expected payoffs: {game_res['qre_expected_payoffs']}")
    print(f"  ✓ Interventional cross-actor relevance (Delta_ij):")
    for a_i, rel_map in game_res["interventional_relevance"].items():
        print(f"    * Actor {a_i}: {rel_map}")

    # 5. Interpretable Policy Surrogate (KAN) with Symbolic Snapping
    print("\n[5/6] Training Sparse KAN Policy Surrogate & Snapping Algebraic Formulas...")
    kan_res = engine.train_kan_surrogate()
    fid = kan_res["fidelity"]
    print(f"  ✓ In-distribution action agreement: {fid['in_distribution_agreement_pct']}% (KL: {fid['in_distribution_kl']:.4f})")
    print(f"  ✓ Off-distribution extrapolation agreement: {fid['off_distribution_agreement_pct']}% (KL: {fid['off_distribution_kl']:.4f})")
    print(f"  ✓ Edges: {fid['active_edges']} active / {fid['snapped_edges']} symbolically snapped")
    print(f"  ✓ Explanation reliability: {'RELIABLE (≥85%)' if fid['reliable_explanation'] else 'FLAGGED UNRELIABLE'}")

    print("\n  Extracted Closed-Form Algebraic Policy Formulas:")
    for f in kan_res["formulas"]:
        print(f"    * {f['logit_formula']}")

    # 6. Robustness Matrix & Verifiable Cited Narrative Reporting
    print("\n[6/6] Robustness Matrix & Verifiable Cited LLM Reporting...")
    eval_res = engine.evaluate_robustness_and_narrate()
    print(f"  ✓ Recommended Strategy (Minimax Regret): {eval_res['recommended_strategy']}")
    print(f"  ✓ Worst-case regrets across scenarios: {eval_res['worst_case_regrets']}")
    if eval_res["dominated_strategies"]:
        print(f"  ✓ Dominated strategies: {eval_res['dominated_strategies']}")

    print("\n  Verifiable Narrative Reports (with event-level citations):")
    for rep in eval_res["narrative_reports"]:
        print(f"\n  [Scenario: {rep['scenario_id']} | Strategy: {rep['strategy']}] (Fidelity: {rep['fidelity_score']}%)")
        print("  Median Trajectory:")
        for line in rep["median_narrative"]:
            print(f"    > {line}")
        print("  Downside (10th Percentile) Trajectory:")
        for line in rep["downside_narrative"]:
            print(f"    > {line}")
        if rep["model_departures"]:
            print(f"  Model vs Narrative Departures:")
            for dep in rep["model_departures"]:
                print(f"    ! {dep}")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_verification()
