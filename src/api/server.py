"""
FastAPI Backend Server for Strategic Foresight Engine.
Serves interactive APIs for TKG visualization, scenario compilation & testing,
particle filter monitoring, game-theoretic equilibria, KAN formula extraction,
and verifiable narrative reporting.
"""
import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from src.pipeline import ForesightEngine
from src.core.actions import Action
from src.extraction.observation_model import Claim

app = FastAPI(title="Strategic Foresight Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine singleton
engine = ForesightEngine()

# Initialize with standard benchmark scenarios
benchmark_scenarios = [
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

print("Initializing engine benchmark scenarios...")
engine.compile_assigned_scenarios(benchmark_scenarios)
print("Engine initialization complete (training on demand).")


class CompileRequest(BaseModel):
    id: str
    title: str
    narrative_text: str


class ExtractionRequest(BaseModel):
    document_text: str
    source_outlet: str = "Reuters"


@app.get("/api/state")
def get_world_state():
    """Returns current sparse TKG state and exogenous indicators."""
    return engine.current_state.to_dict()


@app.get("/api/scenarios")
def get_scenarios():
    """Returns all compiled scenarios and their 5-stage test findings."""
    res = []
    for sid, spec in engine.compiled_specs.items():
        tests = engine.test_results.get(sid, [])
        res.append({
            "spec": spec.to_dict(),
            "tests": [
                {
                    "test_name": t.test_name,
                    "status": t.status,
                    "summary": t.summary,
                    "metric_value": t.metric_value,
                    "details": t.details,
                }
                for t in tests
            ]
        })
    return {"scenarios": res}


@app.post("/api/compile")
def compile_scenario(req: CompileRequest):
    """Compiles a new scenario narrative into formal specification and runs 5 tests."""
    res = engine.compile_assigned_scenarios([{
        "id": req.id,
        "title": req.title,
        "text": req.narrative_text
    }])
    return res


@app.post("/api/extract")
def extract_present_claims(req: ExtractionRequest):
    """Extracts grounded claims from present reporting using LLM noisy sensor."""
    res = engine.run_present_state_extraction([{
        "text": req.document_text,
        "source": req.source_outlet
    }])
    return res


@app.get("/api/signposts")
def get_signposts():
    """Returns forward 6-12 month diagnostic signposts."""
    signposts = engine.signpost_id.identify_signposts(engine.scenarios, engine.current_state)
    return {
        "signposts": [
            {
                "variable": s.variable_name,
                "type": s.variable_type,
                "divergence": s.divergence_score,
                "scenario_predictions": s.scenario_predictions,
                "threshold": s.recommended_trigger_threshold,
                "direction": s.trigger_direction,
            }
            for s in signposts
        ]
    }


@app.get("/api/psro")
def get_psro_results():
    """Returns PSRO meta-game equilibria and interventional relevance."""
    if not engine.psro_result:
        engine.run_game_theoretic_psro()
    return {
        "nash_conv_lower_bound": engine.psro_result.nash_conv,
        "iterations_run": engine.psro_result.iterations_run,
        "qre_expected_payoffs": {k: round(v, 3) for k, v in engine.qre_result.expected_payoffs.items()},
        "interventional_relevance": engine.qre_result.interventional_relevance,
    }


@app.get("/api/kan")
def get_kan_surrogate():
    """Returns KAN surrogate architecture, snapped formulas, and fidelity scores."""
    if not engine.kan_model:
        engine.train_kan_surrogate()

    edges_data = []
    for e in engine.snapped_edges:
        edges_data.append({
            "layer": e.layer_index,
            "input_idx": e.input_index,
            "output_idx": e.output_index,
            "is_active": e.is_active,
            "is_snapped": e.is_snapped,
            "formula": e.formula,
            "r2": e.r2,
            "raw_spline_points": e.raw_spline_points,
        })

    return {
        "fidelity": {
            "in_distribution_agreement_pct": engine.fidelity_metrics.in_distribution_agreement,
            "in_distribution_kl": engine.fidelity_metrics.in_distribution_kl,
            "off_distribution_agreement_pct": engine.fidelity_metrics.off_distribution_agreement,
            "off_distribution_kl": engine.fidelity_metrics.off_distribution_kl,
            "active_edges": engine.fidelity_metrics.active_edges,
            "snapped_edges": engine.fidelity_metrics.snapped_edges,
            "reliable_explanation": engine.fidelity_metrics.reliable_explanation,
        },
        "formulas": engine.snapped_formulas,
        "edges": edges_data,
        "concept_names": ["Trade Exposure", "Rival Tariff", "Tension", "GDP Growth", "Energy Stress", "Tau (Horizon)"],
        "action_names": [a.name() for a in engine.game.action_spaces["USA"]],
    }


@app.get("/api/robustness")
def get_robustness():
    """Returns strategy x scenario robustness matrix and verifiable cited narrative reports."""
    if not engine.robustness_matrix:
        engine.evaluate_robustness_and_narrate()

    cells = []
    for c in engine.robustness_matrix.matrix.values():
        cells.append({
            "strategy": c.strategy_name,
            "scenario": c.scenario_id,
            "median": c.median_value,
            "downside_p10": c.downside_p10,
            "regret": c.regret,
            "trace": c.trajectory_trace,
        })

    # Find reports
    reports = []
    for sid in engine.compiled_specs:
        cell = engine.robustness_matrix.matrix.get((engine.robustness_matrix.recommended_strategy, sid))
        if cell:
            debrief = engine.reporter.generate_report(cell, engine.compiled_specs[sid], focal_actor="USA")
            reports.append({
                "scenario_id": sid,
                "strategy": cell.strategy_name,
                "median_narrative": [s.text for s in debrief.median_narrative],
                "downside_narrative": [s.text for s in debrief.downside_narrative],
                "model_departures": debrief.model_departures,
                "fidelity_score": debrief.citation_fidelity_score,
            })

    return {
        "recommended_strategy": engine.robustness_matrix.recommended_strategy,
        "worst_case_regrets": engine.robustness_matrix.worst_case_regrets,
        "dominated_strategies": engine.robustness_matrix.dominated_strategies,
        "cells": cells,
        "narrative_reports": reports,
    }


# Mount web directory
web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../web"))
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(web_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
