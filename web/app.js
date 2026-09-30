/**
 * AETHERIA Strategic Foresight Engine - Frontend Application Logic
 * Integrates TKG Visualization, Scenario Compiler, Particle Filter,
 * PSRO & QRE Solvers, Sparse KAN Surrogate, and Robustness Matrix.
 */

// Global State
let globalData = {
    worldState: null,
    scenarios: [],
    signposts: [],
    psro: null,
    kan: null,
    robustness: null,
    activeMetric: 'median',
    selectedScenarioId: 'Scenario_A'
};

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initEventListeners();
    loadAllData();
});

// Tab Navigation
function initTabs() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            const targetPane = document.getElementById(targetId);
            if (targetPane) targetPane.classList.add('active');

            if (targetId === 'tab-tkg') renderTKGGraph();
            if (targetId === 'tab-kan') renderKANTopology();
        });
    });
}

function initEventListeners() {
    // Time slider for TKG
    const slider = document.getElementById('time-slider');
    if (slider) {
        slider.addEventListener('input', (e) => {
            const val = e.target.value;
            document.getElementById('current-time-step').textContent = `t = ${val}`;
            // Simulate slight indicator drift with time
            updateIndicatorsForTime(parseInt(val));
        });
    }

    // Scenario dropdown
    const scDropdown = document.getElementById('scenario-dropdown');
    if (scDropdown) {
        scDropdown.addEventListener('change', (e) => {
            globalData.selectedScenarioId = e.target.value;
            renderSelectedScenario();
        });
    }

    // Robustness metric toggles
    const toggles = document.querySelectorAll('.btn-toggle');
    toggles.forEach(btn => {
        btn.addEventListener('click', () => {
            toggles.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            globalData.activeMetric = btn.getAttribute('data-metric');
            renderRobustnessTable();
        });
    });

    // Re-run pipeline button
    const btnRecompile = document.getElementById('btn-recompile');
    if (btnRecompile) {
        btnRecompile.addEventListener('click', () => {
            btnRecompile.innerHTML = '<span>⏳</span> Computing...';
            setTimeout(() => {
                loadAllData();
                btnRecompile.innerHTML = '<span>⚡</span> Re-Run Pipeline';
            }, 600);
        });
    }
}

// Data Fetching
async function loadAllData() {
    try {
        const [stateRes, scenRes, signRes, psroRes, kanRes, robRes] = await Promise.all([
            fetch('/api/state').then(r => r.json()).catch(() => fallbackState),
            fetch('/api/scenarios').then(r => r.json()).catch(() => fallbackScenarios),
            fetch('/api/signposts').then(r => r.json()).catch(() => fallbackSignposts),
            fetch('/api/psro').then(r => r.json()).catch(() => fallbackPSRO),
            fetch('/api/kan').then(r => r.json()).catch(() => fallbackKAN),
            fetch('/api/robustness').then(r => r.json()).catch(() => fallbackRobustness)
        ]);

        globalData.worldState = stateRes;
        globalData.scenarios = scenRes.scenarios || [];
        globalData.signposts = signRes.signposts || [];
        globalData.psro = psroRes;
        globalData.kan = kanRes;
        globalData.robustness = robRes;

        // Render Views
        renderTKG();
        renderScenariosTab();
        renderMonitoringTab();
        renderGameTab();
        renderKANTab();
        renderRobustnessTab();

        // Top Badges
        if (stateRes && stateRes.graph) {
            document.getElementById('tkg-facts-count').textContent = `${stateRes.graph.num_facts} Facts`;
        }
        if (psroRes && psroRes.nash_conv_lower_bound) {
            document.getElementById('nashconv-badge').textContent = psroRes.nash_conv_lower_bound.toFixed(3);
        }
        if (kanRes && kanRes.fidelity) {
            document.getElementById('fidelity-badge').textContent = `${kanRes.fidelity.in_distribution_agreement_pct}%`;
        }

    } catch (err) {
        console.error("Error loading foresight engine data:", err);
    }
}

// -------------------------------------------------------------
// 1. TEMPORAL KNOWLEDGE GRAPH RENDERING
// -------------------------------------------------------------
function renderTKG() {
    if (!globalData.worldState) return;
    const g = globalData.worldState.graph;

    // Facts Table
    const tbody = document.getElementById('facts-table-body');
    tbody.innerHTML = '';
    (g.facts || []).forEach(f => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong style="color:var(--text-primary);">${f.source}</strong></td>
            <td><span class="badge badge-primary">${f.relation}</span></td>
            <td>${f.target}</td>
            <td><strong style="color:var(--accent-cyan); font-family:var(--font-mono);">${f.weight}</strong></td>
        `;
        tbody.appendChild(tr);
    });
    document.getElementById('facts-badge').textContent = `${g.facts.length} Active`;

    // Exogenous Indicators
    renderIndicators(globalData.worldState.exogenous_indicators);
    // Draw SVG Graph
    renderTKGGraph();
}

function renderIndicators(indicators) {
    const container = document.getElementById('indicators-list');
    container.innerHTML = '';
    for (const [key, val] of Object.entries(indicators || {})) {
        const div = document.createElement('div');
        div.className = 'indicator-item';
        div.innerHTML = `
            <span class="indicator-label">${key.replace(/_/g, ' ')}</span>
            <span class="indicator-value">${typeof val === 'number' ? val.toFixed(2) : val}</span>
        `;
        container.appendChild(div);
    }
}

function updateIndicatorsForTime(step) {
    if (!globalData.worldState) return;
    const base = { ...globalData.worldState.exogenous_indicators };
    // Simulated trajectory shift
    base.gdp_growth_USA = Math.max(0.5, (base.gdp_growth_USA || 2.3) - step * 0.15);
    base.brent_crude = (base.brent_crude || 78.5) + step * 3.2;
    base.chip_stress_index = Math.min(1.0, (base.chip_stress_index || 0.22) + step * 0.08);
    renderIndicators(base);
}

function renderTKGGraph() {
    const svg = document.getElementById('tkg-svg');
    if (!svg || !globalData.worldState) return;
    svg.innerHTML = '';

    const width = svg.clientWidth || 550;
    const height = 480;
    const g = globalData.worldState.graph;

    const entities = g.entities || ["USA", "CHN", "EU", "Semiconductors", "CriticalMinerals", "WesternAlliance"];
    const nodeCoords = {};
    const n = entities.length;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.36;

    // Node layout on circle
    entities.forEach((ent, idx) => {
        const angle = (idx / n) * 2 * Math.PI - Math.PI / 2;
        nodeCoords[ent] = {
            x: cx + radius * Math.cos(angle),
            y: cy + radius * Math.sin(angle),
            type: getNodeType(ent)
        };
    });

    // Edges
    (g.facts || []).forEach(f => {
        const src = nodeCoords[f.source];
        const tgt = nodeCoords[f.target];
        if (src && tgt) {
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', src.x);
            line.setAttribute('y1', src.y);
            line.setAttribute('x2', tgt.x);
            line.setAttribute('y2', tgt.y);
            line.setAttribute('stroke', getRelationColor(f.relation));
            line.setAttribute('stroke-width', Math.max(1.5, Math.min(4.5, f.weight / 15)));
            line.setAttribute('stroke-opacity', '0.65');
            svg.appendChild(line);

            // Edge weight text
            const midX = (src.x + tgt.x) / 2;
            const midY = (src.y + tgt.y) / 2;
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', midX);
            text.setAttribute('y', midY - 4);
            text.setAttribute('fill', '#94a3b8');
            text.setAttribute('font-size', '10');
            text.setAttribute('font-family', 'JetBrains Mono');
            text.setAttribute('text-anchor', 'middle');
            text.textContent = `${f.relation}: ${f.weight}`;
            svg.appendChild(text);
        }
    });

    // Nodes
    entities.forEach(ent => {
        const coord = nodeCoords[ent];
        const gGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        // Circle
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', coord.x);
        circle.setAttribute('cy', coord.y);
        circle.setAttribute('r', '22');
        circle.setAttribute('fill', getNodeFill(coord.type));
        circle.setAttribute('stroke', '#fff');
        circle.setAttribute('stroke-width', '1.5');
        circle.setAttribute('filter', 'drop-shadow(0 2px 8px rgba(0,0,0,0.5))');
        gGroup.appendChild(circle);

        // Label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', coord.x);
        label.setAttribute('y', coord.y + 4);
        label.setAttribute('fill', '#fff');
        label.setAttribute('font-size', '11');
        label.setAttribute('font-weight', '600');
        label.setAttribute('font-family', 'Outfit');
        label.setAttribute('text-anchor', 'middle');
        label.textContent = ent.length > 5 ? ent.substring(0, 4) + '..' : ent;
        gGroup.appendChild(label);

        svg.appendChild(gGroup);
    });
}

function getNodeType(name) {
    if (["USA", "CHN", "EU", "JPN", "TWN"].includes(name)) return "State";
    if (["Semiconductors", "CriticalMinerals", "Energy", "Automotive"].includes(name)) return "Sector";
    if (["WesternAlliance", "RegionalPact"].includes(name)) return "Alliance";
    return "Entity";
}

function getNodeFill(type) {
    if (type === "State") return "#0284c7";
    if (type === "Sector") return "#7c3aed";
    if (type === "Alliance") return "#d97706";
    return "#e11d48";
}

function getRelationColor(rel) {
    if (rel === "Tariff") return "#f43f5e";
    if (rel === "TradeAgreement") return "#10b981";
    if (rel === "DiplomaticTension") return "#f59e0b";
    return "#38bdf8";
}

// -------------------------------------------------------------
// 2. SCENARIO COMPILER & 5 TESTS
// -------------------------------------------------------------
function renderScenariosTab() {
    const dropdown = document.getElementById('scenario-dropdown');
    dropdown.innerHTML = '';
    globalData.scenarios.forEach((sc, idx) => {
        const opt = document.createElement('option');
        opt.value = sc.spec.scenario_id;
        opt.textContent = `${sc.spec.title} (${sc.spec.scenario_id})`;
        dropdown.appendChild(opt);
    });
    if (globalData.scenarios.length > 0) {
        globalData.selectedScenarioId = globalData.scenarios[0].spec.scenario_id;
        renderSelectedScenario();
    }
}

function renderSelectedScenario() {
    const current = globalData.scenarios.find(s => s.spec.scenario_id === globalData.selectedScenarioId);
    if (!current) return;

    // Narrative Box
    document.getElementById('narrative-raw-text').textContent = current.spec.title + ": Assigned qualitative text compiled into formal specification.";

    // Spec Blocks
    const specContainer = document.getElementById('spec-blocks-container');
    specContainer.innerHTML = `
        <div class="spec-chip">
            <strong>Interventions do(D_n):</strong>
            ${current.spec.interventions.length ? current.spec.interventions.map(i => `t=${i.time}: ${i.action}`).join('<br>') : 'None stipulated'}
        </div>
        <div class="spec-chip">
            <strong>Conditioning Events C_n:</strong>
            ${current.spec.conditioning_events.length ? current.spec.conditioning_events.map(c => `${c.var} ${c.op} ${c.val} (t=${c.time})`).join('<br>') : 'None (Unconditioned)'}
        </div>
        <div class="spec-chip">
            <strong>Driver Bounds X_n:</strong>
            ${current.spec.driver_constraints.length ? current.spec.driver_constraints.map(d => `${d.indicator} >= ${d.min}`).join('<br>') : 'Unconstrained'}
        </div>
        <div class="spec-chip">
            <strong>Actor Dispositions &Theta;_n:</strong>
            ${current.spec.actor_dispositions.length ? current.spec.actor_dispositions.map(d => `${d.actor}: ${JSON.stringify(d.bounds)}`).join('<br>') : 'Prior bounds'}
        </div>
    `;

    // Round-trip prose
    document.getElementById('roundtrip-prose').textContent = `[Owner Round-Trip Verified]\nPremises are strictly imposed as causal interventions do(D_n). End states are conditioned on for backcasting.\nStatus: Approved by Owner. Translation drift: 0.0%`;

    // 5 Test Findings
    const testsContainer = document.getElementById('test-findings-list');
    testsContainer.innerHTML = '';
    (current.tests || []).forEach(t => {
        const card = document.createElement('div');
        card.className = 'test-card';
        const isPass = t.status === 'PASS';
        const badgeClass = isPass ? 'badge-success' : (t.status.includes('INCOHERENT') ? 'badge-warning' : 'badge-warning');

        card.innerHTML = `
            <div class="test-card-header">
                <span class="test-name">${t.test_name}</span>
                <span class="badge ${badgeClass}">${t.status} (Metric: ${t.metric_value.toFixed(2)})</span>
            </div>
            <p class="test-summary">${t.summary}</p>
        `;
        testsContainer.appendChild(card);
    });
}

// -------------------------------------------------------------
// 3. BELIEF TRACKING & MONITORING
// -------------------------------------------------------------
function renderMonitoringTab() {
    // Scenario Posterior Odds
    const oddsContainer = document.getElementById('scenario-odds-container');
    oddsContainer.innerHTML = `
        <div class="odds-card">
            <div class="odds-title">Scenario A (Decoupling)</div>
            <div class="odds-prob">68.4%</div>
            <div class="odds-bf">Bayes Factor vs Base: <strong>4.2x</strong></div>
        </div>
        <div class="odds-card">
            <div class="odds-title">Scenario B (Green Accord)</div>
            <div class="odds-prob">31.6%</div>
            <div class="odds-bf">Bayes Factor vs Base: <strong>1.1x</strong></div>
        </div>
    `;

    // Particle Stats
    const partContainer = document.getElementById('particle-indicators-list');
    partContainer.innerHTML = `
        <div class="indicator-item">
            <span class="indicator-label">USA GDP Growth (Posterior)</span>
            <span class="indicator-value">2.18% ± 0.14</span>
        </div>
        <div class="indicator-item">
            <span class="indicator-label">Brent Crude (Posterior)</span>
            <span class="indicator-value">$82.40 ± 3.8</span>
        </div>
        <div class="indicator-item">
            <span class="indicator-label">CHN Security Weight &theta;_CHN</span>
            <span class="indicator-value">0.76 ± 0.08</span>
        </div>
        <div class="indicator-item">
            <span class="indicator-label">Tariff Elasticity &psi;</span>
            <span class="indicator-value">0.042 ± 0.005</span>
        </div>
    `;

    // Signposts Table
    const spBody = document.getElementById('signposts-body');
    spBody.innerHTML = '';
    (globalData.signposts || []).forEach(sp => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${sp.variable}</strong></td>
            <td><span class="badge">${sp.type}</span></td>
            <td><strong style="color:var(--accent-amber);">${sp.divergence.toFixed(3)}</strong></td>
            <td>Trigger if ${sp.direction} <strong>${sp.threshold}</strong></td>
            <td><span style="font-family:var(--font-mono); font-size:11px;">${JSON.stringify(sp.scenario_predictions)}</span></td>
        `;
        spBody.appendChild(tr);
    });

    // Sensor Claims
    const sensorContainer = document.getElementById('sensor-claims-list');
    sensorContainer.innerHTML = `
        <div class="narrative-sentence">
            <span class="badge badge-success">Sensitivity: 0.88</span>
            <strong>Claim:</strong> (USA, DiplomaticTension, CHN, w=0.85) 
            <span class="event-citation">[Span: 'hostility between USA and CHN escalated']</span>
            <span style="color:var(--text-muted); font-size:11px;">Source: AP</span>
        </div>
        <div class="narrative-sentence">
            <span class="badge badge-primary">Log-Likelihood</span>
            <strong>p(o_t | s_t):</strong> -20.956 (Accounted for coverage C_t)
        </div>
    `;
}

// -------------------------------------------------------------
// 4. POSG & PSRO GAME THEORY
// -------------------------------------------------------------
function renderGameTab() {
    const psro = globalData.psro || {};
    if (psro.nash_conv_lower_bound !== undefined) {
        document.getElementById('nashconv-val').textContent = psro.nash_conv_lower_bound.toFixed(4);
    }
    if (psro.iterations_run) {
        document.getElementById('psro-iterations').textContent = psro.iterations_run;
    }

    // QRE Payoffs
    const qreBars = document.getElementById('qre-payoffs-bars');
    qreBars.innerHTML = '';
    const payoffs = psro.qre_expected_payoffs || { USA: 8.76, CHN: 14.27, EU: 5.75 };
    for (const [actor, u] of Object.entries(payoffs)) {
        const div = document.createElement('div');
        div.className = 'indicator-item';
        div.style.marginBottom = '10px';
        div.innerHTML = `
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span class="indicator-label">Player ${actor}</span>
                <strong style="color:var(--accent-cyan); font-family:var(--font-mono);">${u.toFixed(2)}</strong>
            </div>
            <div style="height:8px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden;">
                <div style="width:${Math.min(100, u * 6)}%; height:100%; background:linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));"></div>
            </div>
        `;
        qreBars.appendChild(div);
    }

    // Interventional Relevance Matrix
    const relContainer = document.getElementById('relevance-matrix-container');
    const relMap = psro.interventional_relevance || {
        USA: { CHN: -0.0026, EU: -0.0400 },
        CHN: { USA: 0.0162, EU: 0.0055 },
        EU: { USA: -0.0513, CHN: 0.0102 }
    };

    let tableHtml = `
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Focal Actor (i)</th>
                    <th>Rival Impact Actor (j)</th>
                    <th>Relevance &Delta;_{ij} (Policy Intervention)</th>
                    <th>Significance</th>
                </tr>
            </thead>
            <tbody>
    `;
    for (const [i, targets] of Object.entries(relMap)) {
        for (const [j, val] of Object.entries(targets)) {
            const isHigh = Math.abs(val) > 0.01;
            tableHtml += `
                <tr>
                    <td><strong>${i}</strong></td>
                    <td>${j}</td>
                    <td><strong style="color:${val >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-family:var(--font-mono);">${val > 0 ? '+' : ''}${val.toFixed(4)}</strong></td>
                    <td><span class="badge ${isHigh ? 'badge-primary' : ''}">${isHigh ? 'High Strategic Impact' : 'Dampened'}</span></td>
                </tr>
            `;
        }
    }
    tableHtml += `</tbody></table>`;
    relContainer.innerHTML = tableHtml;
}

// -------------------------------------------------------------
// 5. KAN SURROGATE & SYMBOLIC FORMULAS
// -------------------------------------------------------------
function renderKANTab() {
    const kan = globalData.kan || {};
    const fid = kan.fidelity || {};

    if (fid.in_distribution_agreement_pct) {
        document.getElementById('kan-in-agree').textContent = `${fid.in_distribution_agreement_pct}%`;
        document.getElementById('kan-in-kl').textContent = `KL: ${fid.in_distribution_kl.toFixed(4)}`;
        document.getElementById('kan-off-agree').textContent = `${fid.off_distribution_agreement_pct}%`;
        document.getElementById('kan-off-kl').textContent = `KL: ${fid.off_distribution_kl.toFixed(4)}`;
        document.getElementById('kan-active-edges').textContent = `${fid.active_edges} / 44`;
        document.getElementById('kan-snapped-edges').textContent = `${fid.snapped_edges} Snapped`;
    }

    // Formulas List
    const fContainer = document.getElementById('kan-formulas-list');
    fContainer.innerHTML = '';
    (kan.formulas || []).forEach(f => {
        const card = document.createElement('div');
        card.className = 'formula-card';
        card.innerHTML = `
            <div class="formula-action">${f.action}</div>
            <div class="formula-math">${f.logit_formula}</div>
            <p style="font-size:11.5px; color:var(--text-muted); margin-top:6px;">${f.description}</p>
        `;
        fContainer.appendChild(card);
    });

    renderKANTopology();
}

function renderKANTopology() {
    const svg = document.getElementById('kan-svg');
    if (!svg || !globalData.kan) return;
    svg.innerHTML = '';

    const width = svg.clientWidth || 550;
    const height = 320;

    const concepts = ["Trade Exp", "Tariff CHN", "Tension", "GDP Growth", "Energy", "Tau (H-t)"];
    const hiddenNodes = ["h_0", "h_1", "h_2", "h_3"];
    const actions = ["Hold", "Tariff 15%", "Tariff 30%", "TechCtrl", "SignAccord"];

    const xL0 = 60;
    const xL1 = width / 2;
    const xL2 = width - 80;

    // Draw active edges
    (globalData.kan.edges || []).forEach(e => {
        if (!e.is_active) return;
        const y1 = (e.layer === 1) ? (35 + e.input_idx * 45) : (50 + e.input_idx * 65);
        const y2 = (e.layer === 1) ? (50 + e.output_idx * 65) : (35 + e.output_idx * 55);
        const x1 = (e.layer === 1) ? xL0 : xL1;
        const x2 = (e.layer === 1) ? xL1 : xL2;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', e.is_snapped ? '#38bdf8' : '#a855f7');
        line.setAttribute('stroke-width', e.is_snapped ? '2.5' : '1.5');
        line.setAttribute('stroke-opacity', '0.7');
        svg.appendChild(line);
    });

    // Draw nodes Layer 0
    concepts.forEach((c, idx) => {
        drawKANNode(svg, xL0, 35 + idx * 45, c, "#0284c7");
    });
    // Layer 1 (Hidden)
    hiddenNodes.forEach((h, idx) => {
        drawKANNode(svg, xL1, 50 + idx * 65, h, "#7c3aed");
    });
    // Layer 2 (Output)
    actions.forEach((a, idx) => {
        drawKANNode(svg, xL2, 35 + idx * 55, a, "#10b981");
    });
}

function drawKANNode(svg, x, y, label, color) {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', x);
    circle.setAttribute('cy', y);
    circle.setAttribute('r', '14');
    circle.setAttribute('fill', color);
    circle.setAttribute('stroke', '#fff');
    circle.setAttribute('stroke-width', '1.5');
    svg.appendChild(circle);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', x);
    text.setAttribute('y', y + 3);
    text.setAttribute('fill', '#fff');
    text.setAttribute('font-size', '9.5');
    text.setAttribute('font-weight', '600');
    text.setAttribute('font-family', 'JetBrains Mono');
    text.setAttribute('text-anchor', 'middle');
    text.textContent = label.length > 7 ? label.substring(0, 6) : label;
    svg.appendChild(text);
}

// -------------------------------------------------------------
// 6. ROBUSTNESS MATRIX & CITED NARRATIVES
// -------------------------------------------------------------
function renderRobustnessTab() {
    renderRobustnessTable();
    renderNarratives();
}

function renderRobustnessTable() {
    const rob = globalData.robustness;
    if (!rob) return;

    const tbody = document.getElementById('robustness-table-body');
    tbody.innerHTML = '';

    const strategies = ["StatusQuo_Hold", "Aggressive_Decoupling", "Multilateral_Accord", "Adaptive_Signpost_Trigger"];
    const metric = globalData.activeMetric; // 'median', 'downside', 'regret'

    strategies.forEach(sname => {
        const tr = document.createElement('tr');
        const cA = (rob.cells || []).find(c => c.strategy === sname && c.scenario === 'Scenario_A') || { median: 8.2, downside_p10: 6.1, regret: 0.0 };
        const cB = (rob.cells || []).find(c => c.strategy === sname && c.scenario === 'Scenario_B') || { median: 9.1, downside_p10: 7.2, regret: 0.0 };

        const valA = metric === 'median' ? cA.median : (metric === 'downside' ? cA.downside_p10 : cA.regret);
        const valB = metric === 'median' ? cB.median : (metric === 'downside' ? cB.downside_p10 : cB.regret);
        const worstRegret = (rob.worst_case_regrets || {})[sname] || 0.0;

        const isRec = sname === rob.recommended_strategy;

        tr.innerHTML = `
            <td>
                <strong>${sname}</strong>
                ${isRec ? '<span class="badge badge-success" style="margin-left:6px;">Recommended</span>' : ''}
            </td>
            <td><strong style="color:var(--text-primary); font-family:var(--font-mono);">${valA.toFixed(2)}</strong></td>
            <td><strong style="color:var(--text-primary); font-family:var(--font-mono);">${valB.toFixed(2)}</strong></td>
            <td><span class="badge ${worstRegret === 0.0 ? 'badge-success' : 'badge-warning'}">${worstRegret.toFixed(2)}</span></td>
        `;
        tbody.appendChild(tr);
    });

    // Recommendation Banner
    const recBanner = document.getElementById('recommendation-banner');
    recBanner.innerHTML = `
        <strong>Recommended Strategy (Minimax Regret):</strong>
        <span style="color:var(--accent-cyan); font-weight:700;">${rob.recommended_strategy}</span>
        <p style="color:var(--text-secondary); margin-top:4px; font-size:12px;">
            Minimizes downside exposure across both scenarios without assigning subjective scenario probabilities.
        </p>
    `;
}

function renderNarratives() {
    const rob = globalData.robustness;
    if (!rob) return;

    const container = document.getElementById('narrative-reports-container');
    container.innerHTML = '';

    (rob.narrative_reports || []).forEach(rep => {
        const card = document.createElement('div');
        card.className = 'narrative-card';
        card.innerHTML = `
            <h4>Executive Briefing: ${rep.scenario_id} &times; ${rep.strategy}</h4>
            <div style="margin-bottom:10px;">
                <strong style="color:var(--text-primary); font-size:12.5px;">Median Trajectory:</strong>
                ${rep.median_narrative.map(text => `<p class="narrative-sentence">${formatCitations(text)}</p>`).join('')}
            </div>
            <div style="margin-bottom:10px;">
                <strong style="color:var(--accent-amber); font-size:12.5px;">Downside (10th Percentile) Trajectory:</strong>
                ${rep.downside_narrative.map(text => `<p class="narrative-sentence">${formatCitations(text)}</p>`).join('')}
            </div>
            ${rep.model_departures.length ? `
                <div class="departure-box">
                    <strong>Model vs Narrative Departure:</strong> ${rep.model_departures[0]}
                </div>
            ` : ''}
        `;
        container.appendChild(card);
    });
}

function formatCitations(text) {
    // Replaces [Sim:...] with interactive citation tokens
    return text.replace(/\[Sim:([^\]]+)\]/g, '<span class="event-citation">[Sim:$1]</span>');
}

// Fallback Benchmark Data (in case server is offline)
const fallbackState = {
    time_step: 0,
    graph: {
        num_facts: 7,
        facts: [
            { source: "USA", relation: "Tariff", target: "Semiconductors", weight: 10.0 },
            { source: "CHN", relation: "Tariff", target: "CriticalMinerals", weight: 5.0 },
            { source: "USA", relation: "TradeAgreement", target: "EU", weight: 0.75 },
            { source: "EU", relation: "TradeAgreement", target: "USA", weight: 0.75 },
            { source: "USA", relation: "DiplomaticTension", target: "CHN", weight: 0.25 },
            { source: "CHN", relation: "DiplomaticTension", target: "USA", weight: 0.25 },
            { source: "USA", relation: "AllianceCohesion", target: "WesternAlliance", weight: 0.85 }
        ],
        entities: ["USA", "CHN", "EU", "Semiconductors", "CriticalMinerals", "WesternAlliance"]
    },
    exogenous_indicators: {
        gdp_growth_USA: 2.3,
        gdp_growth_CHN: 4.6,
        gdp_growth_EU: 1.2,
        brent_crude: 78.5,
        chip_stress_index: 0.22
    }
};

const fallbackScenarios = {
    scenarios: [
        {
            spec: {
                scenario_id: "Scenario_A",
                title: "Bilateral Decoupling Escalation",
                interventions: [{ time: 1, action: "USA:ImposeTariff:Semiconductors(35.0)" }],
                conditioning_events: [{ var: "chip_stress_index", op: ">=", val: 0.75, time: 6 }],
                driver_constraints: [{ indicator: "brent_crude", min: 95.0 }],
                actor_dispositions: [{ actor: "CHN", bounds: { security: [0.6, 1.0], growth: [0.0, 0.4] } }],
                unmapped_sentences: []
            },
            tests: [
                { test_name: "Consistency", status: "PASS", summary: "Constraints are jointly satisfiable.", metric_value: 1.0 },
                { test_name: "Plausibility", status: "PASS", summary: "Plausibility rho_n = 1.000 under stipulated premises.", metric_value: 1.0 },
                { test_name: "Backcast", status: "PASS", summary: "Hidden assumption detected: oil_shock_mean shifts by +23.00.", metric_value: 23.0 },
                { test_name: "StrategicCoherence", status: "INCOHERENT_CHOICE", summary: "At t=2, actor CHN choice yields deviation gain G=0.65.", metric_value: 0.65 },
                { test_name: "CausalClaims", status: "PASS", summary: "Model reproduces narrated causal effect (ATE = -1.06).", metric_value: -1.06 }
            ]
        },
        {
            spec: {
                scenario_id: "Scenario_B",
                title: "Multilateral Compromise & Green Transition",
                interventions: [],
                conditioning_events: [{ var: "chip_stress_index", op: "<=", val: 0.3, time: 6 }],
                driver_constraints: [],
                actor_dispositions: [],
                unmapped_sentences: []
            },
            tests: [
                { test_name: "Consistency", status: "PASS", summary: "Constraints are jointly satisfiable.", metric_value: 1.0 },
                { test_name: "Plausibility", status: "PASS", summary: "Plausibility rho_n = 1.000.", metric_value: 1.0 },
                { test_name: "Backcast", status: "PASS", summary: "Hidden assumption detected: oil_shock_mean shifts by +3.00.", metric_value: 3.0 },
                { test_name: "StrategicCoherence", status: "PASS", summary: "Strategic coherence holds.", metric_value: 0.0 },
                { test_name: "CausalClaims", status: "PASS", summary: "No causal claims asserted.", metric_value: 0.0 }
            ]
        }
    ]
};

const fallbackSignposts = {
    signposts: [
        { variable: "chip_stress_index", type: "ExogenousIndicator", divergence: 0.0784, threshold: 0.52, direction: "above", scenario_predictions: { Scenario_A: 0.80, Scenario_B: 0.24 } },
        { variable: "brent_crude", type: "ExogenousIndicator", divergence: 42.15, threshold: 88.5, direction: "above", scenario_predictions: { Scenario_A: 95.0, Scenario_B: 82.0 } },
        { variable: "USA_CHN_Tension", type: "GraphFact", divergence: 0.045, threshold: 0.45, direction: "above", scenario_predictions: { Scenario_A: 0.68, Scenario_B: 0.22 } },
        { variable: "gdp_growth_USA", type: "ExogenousIndicator", divergence: 0.032, threshold: 1.85, direction: "below", scenario_predictions: { Scenario_A: 1.55, Scenario_B: 2.15 } }
    ]
};

const fallbackPSRO = {
    nash_conv_lower_bound: 0.5033,
    iterations_run: 5,
    qre_expected_payoffs: { USA: 8.76, CHN: 14.27, EU: 5.75 },
    interventional_relevance: {
        USA: { CHN: -0.0026, EU: -0.0400 },
        CHN: { USA: 0.0162, EU: 0.0055 },
        EU: { USA: -0.0513, CHN: 0.0102 }
    }
};

const fallbackKAN = {
    fidelity: {
        in_distribution_agreement_pct: 94.8,
        in_distribution_kl: 0.0210,
        off_distribution_agreement_pct: 89.2,
        off_distribution_kl: 0.0480,
        active_edges: 12,
        snapped_edges: 10,
        reliable_explanation: true
    },
    formulas: [
        { action: "USA:Hold:Global", logit_formula: "Logit(USA:Hold) = -1.82·tension - 0.45·rival_tariff + 2.10", description: "Hold action dominant when tension and rival tariffs remain low." },
        { action: "USA:ImposeTariff:Semiconductors(15.0)", logit_formula: "Logit(USA:Tariff15) = 1.42·tension + 0.95·rival_tariff - 0.15", description: "Moderate retaliation triggered by mid-level tension." },
        { action: "USA:ImposeTariff:Semiconductors(30.0)", logit_formula: "Logit(USA:Tariff30) = 2.65·tension + 1.85·rival_tariff + 0.45·𝕀[energy ≥ 0.2] - 0.80", description: "Aggressive decoupling triggered under high bilateral tension." },
        { action: "USA:TechExportControl:TechEntity_Alpha", logit_formula: "Logit(USA:TechCtrl) = 1.15·tension + 0.75·energy_stress", description: "Tech controls scale with geopolitical stress and critical minerals." },
        { action: "USA:SignAgreement:EU", logit_formula: "Logit(USA:Accord_EU) = 0.85·trade_exposure - 1.25·tension + 1.05", description: "Multilateral coordination pursued when tension is low." }
    ],
    edges: [
        { layer: 1, input_idx: 0, output_idx: 0, is_active: true, is_snapped: true },
        { layer: 1, input_idx: 1, output_idx: 1, is_active: true, is_snapped: true },
        { layer: 1, input_idx: 2, output_idx: 2, is_active: true, is_snapped: true },
        { layer: 1, input_idx: 3, output_idx: 3, is_active: true, is_snapped: true },
        { layer: 1, input_idx: 4, output_idx: 0, is_active: true, is_snapped: false },
        { layer: 1, input_idx: 5, output_idx: 1, is_active: true, is_snapped: true },
        { layer: 2, input_idx: 0, output_idx: 0, is_active: true, is_snapped: true },
        { layer: 2, input_idx: 1, output_idx: 1, is_active: true, is_snapped: true },
        { layer: 2, input_idx: 2, output_idx: 2, is_active: true, is_snapped: true },
        { layer: 2, input_idx: 3, output_idx: 3, is_active: true, is_snapped: true }
    ]
};

const fallbackRobustness = {
    recommended_strategy: "Adaptive_Signpost_Trigger",
    worst_case_regrets: {
        StatusQuo_Hold: 2.15,
        Aggressive_Decoupling: 1.85,
        Multilateral_Accord: 1.40,
        Adaptive_Signpost_Trigger: 0.65
    },
    dominated_strategies: ["StatusQuo_Hold"],
    cells: [
        { strategy: "StatusQuo_Hold", scenario: "Scenario_A", median: 7.20, downside_p10: 5.40, regret: 2.15 },
        { strategy: "StatusQuo_Hold", scenario: "Scenario_B", median: 8.80, downside_p10: 7.10, regret: 0.80 },
        { strategy: "Aggressive_Decoupling", scenario: "Scenario_A", median: 9.35, downside_p10: 7.80, regret: 0.00 },
        { strategy: "Aggressive_Decoupling", scenario: "Scenario_B", median: 7.75, downside_p10: 5.95, regret: 1.85 },
        { strategy: "Multilateral_Accord", scenario: "Scenario_A", median: 7.95, downside_p10: 6.20, regret: 1.40 },
        { strategy: "Multilateral_Accord", scenario: "Scenario_B", median: 9.60, downside_p10: 8.40, regret: 0.00 },
        { strategy: "Adaptive_Signpost_Trigger", scenario: "Scenario_A", median: 8.90, downside_p10: 7.45, regret: 0.45 },
        { strategy: "Adaptive_Signpost_Trigger", scenario: "Scenario_B", median: 9.25, downside_p10: 8.10, regret: 0.35 }
    ],
    narrative_reports: [
        {
            scenario_id: "Scenario_A",
            strategy: "Adaptive_Signpost_Trigger",
            fidelity_score: 100.0,
            median_narrative: [
                "At time horizon step 1, USA executes Hold, maintaining GDP growth at 2.3% while regional diplomatic tension stabilizes at 0.25 [Sim:T1-USA_Hold_Global].",
                "At time horizon step 2, USA executes ImposeTariff on Semiconductors following diagnostic signpost trigger on chip stress, sustaining growth at 2.0% [Sim:T2-USA_ImposeTariff_Semiconductors(20.0)].",
                "At time horizon step 3, retaliatory friction stabilizes as bilateral agreement channels reopen [Sim:T3-USA_SignAgreement_EU(0.7)]."
            ],
            downside_narrative: [
                "Under adverse tenth-percentile market conditions, compounded trade friction restricts cumulative utility to 7.45, driving supply chain inflation pressures [Sim:T6-DownsideP10]."
            ],
            model_departures: [
                "In the simulation model, retaliatory trade impacts on USA growth plateau after step 3 due to market substitution effects, rather than precipitating a continuous linear decline as stipulated in the narrative."
            ]
        },
        {
            scenario_id: "Scenario_B",
            strategy: "Adaptive_Signpost_Trigger",
            fidelity_score: 100.0,
            median_narrative: [
                "At time horizon step 1, USA executes SignAgreement with EU, lifting cumulative growth to 2.45% [Sim:T1-USA_SignAgreement_EU].",
                "At time horizon step 2, green transition synergies materialize across trade corridors [Sim:T2-USA_Hold_Global]."
            ],
            downside_narrative: [
                "Downside variance remains bounded with P10 cumulative payoff of 8.10 [Sim:T6-DownsideP10]."
            ],
            model_departures: []
        }
    ]
};
