import { FORMATS, ballsToLabel, phaseOf } from "./formats.js";
import { teamVisual } from "./franchises.js";

document.addEventListener("DOMContentLoaded", () => {
    // Respect prefers-reduced-motion: disable Chart.js's own animation
    // engine (CSS media queries can't reach into <canvas> rendering).
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        Chart.defaults.animation = false;
    }

    // --------------------------------------------------
    // Safe Event Listener Helper
    // --------------------------------------------------
    function on(id, evt, fn) {
        const el = document.getElementById(id);
        if (el) el.addEventListener(evt, fn);
        else console.warn(`Element '${id}' not found, skipping listener for '${evt}'`);
    }

    // --------------------------------------------------
    // Toast Notifications (non-blocking replacement for alert())
    // --------------------------------------------------
    const toastContainer = document.getElementById("toast-container");

    function showToast(message, type = "error", duration = 6000) {
        if (!toastContainer) { console.warn("Toast:", message); return; }

        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.setAttribute("role", "alert");

        const icon = document.createElement("span");
        icon.className = "toast-icon";
        icon.setAttribute("aria-hidden", "true");
        icon.textContent = type === "success" ? "✓" : type === "info" ? "i" : "!";

        const body = document.createElement("span");
        body.className = "toast-body";
        body.textContent = message;

        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "toast-close";
        closeBtn.setAttribute("aria-label", "Dismiss notification");
        closeBtn.textContent = "✕";

        let dismissed = false;
        function dismiss() {
            if (dismissed) return;
            dismissed = true;
            toast.classList.add("toast-out");
            toast.addEventListener("animationend", () => toast.remove(), { once: true });
            setTimeout(() => toast.remove(), 400); // fallback if reduced-motion skips the animation
        }
        closeBtn.addEventListener("click", dismiss);

        toast.appendChild(icon);
        toast.appendChild(body);
        toast.appendChild(closeBtn);
        toastContainer.appendChild(toast);

        if (duration > 0) setTimeout(dismiss, duration);
    }

    // --------------------------------------------------
    // Tab Switching Logic
    // --------------------------------------------------
    const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
    const tabContents = document.querySelectorAll(".tab-content");

    function activateTab(btn) {
        const targetTab = btn.getAttribute("data-tab");

        tabButtons.forEach(b => {
            b.classList.remove("active");
            b.setAttribute("aria-selected", "false");
            b.tabIndex = -1;
        });
        tabContents.forEach(c => c.classList.remove("active"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        btn.tabIndex = 0;
        document.getElementById(`tab-${targetTab}`).classList.add("active");

        // Handle lazy layout updates of charts when tab changes
        if (targetTab === "analytics") {
            loadAnalytics();
        } else if (targetTab === "live") {
            loadLiveGames();
        } else if (targetTab === "betting") {
            loadBetting();
        } else if (targetTab === "eloexplorer") {
            initEloExplorer();
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => activateTab(btn));
    });

    // Arrow-key navigation between tabs (WAI-ARIA tabs pattern, automatic activation)
    const navTabsEl = document.querySelector(".nav-tabs");
    if (navTabsEl) {
        navTabsEl.addEventListener("keydown", (e) => {
            const currentIndex = tabButtons.indexOf(document.activeElement);
            if (currentIndex === -1) return;

            let nextIndex = null;
            if (e.key === "ArrowRight" || e.key === "ArrowDown") {
                nextIndex = (currentIndex + 1) % tabButtons.length;
            } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
                nextIndex = (currentIndex - 1 + tabButtons.length) % tabButtons.length;
            } else if (e.key === "Home") {
                nextIndex = 0;
            } else if (e.key === "End") {
                nextIndex = tabButtons.length - 1;
            }

            if (nextIndex !== null) {
                e.preventDefault();
                tabButtons[nextIndex].focus();
                activateTab(tabButtons[nextIndex]);
            }
        });
    }

    // --------------------------------------------------
    // SHARED RESULT COMPONENTS (win split, leaderboards)
    // --------------------------------------------------

    // Option B win prediction: one panel where each side's WIDTH encodes its win probability,
    // filled with the team's colour (real franchise brand colour, else derived from the name).
    // a / b are { name, prob (0..1), odds? }. Optional `tie` { prob (0..1), odds? } renders a
    // small badge on the centre divider (used by the Monte Carlo tab).
    function winSplitHTML(a, b, tie) {
        const va = teamVisual(a.name), vb = teamVisual(b.name);
        const pctA = a.prob * 100, pctB = b.prob * 100;
        const aWins = a.prob >= b.prob;
        const seg = (t, pct, grad, isWinner, right) => `
            <div class="winsplit-seg ${right ? "right" : ""}" style="flex:${Math.max(pct, 7)} 1 0; background:${grad};">
                <div>
                    ${isWinner ? '<span class="winsplit-verdict">Projected winner</span><br>' : ""}
                    <span class="winsplit-team">${t.name}</span>
                </div>
                <div>
                    <div class="winsplit-pct">${pct.toFixed(1)}<small>%</small></div>
                    ${t.odds ? `<div class="winsplit-odds">Fair ${t.odds}</div>` : ""}
                </div>
            </div>`;
        // Tie renders as its own middle segment. Its flex-grow is proportional to the tie
        // probability, but a CSS min-width keeps it legible even when that probability is tiny.
        const tieSeg = tie ? `
            <div class="winsplit-tie-seg" style="flex:${(tie.prob * 100).toFixed(2)} 1 0;" title="${tie.odds ? `Fair ${tie.odds}` : ""}">
                <span class="wt-label">Tie</span>
                <span class="wt-val">${(tie.prob * 100).toFixed(1)}%</span>
            </div>`
            : `<div class="winsplit-divider"></div>`;
        return `<div class="winsplit">
            ${seg(a, pctA, va.gradient, aWins, false)}
            ${tieSeg}
            ${seg(b, pctB, vb.gradient, !aWins, true)}
        </div>`;
    }

    function renderWinSplit(containerId, a, b, tie) {
        const el = document.getElementById(containerId);
        if (el) el.innerHTML = winSplitHTML(a, b, tie);
    }

    // Leaderboard (top-batsman / props): rank, name, prob, odds, SE + a mini likelihood bar,
    // with a team-coloured left accent. `obj` is name -> {p_top|p_hit, odds, mc_se, ...}.
    function renderLeaderboard(obj, prop, accentColor) {
        if (!obj) return "<div style='color:#94a3b8; padding:10px 4px;'>Not available for this innings.</div>";
        const rows = Object.entries(obj)
            .map(([name, stats]) => ({ name, ...stats }))
            .sort((x, y) => y[prop] - x[prop])
            .slice(0, 5);
        const maxP = rows.length ? Math.max(...rows.map(r => r[prop])) || 1 : 1;
        const accent = accentColor || "var(--primary-grad)";
        return rows.map((p, i) => `
            <div class="leaderboard-item">
                <span class="lb-accent" style="background:${accent}"></span>
                <span class="leaderboard-rank">${i + 1}</span>
                <span class="leaderboard-name">${p.name}</span>
                <span class="leaderboard-prob">${(p[prop] * 100).toFixed(1)}%</span>
                <span class="leaderboard-odds">${p.odds}</span>
                <span class="leaderboard-se">±${(p.mc_se * 100).toFixed(1)}% SE</span>
                <span class="lb-mini"><i style="width:${Math.max(4, (p[prop] / maxP) * 100)}%"></i></span>
            </div>`).join("");
    }

    // Three-tile projected-score distribution (P5 / Median / P95), median emphasised.
    function statTilesHTML(dist, subLabel) {
        return `
            <div class="stat-tile"><div class="st-label">P5</div><div class="st-val">${dist.p5_score}</div><div class="st-sub">floor</div></div>
            <div class="stat-tile hi"><div class="st-label">Median</div><div class="st-val">${Math.round(dist.mean_score)}</div><div class="st-sub">${subLabel || ""}</div></div>
            <div class="stat-tile"><div class="st-label">P95</div><div class="st-val">${dist.p95_score}</div><div class="st-sub">ceiling</div></div>`;
    }

    // --------------------------------------------------
    // Chart.js Gauge Config Helper
    // --------------------------------------------------
    function initGaugeChart(canvasId, initialVal = 0) {
        const ctx = document.getElementById(canvasId).getContext("2d");
        return new Chart(ctx, {
            type: "doughnut",
            data: {
                datasets: [{
                    data: [initialVal, 100 - initialVal],
                    backgroundColor: [
                        "#00f2fe",
                        "rgba(255, 255, 255, 0.05)"
                    ],
                    borderWidth: 0,
                    borderRadius: 10
                }]
            },
            options: {
                circumference: 180,
                rotation: 270,
                cutout: "85%",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            }
        });
    }

    function updateGauge(chart, val, displayId, accent = "teal") {
        const displayVal = Math.round(val * 100);
        document.getElementById(displayId).textContent = `${displayVal}%`;
        
        // Update Chart data
        chart.data.datasets[0].data = [displayVal, Math.max(0, 100 - displayVal)];
        
        // Dynamic colors based on win probability
        let color = "#00f2fe"; // Teal
        if (accent === "teal") {
            if (displayVal > 70) color = "#00f2fe";
            else if (displayVal > 40) color = "#4facfe";
            else color = "#f43f5e"; // Pink/Red for low probability
        } else {
            color = "#a855f7"; // Purple for scenarios
        }

        chart.data.datasets[0].backgroundColor = [color, "rgba(255, 255, 255, 0.05)"];
        chart.update();
    }

    const simGauge = initGaugeChart("sim-gauge-chart", 50);

    // Initialize metrics displays
    updateGauge(simGauge, 0.50, "sim-prob-display");

    // --------------------------------------------------
    // TAB 0: LIVE SIMULATOR LOGIC
    // --------------------------------------------------
    let metadataCache = null;

    async function fetchMetadata() {
        if (metadataCache) return metadataCache;
        try {
            const resp = await fetch("/api/metadata");
            metadataCache = await resp.json();
            populateSimulationDropdowns();
            return metadataCache;
        } catch (e) {
            console.error("Error fetching metadata:", e);
        }
    }

    function venuesForFormat(formatVal) {
        if (!metadataCache) return {};
        return metadataCache[`${formatVal}_venues`] || metadataCache.t20_venues || metadataCache.venues || {};
    }

    // Generic {fmt}_teams lookup -- mirrors venuesForFormat above. Any format present in
    // /api/metadata's response (t20/od/hnd/t20w, and future odw/hndw) works with zero changes
    // here; unrecognised formatVal falls back to t20_teams, matching the backend's
    // MODELS.get(format_type, MODELS["t20"]) fallback convention.
    function teamsForFormat(formatVal) {
        if (!metadataCache) return {};
        return metadataCache[`${formatVal}_teams`] || metadataCache.t20_teams || {};
    }

    function populateSimulationDropdowns() {
        if (!metadataCache) return;

        const formatVal = document.querySelector('input[name="sim-format"]:checked')?.value || "t20";
        const fmt = FORMATS[formatVal];
        const teamsDict = teamsForFormat(formatVal);
        const venuesDict = venuesForFormat(formatVal);

        const simSourceSel = document.getElementById("sim-source");
        simSourceSel.value = ratingSource;
        refreshSourceSelectOptions(simSourceSel, formatVal);

        const batSelect = document.getElementById("sim-batting-team");
        const bowlSelect = document.getElementById("sim-bowling-team");
        const venueSelect = document.getElementById("sim-venue");

        const curBat = batSelect.value;
        const curBowl = bowlSelect.value;
        const curVenue = venueSelect.value;

        batSelect.innerHTML = '<option value="">-- Select Batting Team --</option>';
        bowlSelect.innerHTML = '<option value="">-- Select Bowling Team --</option>';
        venueSelect.innerHTML = '<option value="">-- Select Venue --</option>';

        for (const [team, elo] of Object.entries(teamsDict || {})) {
            batSelect.add(new Option(`${team} (${elo} Elo)`, team));
            bowlSelect.add(new Option(`${team} (${elo} Elo)`, team));
        }

        for (const [venue, avg] of Object.entries(venuesDict || {})) {
            venueSelect.add(new Option(`${venue} (Avg: ${avg})`, venue));
        }

        if (curBat && teamsDict && teamsDict[curBat]) batSelect.value = curBat;
        if (curBowl && teamsDict && teamsDict[curBowl]) bowlSelect.value = curBowl;
        if (curVenue && venuesDict && venuesDict[curVenue]) venueSelect.value = curVenue;
    }

    document.querySelectorAll('input[name="sim-format"]').forEach(radio => {
        radio.addEventListener('change', () => {
            const fId = document.querySelector('input[name="sim-format"]:checked').value;

            const fmt = FORMATS[fId];
            document.getElementById("sim-target").value = fmt.parScore;
            
            // Reset to reasonable format default instead of keeping old invalid values
            document.getElementById("sim-runs-slider").value = Math.round(fmt.parScore / 2);
            document.getElementById("sim-balls-slider").value = Math.round(fmt.totalBalls / 2);
            document.getElementById("sim-wickets-slider").value = 3;
            
            updateSimSlidersMax();

            populateSimulationDropdowns();
            runSimulation();
        });
    });

    on("sim-batting-team", "change", (e) => {
        const team = e.target.value;
        const formatVal = document.querySelector('input[name="sim-format"]:checked')?.value || "t20";
        const teamsDict = teamsForFormat(formatVal);
        if (team && teamsDict && teamsDict[team]) {
            document.getElementById("sim-batting-elo").value = teamsDict[team];
        }
        runSimulation();
    });

    on("sim-bowling-team", "change", (e) => {
        const team = e.target.value;
        const formatVal = document.querySelector('input[name="sim-format"]:checked')?.value || "t20";
        const teamsDict = teamsForFormat(formatVal);
        if (team && teamsDict && teamsDict[team]) {
            document.getElementById("sim-bowling-elo").value = teamsDict[team];
        }
        runSimulation();
    });

    on("sim-venue", "change", (e) => {
        const venue = e.target.value;
        const formatVal = document.querySelector('input[name="sim-format"]:checked')?.value || "t20";
        const venuesDict = venuesForFormat(formatVal);
        if (venue && venuesDict && venuesDict[venue] != null) {
            document.getElementById("sim-venue-avg").value = venuesDict[venue];
        }
        runSimulation();
    });

    function updateSimSlidersMax() {
        const formatVal = document.querySelector('input[name="sim-format"]:checked')?.value || "t20";
        const fmt = FORMATS[formatVal];

        const maxBalls = fmt.totalBalls;
        const ballsSlider = document.getElementById("sim-balls-slider");
        ballsSlider.max = maxBalls;
        if (parseFloat(ballsSlider.value) > maxBalls) {
            ballsSlider.value = Math.round(maxBalls / 2);
        }

        const maxTarget = formatVal === "od" ? 450 : 250;
        document.getElementById("sim-runs-slider").max = maxTarget;
        
        document.getElementById("sim-runs-val").textContent = `${document.getElementById("sim-runs-slider").value} runs`;
        const b = parseFloat(ballsSlider.value);
          if (fmt.unit === "balls") {
              document.getElementById("sim-balls-val").style.display = "none";
              const ring = document.getElementById("sim-balls-ring-container");
              if (ring) {
                  ring.style.display = "flex";
                  document.getElementById("sim-balls-ring-text").textContent = b;
                  const pct = (b / fmt.totalBalls) * 100;
                  ring.style.background = `conic-gradient(var(--primary-grad, #00f2fe) ${pct}%, rgba(255,255,255,0.1) 0)`;
              }
          } else {
              const ring = document.getElementById("sim-balls-ring-container");
              if (ring) ring.style.display = "none";
              document.getElementById("sim-balls-val").style.display = "inline";
              document.getElementById("sim-balls-val").textContent = ballsToLabel(fmt, b);
          }

        document.getElementById("sim-wickets-val").textContent = `${document.getElementById("sim-wickets-slider").value} wickets`;
    }

    ["sim-runs-slider", "sim-balls-slider", "sim-wickets-slider"].forEach(id => {
        on(id, "input", () => {
            updateSimSlidersMax();
            runSimulation();
        });
    });

    // +/- fine-adjustment buttons next to each slider (1 unit per click). Dispatching "input"
    // reuses the sliders' own listener above instead of duplicating the refresh logic.
    document.querySelectorAll(".adjust-btn[data-target]").forEach(btn => {
        btn.addEventListener("click", () => {
            const slider = document.getElementById(btn.dataset.target);
            if (!slider) return;
            const step = parseFloat(btn.dataset.step) || 1;
            const min = parseFloat(slider.min) || 0;
            const max = parseFloat(slider.max) || 100;
            slider.value = Math.min(max, Math.max(min, (parseFloat(slider.value) || 0) + step));
            slider.dispatchEvent(new Event("input", { bubbles: true }));
        });
    });

    // Entering a target implies a fresh chase: no balls bowled yet (balls remaining = full
    // innings) and no wickets down, with the full target still needed.
    on("sim-target", "input", () => {
        const formatVal = document.querySelector('input[name="sim-format"]:checked')?.value || "t20";
        const fmt = FORMATS[formatVal];
        const target = parseFloat(document.getElementById("sim-target").value);

        if (!isNaN(target)) {
            const runsSlider = document.getElementById("sim-runs-slider");
            runsSlider.value = Math.min(target, parseFloat(runsSlider.max) || target);
            document.getElementById("sim-balls-slider").value = fmt.totalBalls;
            document.getElementById("sim-wickets-slider").value = 0;
            updateSimSlidersMax();
        }
        runSimulation();
    });

    ["sim-batting-elo", "sim-bowling-elo", "sim-venue-avg"].forEach(id => {
        on(id, "input", runSimulation);
    });

    on("sim-form", "submit", (e) => {
        e.preventDefault();
        runSimulation();
    });

    on("sim-source", "change", (e) => {
        setRatingSource(e.target.value);
        runSimulation();
    });

    async function runSimulation() {
        const formatVal = document.querySelector('input[name="sim-format"]:checked')?.value || "t20";
        const source = document.getElementById("sim-source").value;
        const fmt = FORMATS[formatVal];

        const target = parseFloat(document.getElementById("sim-target").value) || fmt.parScore;
        const runsRemaining = parseFloat(document.getElementById("sim-runs-slider").value) || 50;
        const ballsRemaining = parseFloat(document.getElementById("sim-balls-slider").value) || 30;
        const wicketsLost = parseFloat(document.getElementById("sim-wickets-slider").value) || 3;

        const battingElo = parseFloat(document.getElementById("sim-batting-elo").value) || 1500;
        const bowlingElo = parseFloat(document.getElementById("sim-bowling-elo").value) || 1500;
        const venueAvg = parseFloat(document.getElementById("sim-venue-avg").value) || fmt.parScore;

        try {
            const response = await fetch("/api/simulate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    format: formatVal,
                    target: target,
                    runs_remaining: runsRemaining,
                    balls_remaining: ballsRemaining,
                    wickets_lost: wicketsLost,
                    batting_team_elo: battingElo,
                    bowling_team_elo: bowlingElo,
                    venue_avg_first_innings_score: venueAvg,
                    source: source
                })
            });

            const data = await response.json();
            if (data.error) { showToast(data.error, "error"); return; }
            setSourceBadge("sim-source-badge", data.source || source);

            const prob = data.win_probability;
            updateGauge(simGauge, prob, "sim-prob-display");

            const ciLower = data.ci_lower !== undefined ? data.ci_lower : prob;
            const ciUpper = data.ci_upper !== undefined ? data.ci_upper : prob;
            const stdDev = data.std_dev !== undefined ? data.std_dev : 0;

            const lowerPct = Math.round(ciLower * 100);
            const upperPct = Math.round(ciUpper * 100);
            const probPct = Math.round(prob * 100);

            document.getElementById("sim-ci-lower").textContent = `${lowerPct}%`;
            document.getElementById("sim-ci-upper").textContent = `${upperPct}%`;
            document.getElementById("sim-uncertainty-badge").textContent = `±${(stdDev * 100).toFixed(1)}% std dev`;

            const fillBar = document.getElementById("sim-ci-bar-fill");
            const marker = document.getElementById("sim-ci-marker");

            fillBar.style.left = `${lowerPct}%`;
            fillBar.style.width = `${Math.max(1, upperPct - lowerPct)}%`;
            marker.style.left = `${probPct}%`;

            const badgesContainer = document.getElementById("sim-fold-badges");
            if (data.fold_probabilities && data.fold_probabilities.length > 0) {
                badgesContainer.innerHTML = data.fold_probabilities
                    .map(fp => `<span class="fold-badge">Fold: ${(fp * 100).toFixed(1)}%</span>`)
                    .join("");
            }

            document.getElementById("sim-rrr-display").textContent = data.required_run_rate !== undefined ? data.required_run_rate : "--";
            document.getElementById("sim-crr-display").textContent = data.run_rate !== undefined ? data.run_rate : "--";
            const eloDiff = Math.round(battingElo - bowlingElo);
            document.getElementById("sim-elo-diff-display").textContent = (eloDiff > 0 ? `+${eloDiff}` : `${eloDiff}`);
            document.getElementById("sim-phase-display").textContent = data.phase ? data.phase.toUpperCase() : "--";
            // Phase Pill styling hook
            const phaseEl = document.getElementById("sim-phase-display");
            if (data.phase) { phaseEl.className = "metric-value phase-pill " + data.phase; } else { phaseEl.className = "metric-value"; }

            renderExplainability("sim-explainability-list", data.explainability);
        } catch (err) {
            console.error("Simulation error:", err);
        }
    }

    fetchMetadata().then(() => {
        updateSimSlidersMax();
        runSimulation();
        populateFirstInningsVenues();
        populateFirstInningsTeams();
        const fiSourceSel = document.getElementById("fi-source");
        fiSourceSel.value = ratingSource;
        refreshSourceSelectOptions(fiSourceSel, document.querySelector('input[name="fi-format"]:checked')?.value || "t20");
    });

    // --------------------------------------------------
    // Live Tab Request Handler & Dynamic Limit Updates
    function renderExplainability(containerId, explainabilityData) {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (!explainabilityData || explainabilityData.length === 0) {
            container.innerHTML = '<div class="explainability-placeholder">No factor contributions available for terminal state.</div>';
            return;
        }

        container.innerHTML = explainabilityData.map(item => {
            const isPos = item.impact === "positive";
            const badgeClass = isPos ? "impact-pos" : "impact-neg";
            const valClass = isPos ? "pos" : "neg";
            const badgeText = isPos ? "+UP" : "-DOWN";
            const signVal = isPos ? `+${item.contribution.toFixed(2)}` : item.contribution.toFixed(2);

            return `
                <div class="contrib-item" title="${item.narrative}">
                    <div class="contrib-info">
                        <span class="impact-badge ${badgeClass}">${badgeText}</span>
                        <span class="contrib-name">${item.feature}</span>
                    </div>
                    <span class="contrib-val ${valClass}">${signVal}</span>
                </div>
            `;
        }).join("");
    }

    // --------------------------------------------------
    // First Innings Predictor Request Handler & Limits
    // --------------------------------------------------
    const firstInningsForm = document.getElementById("firstinnings-predict-form");
    const firstInningsBallsInput = document.getElementById("firstinnings-balls");
    const firstInningsBallsDesc = document.getElementById("firstinnings-balls-desc");

    function updateFirstInningsBallsLimit() {
        const formatVal = document.querySelector('input[name="fi-format"]:checked')?.value || "t20";
        const fmt = FORMATS[formatVal];
        const maxBalls = fmt.totalBalls;
        firstInningsBallsInput.max = maxBalls;
        firstInningsBallsDesc.textContent = `Deliveries bowled (max ${maxBalls})`;
        if (parseFloat(firstInningsBallsInput.value) > maxBalls) {
            firstInningsBallsInput.value = maxBalls;
        }
    }

    // Venue dropdown for the live-state predictor — reuses the same venuesForFormat() lookup
    // (venue -> historical 1st-innings average) that backs the Live Simulator tab's venue select.
    function populateFirstInningsVenues() {
        if (!metadataCache) return;
        const formatVal = document.querySelector('input[name="fi-format"]:checked')?.value || "t20";
        const venuesDict = venuesForFormat(formatVal);
        const venueSelect = document.getElementById("firstinnings-venue");
        const curVenue = venueSelect.value;

        venueSelect.innerHTML = '<option value="">-- Select Venue (optional) --</option>';
        for (const [venue, avg] of Object.entries(venuesDict || {})) {
            venueSelect.add(new Option(`${venue} (Avg: ${avg})`, venue));
        }
        if (curVenue && venuesDict && venuesDict[curVenue] != null) venueSelect.value = curVenue;
    }

    // Team dropdowns for the live-state predictor's "Two Sides" context mode — reuses the
    // same teamsDict (team -> Elo) lookup that backs the Live Simulator tab's team selects.
    function populateFirstInningsTeams() {
        if (!metadataCache) return;
        const formatVal = document.querySelector('input[name="fi-format"]:checked')?.value || "t20";
        const teamsDict = teamsForFormat(formatVal);

        const batSelect = document.getElementById("firstinnings-team-batting");
        const bowlSelect = document.getElementById("firstinnings-team-bowling");
        const curBat = batSelect.value;
        const curBowl = bowlSelect.value;

        batSelect.innerHTML = '<option value="">-- Select Batting Team --</option>';
        bowlSelect.innerHTML = '<option value="">-- Select Bowling Team --</option>';
        for (const [team, elo] of Object.entries(teamsDict || {})) {
            batSelect.add(new Option(`${team} (${elo} Elo)`, team));
            bowlSelect.add(new Option(`${team} (${elo} Elo)`, team));
        }
        if (curBat && teamsDict && teamsDict[curBat] != null) batSelect.value = curBat;
        if (curBowl && teamsDict && teamsDict[curBowl] != null) bowlSelect.value = curBowl;
    }

    document.querySelectorAll('input[name="fi-context"]').forEach(r => r.addEventListener("change", (e) => {
        const mode = e.target.value;
        document.getElementById("fi-context-competition-group").style.display = mode === "teams" ? "none" : "";
        document.getElementById("fi-context-teams-group").style.display = mode === "teams" ? "" : "none";
    }));

    on("firstinnings-venue", "change", (e) => {
        const venue = e.target.value;
        const formatVal = document.querySelector('input[name="fi-format"]:checked')?.value || "t20";
        const venuesDict = venuesForFormat(formatVal);
        if (venue && venuesDict && venuesDict[venue] != null) {
            document.getElementById("firstinnings-venue-avg").value = venuesDict[venue];
        }
    });

    document.querySelectorAll('input[name="fi-format"]').forEach(r => r.addEventListener("change", (e) => {
        updateFirstInningsBallsLimit();
        populateFirstInningsVenues();
        populateFirstInningsTeams();
        refreshSourceSelectOptions(document.getElementById("fi-source"), e.target.value);
    }));
    updateFirstInningsBallsLimit();

    on("fi-source", "change", (e) => setRatingSource(e.target.value));

    firstInningsForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formatVal = document.querySelector('input[name="fi-format"]:checked')?.value || "t20";
        const contextMode = document.querySelector('input[name="fi-context"]:checked')?.value || "competition";
        const teamsDict = teamsForFormat(formatVal);

        let battingElo = 1500.0;
        let bowlingElo = 1500.0;
        if (contextMode === "teams") {
            const battingTeam = document.getElementById("firstinnings-team-batting").value;
            const bowlingTeam = document.getElementById("firstinnings-team-bowling").value;
            if (battingTeam && teamsDict && teamsDict[battingTeam] != null) battingElo = teamsDict[battingTeam];
            if (bowlingTeam && teamsDict && teamsDict[bowlingTeam] != null) bowlingElo = teamsDict[bowlingTeam];
        }

        const source = document.getElementById("fi-source").value;
        const payload = {
            format: formatVal,
            runs_scored: parseFloat(document.getElementById("firstinnings-runs").value),
            balls_faced: parseFloat(firstInningsBallsInput.value),
            wickets_lost: parseFloat(document.getElementById("firstinnings-wickets").value),
            competition: document.getElementById("firstinnings-comp").value,
            batting_team_elo: battingElo,
            bowling_team_elo: bowlingElo,
            venue_avg_first_innings_score: parseFloat(document.getElementById("firstinnings-venue-avg").value) || 160.0,
            source: source
        };

        try {
            const res = await fetch("/api/predict_first_innings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (res.ok) {
                setSourceBadge("fi-source-badge", data.source || source);
                document.getElementById("firstinnings-predicted-display").textContent = data.predicted_score;
                document.getElementById("firstinnings-proj-display").textContent = data.projected_score;
                document.getElementById("firstinnings-crr-display").textContent = data.run_rate.toFixed(2);
                
                // Update progress bar (scale 50 to 250)
                const percent = Math.min(100, Math.max(0, ((data.predicted_score - 50) / 200) * 100));
                document.getElementById("firstinnings-progress-fill").style.width = `${percent}%`;
            } else {
                showToast(data.error || "Prediction failed.", "error");
            }
        } catch (err) {
            console.error("First innings prediction failed", err);
        }
    });

    // --------------------------------------------------
    // First Innings Tab: Live/Monte-Carlo sub-mode toggle
    // --------------------------------------------------
    document.getElementById("fi-mode-tabs").addEventListener("click", (e) => {
        const btn = e.target.closest(".mc-subtab");
        if (!btn) return;
        document.querySelectorAll("#fi-mode-tabs .mc-subtab").forEach(b => b.classList.toggle("active", b === btn));
        const mode = btn.dataset.fiMode;
        document.getElementById("fi-live").style.display = mode === "live" ? "" : "none";
        document.getElementById("fi-mc-panel").style.display = mode === "mc" ? "" : "none";
        if (mode === "mc") {
            fiMcFilters.refresh();
            fiMcHistogramChart?.resize();
        }
    });

    // --------------------------------------------------
    // First Innings Tab: Monte Carlo Sim (1st innings only)
    // --------------------------------------------------
    const fiFilterFormat = document.getElementById("fi-filter-format");
    const fiFilterYear = document.getElementById("fi-filter-year");
    const fiFilterLeague = document.getElementById("fi-filter-league");
    const fiFilterTeam = document.getElementById("fi-filter-team");
    const fiMatchSelect = document.getElementById("fi-match-select");
    const fiMatchCount = document.getElementById("fi-match-count");
    const fiMatchSummary = document.getElementById("fi-match-summary");
    const fiMcRunBtn = document.getElementById("fi-mc-run-btn");

    const fiMcFilters = bindMatchFilters({
        formatSel: fiFilterFormat, yearSel: fiFilterYear, leagueSel: fiFilterLeague,
        teamSel: fiFilterTeam, matchSel: fiMatchSelect, countEl: fiMatchCount,
    });

    // The currently-selected historical match; both sides' point-in-time squads (real XI +
    // their Elo as of that match) are pulled server-side via extract_match_squads(match_id).
    let fiSelectedMatch = null;

    function renderFiMatchSummary() {
        const matchId = fiMatchSelect.value;
        const fmt = fiMatchSelect.options[fiMatchSelect.selectedIndex]?.dataset.format || "";
        if (!matchId) {
            fiMatchSummary.style.display = "none";
            document.getElementById("fi-lineups-grid").style.display = "none";
            fiSelectedMatch = null;
            fiMcRunBtn.disabled = true;
            return;
        }
        const entry = matchCatalog.find(m => m.match_id === matchId && m.format_type === fmt);
        if (!entry) return;
        fiSelectedMatch = { matchId, format: fmt };
        refreshSourceSelectOptions(document.getElementById("fi-mc-source"), fmt);
        document.getElementById("fi-batting-team").textContent = entry.defending_team;
        document.getElementById("fi-bowling-team").textContent = entry.chasing_team;
        document.getElementById("fi-match-date").textContent = entry.date || "--";
        document.getElementById("fi-actual-score").textContent = entry.target - 1;
        fiMatchSummary.style.display = "block";
        fiMcRunBtn.disabled = false;
        loadFiMatchLineups(matchId, fmt);
    }
    fiMatchSelect.addEventListener("change", renderFiMatchSummary);

    // Lineups & Elo for the First Innings MC panel — only the batting side's lineup and the
    // bowling side's attack matter here (it's a 1st-innings-only sim), so the other two roles
    // are left unrendered (renderSquadCard no-ops on the panel's missing containers for them).
    // Shares lineupsCache with Match Explorer since the same match_id+format key is identical.
    async function loadFiMatchLineups(matchId, fmt) {
        const grid = document.getElementById("fi-lineups-grid");
        const key = lineupsKeyFor(matchId, fmt);
        const cached = lineupsCache.get(key);
        if (cached) {
            renderFiMatchLineups(cached);
            return;
        }
        grid.style.display = "none";
        try {
            const res = await fetch(`/api/match/${matchId}/squads?format=${fmt}`);
            const data = await res.json();
            if (!res.ok || data.error) return;
            lineupsCache.set(key, data);
            renderFiMatchLineups(data);
        } catch (err) {
            console.error("Failed to fetch match lineups", err);
        }
    }

    function renderFiMatchLineups(data) {
        renderSquadCard("fi-lineups-team-1", data.defending_team, {
            overall: data.defending_team_elo,
            battingSquad: data.defending_team_batting_squad_elo,
            bowlingSquad: data.defending_team_bowling_squad_elo,
            effective: data.defending_team_effective_elo,
        });
        renderSquadCard("fi-lineups-team-2", data.chasing_team, {
            overall: data.chasing_team_elo,
            battingSquad: data.chasing_team_batting_squad_elo,
            bowlingSquad: data.chasing_team_bowling_squad_elo,
            effective: data.chasing_team_effective_elo,
        });
        document.getElementById("fi-lineups-grid").style.display = "grid";
    }

    const runFiMcProgress = makeProgressBar("fi-mc-progress-bar", "fi-mc-progress-pct", "fiMcEstimatedMs", 4000);
    const fiMcCache = new Map();
    function fiMcKey(fmt, matchId, source) { return `${fmt}:${matchId}:${source}`; }

    fiMcRunBtn.addEventListener("click", () => {
        if (!fiSelectedMatch) return;
        runFiMcSimulation(fiSelectedMatch.format, fiSelectedMatch.matchId);
    });
    on("fi-mc-source", "change", (e) => setRatingSource(e.target.value));

    async function runFiMcSimulation(format, matchId) {
        document.getElementById("fi-mc-results").style.display = "none";
        document.getElementById("fi-mc-loading").style.display = "block";
        const progress = runFiMcProgress();
        const source = document.getElementById("fi-mc-source").value;
        const key = fiMcKey(format, matchId, source);

        try {
            let data = fiMcCache.get(key);
            if (!data) {
                const res = await fetch("/api/montecarlo_first_innings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ format, match_id: matchId, n_sims: 10000, source })
                });
                data = await res.json();
                if (!data.error) fiMcCache.set(key, data);
            }
            setSourceBadge("fi-mc-source-badge", data.source || source);

            if (data.error) {
                progress.cancel();
                document.getElementById("fi-mc-loading").style.display = "none";
                showToast(data.error, "error");
                return;
            }
            progress.finish();
            document.getElementById("fi-mc-loading").style.display = "none";
            document.getElementById("fi-mc-experimental-warning").style.display = data.simulation_experimental ? "block" : "none";
            renderFiMcData(data);
            document.getElementById("fi-mc-results").style.display = "block";
        } catch (err) {
            console.error(err);
            progress.cancel();
            document.getElementById("fi-mc-loading").style.display = "none";
            showToast("Simulation failed.", "error");
        }
    }

    let lastFiMcData = null;
    let fiMcRole = "batting";
    let fiMcHistogramChart = null;

    function fiMcDrawHistogram(hist, borderColor) {
        if (fiMcHistogramChart) fiMcHistogramChart.destroy();
        const ctx = document.getElementById("fi-mc-histogram").getContext("2d");
        fiMcHistogramChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: hist.labels,
                datasets: [{
                    label: 'Simulations',
                    data: hist.counts,
                    backgroundColor: 'rgba(0, 242, 254, 0.35)',
                    borderColor,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: "#94a3b8" } },
                    y: { display: false }
                },
                plugins: { legend: { display: false }, tooltip: { enabled: true } }
            }
        });
    }

    function fiMcSetMarket(n, title, obj, prop, color) {
        document.getElementById(`fi-mc-m${n}-title`).textContent = title;
        document.getElementById(`fi-mc-m${n}-list`).innerHTML = renderLeaderboard(obj, prop, color);
    }

    // Renders the /api/montecarlo_first_innings response: batting card (score distribution,
    // top batsman, 30+/50+) for team_a, or bowling card (wickets distribution, top bowler,
    // 2+/3+ wkts) for team_b — there's only one side of each discipline since it's one innings.
    function renderFiMcMarkets() {
        const data = lastFiMcData;
        if (!data) return;
        const vBat = teamVisual(data.team_a);
        const vBowl = teamVisual(data.team_b);
        const battingBlock = document.getElementById("fi-mc-batting-block");

        if (fiMcRole === "batting") {
            battingBlock.style.display = "";
            document.getElementById("fi-mc-stat-tiles").innerHTML =
                statTilesHTML(data.total_runs, "10,000 sims");
            document.getElementById("fi-mc-overunder").innerHTML = mcOverUnderHTML(data.total_runs);

            document.getElementById("fi-mc-hist-title").textContent = `Projected 1st-innings score — ${data.team_a}`;
            fiMcDrawHistogram(data.hist, vBat.color);

            fiMcSetMarket(1, `Top Batsman — ${data.team_a}`, data.top_batsman, "p_top", vBat.color);
            fiMcSetMarket(2, `30+ Runs — ${data.team_a}`, data.props_30, "p_hit", vBat.color);
            fiMcSetMarket(3, `50+ Runs — ${data.team_a}`, data.props_50, "p_hit", vBat.color);
        } else {
            battingBlock.style.display = "none";
            document.getElementById("fi-mc-hist-title").textContent = `Projected wickets taken — ${data.team_b}`;
            fiMcDrawHistogram(data.wkts_hist, vBowl.color);

            fiMcSetMarket(1, `Most Wickets — ${data.team_b}`, data.top_bowler, "p_top", vBowl.color);
            fiMcSetMarket(2, `2+ Wickets — ${data.team_b}`, data.bowler_2w, "p_hit", vBowl.color);
            fiMcSetMarket(3, `3+ Wickets — ${data.team_b}`, data.bowler_3w, "p_hit", vBowl.color);
        }

        document.querySelectorAll("#fi-mc-roletabs .mc-subtab").forEach(b =>
            b.classList.toggle("active", b.dataset.fiRole === fiMcRole));
    }

    function renderFiMcData(data) {
        lastFiMcData = data;
        fiMcRole = "batting";
        renderFiMcMarkets();
    }

    document.getElementById("fi-mc-roletabs").addEventListener("click", (e) => {
        const btn = e.target.closest(".mc-subtab");
        if (!btn) return;
        fiMcRole = btn.dataset.fiRole;
        renderFiMcMarkets();
    });

    // --------------------------------------------------
    // Match Explorer Tab
    // --------------------------------------------------
    const matchSelect = document.getElementById("match-select");
    const explSummary = document.getElementById("explorer-match-summary");
    let timelineChart = null;
    let wormChart = null;
    let manhattanChart = null;

    // Team colours for the worm / Manhattan charts: 1st-innings team vs 2nd-innings (chasing) team.
    const INNINGS_COLORS = { 1: "#a855f7", 2: "#00f2fe" };

    function hideExplorerCharts() {
        ["timeline-chart-container", "worm-chart-container", "manhattan-chart-container"]
            .forEach(id => { document.getElementById(id).style.display = "none"; });
    }

    // Full match catalog held in memory; every filtered dropdown (Match Explorer, First
    // Innings MC panel) is rebuilt from this in-memory list so the browser never has to
    // render all ~17k options at once (which left it appearing empty).
    let matchCatalog = [];
    const MAX_MATCH_OPTIONS = 500;  // cap the rendered list; narrow filters to see more
    // Determinate loading bar for the timeline request (same adaptive method as the MC sim).
    const runTimelineProgress = makeProgressBar("expl-progress-bar", "expl-progress-pct", "explTimelineEstimatedMs", 1500);

    function fillSelect(select, values, { keep = "" } = {}) {
        const first = select.querySelector("option");  // preserve the "All" placeholder
        select.innerHTML = "";
        if (first) select.appendChild(first);
        values.forEach(v => {
            const opt = document.createElement("option");
            opt.value = v.value;
            opt.textContent = v.label;
            select.appendChild(opt);
        });
        select.value = keep;
        if (select.value !== keep) select.value = "";  // selection no longer valid → reset to All
    }

    function matchLabel(m) {
        const date = m.date ? ` — ${m.date}` : "";
        const comp = m.competition ? ` [${m.competition.toUpperCase()}]` : "";
        return `[${(m.format_type || "").toUpperCase()}]${comp} ${m.chasing_team} vs ${m.defending_team} (Target ${m.target} — Chasing ${m.result.toUpperCase()})${date}`;
    }

    function uniqueSorted(items, { numericDesc = false } = {}) {
        const set = [...new Set(items.filter(Boolean))];
        set.sort((a, b) => numericDesc ? Number(b) - Number(a) : String(a).localeCompare(String(b)));
        return set;
    }

    // Binds a Format/Year/League/Team filter row + a match <select> to the shared matchCatalog.
    // Shared by the Match Explorer and the First Innings tab's Monte Carlo panel — both need the
    // identical "narrow down, then pick one historical match" flow. Callers own the match select's
    // own 'change' listener (what happens once a match is picked differs between the two).
    function bindMatchFilters({ formatSel, yearSel, leagueSel, teamSel, matchSel, countEl }) {
        // Which catalog entries survive the currently-selected filters (optionally ignoring one
        // dimension, so each filter's own option list stays populated with the others applied).
        function filteredMatches(ignore) {
            const fmt = formatSel.value;
            const year = yearSel.value;
            const league = leagueSel.value;
            const team = teamSel.value;
            return matchCatalog.filter(m => {
                if (ignore !== "format" && fmt && m.format_type !== fmt) return false;
                if (ignore !== "year" && year && m.year !== year) return false;
                if (ignore !== "league" && league && m.competition !== league) return false;
                if (ignore !== "team" && team && m.chasing_team !== team && m.defending_team !== team) return false;
                return true;
            });
        }

        function refreshFilterOptions() {
            // Each dropdown's options reflect the other active filters, not its own.
            fillSelect(formatSel,
                uniqueSorted(filteredMatches("format").map(m => m.format_type))
                    .map(v => ({ value: v, label: v.toUpperCase() })),
                { keep: formatSel.value });
            fillSelect(yearSel,
                uniqueSorted(filteredMatches("year").map(m => m.year), { numericDesc: true })
                    .map(v => ({ value: v, label: v })),
                { keep: yearSel.value });
            fillSelect(leagueSel,
                uniqueSorted(filteredMatches("league").map(m => m.competition))
                    .map(v => ({ value: v, label: v.toUpperCase() })),
                { keep: leagueSel.value });
            fillSelect(teamSel,
                uniqueSorted(filteredMatches("team").flatMap(m => [m.chasing_team, m.defending_team]))
                    .map(v => ({ value: v, label: v })),
                { keep: teamSel.value });
        }

        function refreshMatchList() {
            const matches = filteredMatches();
            const prev = matchSel.value;
            const prevFmt = matchSel.options[matchSel.selectedIndex]?.dataset.format || "";
            matchSel.innerHTML = '<option value="">-- Select a match from database --</option>';
            matches.slice(0, MAX_MATCH_OPTIONS).forEach(m => {
                const option = document.createElement("option");
                option.value = m.match_id;
                option.dataset.format = m.format_type;  // disambiguates ids shared across formats
                option.textContent = matchLabel(m);
                matchSel.appendChild(option);
            });
            // Preserve the current selection if it survived the filter change.
            if (prev) {
                const still = [...matchSel.options].find(o => o.value === prev && o.dataset.format === prevFmt);
                if (still) matchSel.value = prev;
            }
            const shown = Math.min(matches.length, MAX_MATCH_OPTIONS);
            if (countEl) countEl.textContent = matches.length > MAX_MATCH_OPTIONS
                ? `(showing ${shown} of ${matches.length} — narrow filters to see more)`
                : `(${matches.length} match${matches.length === 1 ? "" : "es"})`;
        }

        function refresh() {
            refreshFilterOptions();
            refreshMatchList();
        }

        [formatSel, yearSel, leagueSel, teamSel].forEach(sel => sel.addEventListener("change", refresh));

        return { refresh };
    }

    const explFilterFormat = document.getElementById("expl-filter-format");
    const explFilterYear = document.getElementById("expl-filter-year");
    const explFilterComp = document.getElementById("expl-filter-competition");
    const explFilterTeam = document.getElementById("expl-filter-team");
    const explMatchCount = document.getElementById("expl-match-count");
    const explorerFilters = bindMatchFilters({
        formatSel: explFilterFormat, yearSel: explFilterYear, leagueSel: explFilterComp,
        teamSel: explFilterTeam, matchSel: matchSelect, countEl: explMatchCount,
    });

    async function loadMatchCatalog() {
        try {
            const res = await fetch("/api/matches");
            matchCatalog = await res.json();
            explorerFilters.refresh();
            fiMcFilters.refresh();
        } catch (err) {
            console.error("Failed to load match list", err);
        }
    }

    // The shared filters + match dropdown drive whichever sub-mode is active.
    let explorerMode = "bbb";
    // Ball-by-ball responses are cached per match so switching modes (or back to a previously
    // viewed match) re-renders instantly instead of re-hitting the (slow) timeline endpoint.
    const bbbCache = new Map();
    function bbbKeyFor(matchId, fmt, source) { return `${fmt}:${matchId}:${source}`; }

    // Route the current selection to the active mode (ball-by-ball timeline, lineups, or Monte
    // Carlo sim). MC never auto-runs from a selection — it only pre-fills the form; the user
    // must press "Run Simulation" themselves.
    function renderSelectedMatch() {
        const matchId = matchSelect.value;
        const fmt = matchSelect.options[matchSelect.selectedIndex]?.dataset.format || "";
        if (!matchId) {
            explSummary.style.display = "none";
            hideExplorerCharts();
            return;
        }
        // The timeline source toggle only affects ball-by-ball mode (the other two modes hit
        // endpoints that aren't source-aware -- see rating_source_toggle_plan.md Phase 6's
        // deferred squad-driven endpoints). Refreshed per-selection since format varies match
        // to match here, unlike a tab with one fixed format selector.
        refreshSourceSelectOptions(document.getElementById("expl-source"), fmt);
        if (explorerMode === "mc") {
            prefillMcForMatch(matchId, fmt);
        } else if (explorerMode === "lineups") {
            loadMatchLineups(matchId, fmt);
        } else {
            loadBallByBall(matchId, fmt, document.getElementById("expl-source").value);
        }
    }

    matchSelect.addEventListener("change", renderSelectedMatch);
    on("expl-source", "change", (e) => {
        setRatingSource(e.target.value);
        if (explorerMode === "bbb" && matchSelect.value) {
            const fmt = matchSelect.options[matchSelect.selectedIndex]?.dataset.format || "";
            loadBallByBall(matchSelect.value, fmt, e.target.value);
        }
    });

    // --------------------------------------------------
    // Shared Lineups & Elo rendering — used by Match Explorer, First Innings MC, and Live MC.
    // Every squad dict here has the same shape (from extract_match_squads / build_live_team):
    // {name, lineup: [{name, effective_elo, ...}], attack: [{name, effective_elo, ...}]}.
    // --------------------------------------------------
    const lineupsCache = new Map();  // `${fmt}:${matchId}` -> /api/match/<id>/squads response
    function lineupsKeyFor(matchId, fmt) { return `${fmt}:${matchId}`; }

    // player.rated is only present for live-scraped squads (build_live_team() in app.py) --
    // historical squads (find_recent_squad/extract_match_squads) have no such flag and are
    // always real ratings, so `rated === false` (not just falsy) is the only "unrated" case.
    //
    // effective_elo (shown, primary) is Bayesian-shrunk toward the team's Elo for players
    // with limited match history -- see get_effective_elo() in player_elo_engine.py -- so a
    // low-sample player on a strong team can show well above their own raw_elo (their own
    // current, un-blended rating -- NOT a peak or career average). Surface raw_elo as a
    // secondary figure whenever it meaningfully differs, so that isn't mistaken for a data error.
    //
    // overs_per_inns is present on ATTACK rows only. The attack is ordered by expected bowling
    // workload, not by Elo (see frontline_bowl_sort_key), so without showing that figure the
    // card reads as an unsorted Elo list. Shown first because it IS the sort key.
    function playerRowHTML(player, i, accentColor) {
        const elo = Math.round(player.effective_elo);
        const rawElo = player.raw_elo != null ? Math.round(player.raw_elo) : null;
        const showRaw = rawElo != null && rawElo !== elo;
        const unrated = player.rated === false;
        const flag = unrated
            ? `<span class="player-unrated-flag" title="No historical rating for this player — showing neutral 1500 Elo">⚠</span>`
            : "";
        const bits = [];
        if (player.overs_per_inns != null) {
            bits.push(`<span title="Average overs bowled per innings across this player's recent appearances — this is what the attack is ordered by, not Elo.">${player.overs_per_inns} ov/inns</span>`);
        }
        if (showRaw) {
            bits.push(`<span title="This player's own current rating on its own, before the team-strength blend applied for small sample sizes. Not a peak or career-average figure.">own ${rawElo}</span>`);
        }
        const rawEloHTML = bits.length
            ? `<span class="leaderboard-raw-elo">${bits.join(" · ")}</span>`
            : "<span></span>";
        return `
            <div class="leaderboard-item${unrated ? " leaderboard-item-unrated" : ""}">
                <span class="lb-accent" style="background:${accentColor}"></span>
                <span class="leaderboard-rank">${i + 1}</span>
                <span class="leaderboard-name">${escHtml(player.name)}${flag}</span>
                <span class="leaderboard-prob">${elo} Elo</span>
                ${rawEloHTML}
            </div>`;
    }

    // Phase 4 (effective_game_elo_plan.md): all four stat-tile values come straight from the
    // backend (Phase 3's compute_effective_elo() + the weighted batting_squad_elo/
    // bowling_squad_elo columns the win-prob model actually trained on) — NEVER recomputed
    // client-side. A frontend flat mean here would silently diverge from what the model/
    // backend used the moment the weighting scheme changes (see the plan's "display must use
    // the backend's weighted aggregate" invariant — this replaces the old meanElo() path that
    // violated it).
    const EFFECTIVE_ELO_TIP = "Team W/L rating adjusted for the strength of the XI actually fielded vs. this team's usual side.";

    function squadStatsTilesHTML(overallElo, battingElo, bowlingElo, effectiveElo) {
        const tile = (label, val, tip) => `
            <div class="stat-tile">
                <div class="st-label">${label}${tip ? `<span class="info-tip" tabindex="0" aria-label="${tip}">ⓘ<span class="info-tip-bubble" aria-hidden="true">${tip}</span></span>` : ""}</div>
                <div class="st-val">${val != null ? Math.round(val) : "--"}</div>
            </div>`;
        return tile("Overall Elo", overallElo)
             + tile("Batting Squad Elo", battingElo)
             + tile("Bowling Squad Elo", bowlingElo)
             + tile("Effective Elo", effectiveElo, EFFECTIVE_ELO_TIP);
    }

    // Renders one side into `${prefix}-name` / `${prefix}-stats` / `${prefix}-batting` /
    // `${prefix}-bowling`. Batting/bowling target elements are optional per caller — the First
    // Innings panel only wants one role per side, so its missing element just no-ops here.
    // `stats` = {overall, battingSquad, bowlingSquad, effective}, all backend-sourced.
    function renderSquadCard(prefix, side, stats) {
        const color = teamVisual(side.name).color;
        const nameEl = document.getElementById(`${prefix}-name`);
        if (nameEl) nameEl.textContent = side.name;

        const statsEl = document.getElementById(`${prefix}-stats`);
        if (statsEl) {
            statsEl.innerHTML = squadStatsTilesHTML(stats.overall, stats.battingSquad, stats.bowlingSquad, stats.effective);
        }

        const battingEl = document.getElementById(`${prefix}-batting`);
        if (battingEl) battingEl.innerHTML = (side.lineup || []).map((p, i) => playerRowHTML(p, i, color)).join("");

        const bowlingEl = document.getElementById(`${prefix}-bowling`);
        if (bowlingEl) bowlingEl.innerHTML = (side.attack || []).map((p, i) => playerRowHTML(p, i, color)).join("");
    }

    // --------------------------------------------------
    // Match Explorer: Lineups & Elo sub-panel
    // --------------------------------------------------
    function renderMatchLineups(data) {
        renderSquadCard("lineups-team-1", data.defending_team, {
            overall: data.defending_team_elo,
            battingSquad: data.defending_team_batting_squad_elo,
            bowlingSquad: data.defending_team_bowling_squad_elo,
            effective: data.defending_team_effective_elo,
        });
        renderSquadCard("lineups-team-2", data.chasing_team, {
            overall: data.chasing_team_elo,
            battingSquad: data.chasing_team_batting_squad_elo,
            bowlingSquad: data.chasing_team_bowling_squad_elo,
            effective: data.chasing_team_effective_elo,
        });
        document.getElementById("lineups-empty").style.display = "none";
        document.getElementById("lineups-grid").style.display = "grid";
    }

    async function loadMatchLineups(matchId, fmt) {
        explSummary.style.display = "none";  // the chase-outcome card belongs to ball-by-ball mode
        const key = lineupsKeyFor(matchId, fmt);
        const cached = lineupsCache.get(key);
        if (cached) {
            renderMatchLineups(cached);
            return;
        }
        document.getElementById("lineups-grid").style.display = "none";
        document.getElementById("lineups-empty").style.display = "block";
        document.getElementById("lineups-empty").textContent = "Loading lineups…";
        try {
            const res = await fetch(`/api/match/${matchId}/squads?format=${fmt}`);
            const data = await res.json();
            if (!res.ok || data.error) {
                document.getElementById("lineups-empty").textContent =
                    `Could not load lineups${data.error ? `: ${data.error}` : ""}.`;
                return;
            }
            lineupsCache.set(key, data);
            renderMatchLineups(data);
        } catch (err) {
            console.error("Failed to fetch match lineups", err);
            document.getElementById("lineups-empty").textContent = "Could not load lineups.";
            document.getElementById("lineups-empty").style.display = "block";
        }
    }

    async function loadBallByBall(matchId, fmt, source = ratingSource) {
        const key = bbbKeyFor(matchId, fmt, source);
        const cached = bbbCache.get(key);
        if (cached) {
            renderBallByBallData(cached);
            return;
        }

        // Show the determinate loading bar while the timeline is computed server-side.
        const explLoading = document.getElementById("explorer-loading");
        explSummary.style.display = "none";
        hideExplorerCharts();
        explLoading.style.display = "block";
        const progress = runTimelineProgress();

        try {
            const res = await fetch(`/api/match/${matchId}?format=${fmt}&source=${source}`);
            const data = await res.json();

            // Surface backend errors / empty timelines instead of silently throwing below.
            if (!res.ok || data.error || !Array.isArray(data.timeline) || data.timeline.length === 0) {
                progress.cancel();
                explLoading.style.display = "none";
                console.error("Match timeline unavailable", res.status, data);
                showToast(`Could not load this match's timeline${data.error ? `: ${data.error}` : ""}.`, "error");
                return;
            }

            progress.finish();
            explLoading.style.display = "none";
            bbbCache.set(key, data);
            renderBallByBallData(data);
        } catch (err) {
            progress.cancel();
            explLoading.style.display = "none";
            console.error("Failed to fetch match timeline", err);
            showToast("Could not load this match's timeline.", "error");
        }
    }

    // Renders a (possibly cached) /api/match response: summary card, key moments, timeline,
    // worm, Manhattan. Pulled out of loadBallByBall so cache hits skip straight to rendering.
    function renderBallByBallData(data) {
        setSourceBadge("expl-source-badge", data.source || "elo");
        document.getElementById("expl-chasing-team").textContent = data.chasing_team;
        document.getElementById("expl-defending-team").textContent = data.defending_team;
        document.getElementById("expl-target").textContent = data.target;

        const resultDisplay = document.getElementById("expl-result");
        resultDisplay.textContent = `${data.chasing_team} ${data.result.toUpperCase()} (Scored ${data.chase_score}/${data.timeline[data.timeline.length-1]?.wickets_lost || 10})`;
        resultDisplay.style.color = data.result === "won" ? "#10b981" : "#f43f5e";

        explSummary.style.style = "block";
        explSummary.style.display = "block";
        document.getElementById("timeline-chart-container").style.display = "block";

        // Rank key moments before plotTimeline, which annotates the biggest-swing point's label.
        renderKeyMoments(data.timeline, data.format_type);
        plotTimeline(data.timeline, data.chasing_team, data.format_type);

        // Run worm + Manhattan (both innings) share the same match payload.
        const inningsCharts = Array.isArray(data.innings_charts) ? data.innings_charts : [];
        if (inningsCharts.length) {
            plotWorm(inningsCharts, data.target);
            plotManhattan(inningsCharts);
            document.getElementById("worm-chart-container").style.display = "block";
            document.getElementById("manhattan-chart-container").style.display = "block";
        }
    }

    // In MC mode the shared match dropdown pre-fills the simulator form only — the defending
    // side (bats first) as Team A, the chasing side as Team B. It never runs the sim; that's
    // strictly a "Run Simulation" button action. syncMcPanelToSelection reflects whatever is
    // already known (a cached result from an earlier explicit run, or one still in flight) so
    // switching tabs never loses results — it just never triggers a new computation on its own.
    function prefillMcForMatch(matchId, fmt) {
        explSummary.style.display = "none";  // the chase-outcome card belongs to ball-by-ball mode
        const entry = matchCatalog.find(m => m.match_id === matchId && m.format_type === fmt);
        if (!entry) return;

        document.getElementById("mc-format").value = entry.format_type;
        populateMCDropdowns();  // refresh team lists for this format (labeled with live Elo)
        setSelectValue("mc-team-a", entry.defending_team);
        setSelectValue("mc-team-b", entry.chasing_team);
        // Correct those two options' labels to the Elo each side actually had entering this match.
        setMcOptionElo("mc-team-a", entry.defending_team, entry.defending_team_elo);
        setMcOptionElo("mc-team-b", entry.chasing_team, entry.chasing_team_elo);
        refreshSourceSelectOptions(document.getElementById("mc-source"), entry.format_type);
        syncMcPanelToSelection();
    }

    // Set a <select> to a value, adding the option first if it isn't already listed (historical
    // teams may not appear in the current Elo-based dropdown).
    function setSelectValue(id, val) {
        const sel = document.getElementById(id);
        if (!sel) return;
        if (![...sel.options].some(o => o.value === val)) sel.add(new Option(val, val));
        sel.value = val;
    }

    // populateMCDropdowns() labels every option with each team's CURRENT/live Elo (fine for a
    // freeform custom matchup). When a specific historical match is selected, overwrite that
    // team's option label with its pre-match Elo (matches_list' defending/chasing_team_elo,
    // sourced server-side from the match's actual elo_ratings.parquet row) so the number shown
    // reflects that team's strength directly before this match was played, not today.
    function setMcOptionElo(selectId, teamName, elo) {
        const sel = document.getElementById(selectId);
        if (!sel || elo == null) return;
        const opt = [...sel.options].find(o => o.value === teamName);
        if (opt) opt.textContent = `${teamName} (${elo} Elo)`;
    }

    // Explorer mode toggle: Ball-by-Ball data vs Lineups & Elo vs the Monte Carlo simulator.
    // Swaps the visible sub-panel and re-renders the current selection for the newly-active mode.
    document.getElementById("explorer-mode-tabs").addEventListener("click", (e) => {
        const btn = e.target.closest(".mc-subtab");
        if (!btn) return;
        explorerMode = btn.dataset.explMode;
        document.querySelectorAll("#explorer-mode-tabs .mc-subtab")
            .forEach(b => b.classList.toggle("active", b === btn));
        document.getElementById("explorer-bbb").style.display = explorerMode === "bbb" ? "block" : "none";
        document.getElementById("explorer-lineups").style.display = explorerMode === "lineups" ? "block" : "none";
        document.getElementById("explorer-mc").style.display = explorerMode === "mc" ? "block" : "none";
        // Any existing MC results/histogram stay exactly as they were — switching modes never
        // re-runs a sim. Chart.js miscalculates canvas size while its container is display:none,
        // so nudge it to remeasure now that the panel is visible again.
        if (explorerMode === "mc") mcHistogramChart?.resize();
        renderSelectedMatch();
    });

    // Key-moments toggle: Combined / Wickets / Boundaries — re-renders from the cached timeline.
    document.getElementById("key-moments-tabs").addEventListener("click", (e) => {
        const btn = e.target.closest(".mc-subtab");
        if (!btn) return;
        keyMomentsMode = btn.dataset.kmMode;
        document.querySelectorAll("#key-moments-tabs .mc-subtab")
            .forEach(b => b.classList.toggle("active", b === btn));
        drawKeyMoments();
    });

    function plotTimeline(timeline, teamName, formatId) {
        if (timelineChart) timelineChart.destroy();
        const fmt = FORMATS[formatId || "t20"];

        const ctx = document.getElementById("timeline-chart").getContext("2d");
        
        // Sort timeline ascending by balls faced
        const sortedTimeline = timeline.sort((a, b) => a.balls_faced - b.balls_faced);

        // Find biggest swing
        let maxSwingIdx = -1;
        let maxSwingVal = -1;
        for (let i = 1; i < sortedTimeline.length; i++) {
            const swing = Math.abs(sortedTimeline[i].win_probability - sortedTimeline[i-1].win_probability);
            if (swing > maxSwingVal) {
                maxSwingVal = swing;
                maxSwingIdx = i;
            }
        }
        
        const labels = sortedTimeline.map(item => ballsToLabel(fmt, item.balls_faced));
        
        const probabilities = sortedTimeline.map(item => item.win_probability * 100);

        // Build per-point annotation styling
        const pointRadii = [];
        const pointHoverRadii = [];
        const pointBgColors = [];
        const pointBorderColors = [];
        const pointBorderWidths = [];

        sortedTimeline.forEach((item, i) => {
            let isWicket = item.event_type === "wicket";
            let isSix = item.event_type === "six";
            let isFour = item.event_type === "four";
            let isBiggestSwing = (i === maxSwingIdx);
            
            if (isWicket) {
                pointRadii.push(7);
                pointHoverRadii.push(9);
                pointBgColors.push("#f43f5e");
                pointBorderColors.push("#ffffff");
                pointBorderWidths.push(2);
            } else if (isSix) {
                pointRadii.push(6);
                pointHoverRadii.push(8);
                pointBgColors.push("#a855f7");
                pointBorderColors.push("#ffffff");
                pointBorderWidths.push(1.5);
            } else if (isFour) {
                pointRadii.push(5);
                pointHoverRadii.push(7);
                pointBgColors.push("#00f2fe");
                pointBorderColors.push("#ffffff");
                pointBorderWidths.push(1.5);
            } else if (isBiggestSwing) {
                pointRadii.push(7);
                pointHoverRadii.push(10);
                pointBgColors.push("#f59e0b");
                pointBorderColors.push("#ffffff");
                pointBorderWidths.push(2.5);
                // Attach a special label for the tooltip
                item.event_label = (item.event_label ? item.event_label + " + " : "") + "BIGGEST SWING!";
            } else {
                pointRadii.push(0);
                pointHoverRadii.push(5);
                pointBgColors.push("#00f2fe");
                pointBorderColors.push("#00f2fe");
                pointBorderWidths.push(1);
            }
        });

        // Chart gradient fill
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, "rgba(0, 242, 254, 0.35)");
        gradient.addColorStop(1, "rgba(0, 242, 254, 0.0)");

        const phaseBandsPlugin = {
            id: 'phaseBands',
            beforeDraw: (chart) => {
                const {ctx, chartArea: {top, bottom, left, right}, scales: {x}} = chart;
                ctx.save();
                
                const getPixelForBall = (targetBall) => {
                    for(let i=0; i<sortedTimeline.length; i++) {
                        if(sortedTimeline[i].balls_faced >= targetBall) {
                            return x.getPixelForTick(i);
                        }
                    }
                    return right;
                };

                const ppEndPx = getPixelForBall(fmt.ppEndBall);
                const deathStartPx = getPixelForBall(fmt.deathStartBall);

                // Draw powerplay band (orange tint)
                ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
                ctx.fillRect(left, top, ppEndPx - left, bottom - top);

                // Draw death band (red tint)
                ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
                ctx.fillRect(deathStartPx, top, right - deathStartPx, bottom - top);
                
                ctx.restore();
            }
        };

        timelineChart = new Chart(ctx, {
            type: "line",
            plugins: [phaseBandsPlugin],
            data: {
                labels: labels,
                datasets: [{
                    label: `${teamName} Win %`,
                    data: probabilities,
                    borderColor: "#00f2fe",
                    borderWidth: 3,
                    fill: true,
                    backgroundColor: gradient,
                    tension: 0.3,
                    pointRadius: pointRadii,
                    pointHoverRadius: pointHoverRadii,
                    pointBackgroundColor: pointBgColors,
                    pointBorderColor: pointBorderColors,
                    pointBorderWidth: pointBorderWidths
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8", maxTicksLimit: 20 },
                        title: { display: true, text: fmt.unit === "balls" ? "Balls Faced" : "Overs Faced", color: "#64748b" }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8", callback: (val) => `${val}%` },
                        min: 0,
                        max: 100,
                        title: { display: true, text: "Win Probability", color: "#64748b" }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => {
                                if (!items.length) return "";
                                const index = items[0].dataIndex;
                                const item = sortedTimeline[index];
                                const overStr = fmt.unit === "balls" ? `Ball ${item.balls_faced}` : `Over ${Math.floor(item.balls_faced / 6)}.${item.balls_faced % 6}`;
                                if (item.event_label) {
                                    return `${overStr} — ⚡ ${item.event_label}`;
                                }
                                return overStr;
                            },
                            label: (context) => {
                                const index = context.dataIndex;
                                const item = sortedTimeline[index];
                                return [
                                    `Win Prob: ${context.raw.toFixed(1)}%`,
                                    `Score: ${item.runs_scored} runs (Need ${item.runs_remaining} off ${item.balls_remaining} b)`,
                                    `Wickets Fallen: ${item.wickets_lost}/10`,
                                    `Req. Run Rate: ${item.required_run_rate.toFixed(2)}`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }

    // State for the key-moments panel so the Combined/Wickets/Boundaries toggle can re-render
    // without refetching the match.
    let keyMomentsTimeline = [];
    let keyMomentsFormat = "t20";
    let keyMomentsMode = "combined";

    // Which event types each toggle mode keeps. Combined keeps everything.
    const KEY_MOMENT_FILTERS = {
        combined: null,
        wickets: new Set(["wicket"]),
        boundaries: new Set(["four", "six"]),
    };

    function renderKeyMoments(timeline, formatId) {
        keyMomentsTimeline = timeline;
        keyMomentsFormat = formatId || "t20";
        drawKeyMoments();
    }

    // The three deliveries of the chase that moved the win probability the most (by absolute
    // swing vs the previous ball), filtered to the active toggle mode. Cards under the timeline.
    function drawKeyMoments() {
        const container = document.getElementById("key-moments-list");
        const fmt = FORMATS[keyMomentsFormat || "t20"];
        const sorted = [...keyMomentsTimeline].sort((a, b) => a.balls_faced - b.balls_faced);

        const allow = KEY_MOMENT_FILTERS[keyMomentsMode];
        const swings = [];
        for (let i = 1; i < sorted.length; i++) {
            const it = sorted[i];
            if (allow && !allow.has(it.event_type)) continue;
            swings.push({
                it,
                prev: sorted[i - 1],
                delta: it.win_probability - sorted[i - 1].win_probability,
            });
        }
        swings.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
        const top3 = swings.slice(0, 3);

        if (!top3.length) {
            const noun = keyMomentsMode === "wickets" ? "wickets" : keyMomentsMode === "boundaries" ? "boundaries" : "deliveries";
            container.innerHTML = `<p class="key-moments-empty">No ${noun} to rank for this chase.</p>`;
            return;
        }

        container.innerHTML = top3.map((s, rank) => {
            const it = s.it;
            const overStr = fmt.unit === "balls"
                ? `Ball ${it.balls_faced}`
                : `Over ${Math.floor(it.balls_faced / 6)}.${it.balls_faced % 6}`;
            const ballRuns = Math.max(0, it.runs_scored - s.prev.runs_scored);
            const desc = it.event_type === "wicket" ? "Wicket"
                : it.event_type === "six" ? "SIX!"
                : it.event_type === "four" ? "FOUR!"
                : `${ballRuns} run${ballRuns === 1 ? "" : "s"}`;

            // Batsman line: who was dismissed (and for how many), or who advanced their score.
            let batterLine = "";
            if (it.batter) {
                batterLine = it.event_type === "wicket"
                    ? `${it.batter} dismissed for ${it.batter_score}`
                    : `${it.batter} moved to ${it.batter_score}`;
            }

            const deltaPts = s.delta * 100;
            const up = deltaPts >= 0;
            const color = up ? "#10b981" : "#f43f5e";
            const arrow = up ? "▲" : "▼";
            const before = Math.round(s.prev.win_probability * 100);
            const after = Math.round(it.win_probability * 100);
            return `
                <div class="key-moment" style="border-left-color:${color}">
                    <div class="key-moment-rank">${rank + 1}</div>
                    <div class="key-moment-body">
                        <div class="key-moment-head">
                            <span class="key-moment-over">${overStr}</span>
                            <span class="key-moment-desc">${desc}</span>
                        </div>
                        ${batterLine ? `<div class="key-moment-batter">${batterLine}</div>` : ""}
                        <div class="key-moment-prob">Chase win prob ${before}% → ${after}%</div>
                    </div>
                    <div class="key-moment-delta" style="color:${color}">${arrow} ${up ? "+" : ""}${deltaPts.toFixed(1)} pts</div>
                </div>`;
        }).join("");
    }

    // Run worm: cumulative runs per ball, both innings on one graph. Wickets are marked as
    // enlarged red points on each line. x-axis is ball number so the two innings overlay.
    function plotWorm(inningsCharts, target) {
        if (wormChart) wormChart.destroy();
        const ctx = document.getElementById("worm-chart").getContext("2d");

        const datasets = inningsCharts.map(inn => {
            const color = INNINGS_COLORS[inn.innings] || "#94a3b8";
            return {
                label: `${inn.team} (Inns ${inn.innings})`,
                data: inn.worm.map(p => ({ x: p.ball, y: p.runs })),
                borderColor: color,
                backgroundColor: color,
                borderWidth: 2.5,
                tension: 0.25,
                pointRadius: inn.worm.map(p => p.wicket ? 5 : 0),
                pointHoverRadius: inn.worm.map(p => p.wicket ? 7 : 4),
                pointBackgroundColor: inn.worm.map(p => p.wicket ? "#f43f5e" : color),
                pointBorderColor: inn.worm.map(p => p.wicket ? "#ffffff" : color),
                pointBorderWidth: inn.worm.map(p => p.wicket ? 2 : 0),
            };
        });

        // Dashed target line so the chase can be read against the total to beat.
        const maxBall = Math.max(1, ...inningsCharts.flatMap(inn => inn.worm.map(p => p.ball)));
        const targetLine = (target && target > 0) ? {
            label: `Target ${target}`,
            data: [{ x: 1, y: target }, { x: maxBall, y: target }],
            borderColor: "rgba(148, 163, 184, 0.6)",
            borderWidth: 1.5,
            borderDash: [6, 6],
            pointRadius: 0,
            fill: false,
        } : null;

        wormChart = new Chart(ctx, {
            type: "line",
            data: { datasets: targetLine ? [...datasets, targetLine] : datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "nearest", intersect: false },
                scales: {
                    x: {
                        type: "linear",
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8", maxTicksLimit: 20 },
                        title: { display: true, text: "Balls Faced", color: "#64748b" }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8" },
                        title: { display: true, text: "Cumulative Runs", color: "#64748b" }
                    }
                },
                plugins: {
                    legend: { display: true, labels: { color: "#cbd5e1" } },
                    tooltip: {
                        callbacks: {
                            label: (c) => `${c.dataset.label}: ${c.parsed.y} runs (ball ${c.parsed.x})`
                        }
                    }
                }
            }
        });
    }

    // Manhattan: runs per over as grouped bars, both innings side by side. Overs with a wicket
    // are outlined so they stand out, mirroring the worm's wicket markers.
    function plotManhattan(inningsCharts) {
        if (manhattanChart) manhattanChart.destroy();
        const ctx = document.getElementById("manhattan-chart").getContext("2d");

        const maxOver = Math.max(1, ...inningsCharts.flatMap(inn => inn.manhattan.map(o => o.over)));
        const labels = Array.from({ length: maxOver }, (_, i) => i + 1);

        const datasets = inningsCharts.map(inn => {
            const color = INNINGS_COLORS[inn.innings] || "#94a3b8";
            const byOver = new Map(inn.manhattan.map(o => [o.over, o]));
            return {
                label: `${inn.team} (Inns ${inn.innings})`,
                data: labels.map(ov => byOver.get(ov)?.runs ?? 0),
                _wickets: labels.map(ov => byOver.get(ov)?.wickets ?? 0),
                backgroundColor: color + "cc",
                borderRadius: 3,
            };
        });

        // Stack one red circle per wicket on top of each over's bar so the count is visible at a
        // glance (2 wickets → 2 dots, etc.), mirroring the worm's wicket markers.
        const wicketDotsPlugin = {
            id: "manhattanWicketDots",
            afterDatasetsDraw(chart) {
                const { ctx } = chart;
                chart.data.datasets.forEach((ds, di) => {
                    const meta = chart.getDatasetMeta(di);
                    if (meta.hidden) return;
                    const wkts = ds._wickets || [];
                    meta.data.forEach((bar, i) => {
                        const n = wkts[i] || 0;
                        for (let w = 0; w < n; w++) {
                            const cy = bar.y - 9 - w * 11;  // stack upward above the bar top
                            ctx.beginPath();
                            ctx.arc(bar.x, cy, 4, 0, Math.PI * 2);
                            ctx.fillStyle = "#f43f5e";
                            ctx.strokeStyle = "#ffffff";
                            ctx.lineWidth = 1.5;
                            ctx.fill();
                            ctx.stroke();
                        }
                    });
                });
            }
        };

        manhattanChart = new Chart(ctx, {
            type: "bar",
            plugins: [wicketDotsPlugin],
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: "#94a3b8" },
                        title: { display: true, text: "Over", color: "#64748b" }
                    },
                    y: {
                        beginAtZero: true,
                        grace: "22%",  // headroom above the tallest bar for the wicket dots
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8" },
                        title: { display: true, text: "Runs in Over", color: "#64748b" }
                    }
                },
                plugins: {
                    legend: { display: true, labels: { color: "#cbd5e1" } },
                    tooltip: {
                        callbacks: {
                            label: (c) => {
                                const wkts = c.dataset._wickets?.[c.dataIndex] || 0;
                                const wktStr = wkts ? ` · ${wkts} wkt${wkts > 1 ? "s" : ""}` : "";
                                return `${c.dataset.label}: ${c.parsed.y} runs${wktStr}`;
                            }
                        }
                    }
                }
            }
        });
    }


    // --------------------------------------------------
    // MONTE CARLO SIMULATOR
    // --------------------------------------------------
    let mcHistogramChart = null;
    let lastMcData = null;   // cache of the most recent /api/montecarlo response for side-switching
    // Determinate loading bar factory for single blocking requests that emit no progress events.
    // The fill tracks elapsed/estimate: it fills to ~90% at the estimated finish, then eases
    // asymptotically toward 98% if the run overruns, so it never stalls or claims completion
    // early. finish() snaps to 100% and refines the estimate (EMA) from the actual elapsed time;
    // cancel() resets it on failure. The estimate is per-bar, adaptive, and persisted across
    // reloads under `storageKey`. Used by both the Monte Carlo sim and the Explorer timeline.
    function makeProgressBar(barId, pctId, storageKey, defaultMs) {
        let estimatedMs = parseInt(localStorage.getItem(storageKey), 10) || defaultMs;
        return function start() {
            const bar = document.getElementById(barId);
            const pct = document.getElementById(pctId);
            const startT = performance.now();
            let rafId = null, stopped = false;

            function apply(p) {
                if (bar) bar.style.width = p.toFixed(1) + "%";
                if (pct) pct.textContent = Math.round(p) + "%";
            }
            function frame(now) {
                const ratio = (now - startT) / estimatedMs;
                const p = ratio <= 1 ? ratio * 90 : 90 + (1 - Math.exp(-(ratio - 1))) * 8;
                apply(Math.min(p, 98));
                if (!stopped) rafId = requestAnimationFrame(frame);
            }
            apply(0);
            rafId = requestAnimationFrame(frame);

            return {
                finish() {
                    stopped = true;
                    if (rafId) cancelAnimationFrame(rafId);
                    apply(100);
                    const actual = performance.now() - startT;
                    estimatedMs = Math.round(estimatedMs * 0.4 + actual * 0.6);  // EMA toward reality
                    localStorage.setItem(storageKey, estimatedMs);
                },
                cancel() {
                    stopped = true;
                    if (rafId) cancelAnimationFrame(rafId);
                    apply(0);
                }
            };
        };
    }

    const runMcProgress = makeProgressBar("mc-progress-bar", "mc-progress-pct", "mcEstimatedMs", 6000);

    // MC market view state: which team (a/b) and which discipline (batting/bowling).
    let mcSide = "a";
    let mcRole = "batting";

    function mcOverUnderHTML(dist) {
        return `
            <div class="ou-head">Total runs — line ${dist.line}</div>
            <div class="ou-row">
                <div class="ou-cell over">
                    <span class="ou-label">Over ${dist.line}</span>
                    <span class="ou-prob">${(dist.p_over * 100).toFixed(1)}%</span>
                    <span class="ou-odds">Fair ${dist.over_odds}</span>
                </div>
                <div class="ou-cell under">
                    <span class="ou-label">Under ${dist.line}</span>
                    <span class="ou-prob">${(dist.p_under * 100).toFixed(1)}%</span>
                    <span class="ou-odds">Fair ${dist.under_odds}</span>
                </div>
            </div>`;
    }

    // Shared by the Match Explorer and Live Games Monte Carlo panels — draws (or redraws) the
    // score/wickets histogram into whichever canvas + chart-instance the caller owns.
    function drawHistogramChart(hist, borderColor, canvasId) {
        const existing = Chart.getChart(canvasId);
        if (existing) existing.destroy();
        const ctx = document.getElementById(canvasId).getContext("2d");
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: hist.labels,
                datasets: [{
                    label: 'Simulations',
                    data: hist.counts,
                    backgroundColor: 'rgba(0, 242, 254, 0.35)',
                    borderColor,
                    borderWidth: 1,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: "#94a3b8" } },
                    y: { display: false }
                },
                plugins: { legend: { display: false }, tooltip: { enabled: true } }
            }
        });
    }

    // Shared market-slot renderer for both MC panels. `obj` is null when the backend has
    // deliberately omitted a market (e.g. a live match's already-settled 1st-innings props).
    function setMarketPanel(prefix, n, title, obj, prop, color) {
        const titleEl = document.getElementById(`${prefix}-m${n}-title`);
        const listEl = document.getElementById(`${prefix}-m${n}-list`);
        if (titleEl) titleEl.textContent = title;
        if (listEl) listEl.innerHTML = obj
            ? renderLeaderboard(obj, prop, color)
            : '<div class="u-text-secondary u-text-sm">Not available once this innings is settled.</div>';
    }

    function mcSetMarket(n, title, obj, prop, color) {
        setMarketPanel("mc", n, title, obj, prop, color);
    }

    // Render the selected team's exotic markets for the selected discipline. The team batting
    // first is Team A, so Team A bats in inn1 / bowls in inn2 and Team B bats in inn2 / bowls
    // in inn1 — batting markets read the team's batting innings, bowling markets the other one.
    function renderMcMarkets() {
        const data = lastMcData;
        if (!data) return;
        const isB = mcSide === "b";
        const teamName = isB ? data.team_b : data.team_a;
        const v = teamVisual(teamName);
        const batKey = isB ? "inn2" : "inn1";     // innings this team bats
        const bowlKey = isB ? "inn1" : "inn2";    // innings this team bowls
        const battingBlock = document.getElementById("mc-batting-block");

        if (mcRole === "batting") {
            battingBlock.style.display = "";
            const dist = data[batKey];
            document.getElementById("mc-stat-tiles").innerHTML =
                statTilesHTML(dist, `${(10000).toLocaleString()} sims`);
            document.getElementById("mc-overunder").innerHTML = mcOverUnderHTML(dist);

            document.getElementById("mc-hist-title").textContent = `Projected final score — ${teamName}`;
            mcHistogramChart = drawHistogramChart(data["hist_" + batKey], v.color, "mc-histogram");

            mcSetMarket(1, `Top Batsman — ${teamName}`, data["top_batsman_" + batKey], "p_top", v.color);
            mcSetMarket(2, `30+ Runs — ${teamName}`, data["props_30_" + batKey], "p_hit", v.color);
            mcSetMarket(3, `50+ Runs — ${teamName}`, data["props_50_" + batKey], "p_hit", v.color);
        } else {
            battingBlock.style.display = "none";
            document.getElementById("mc-hist-title").textContent = `Projected wickets taken — ${teamName}`;
            mcHistogramChart = drawHistogramChart(data["wkts_hist_" + bowlKey], v.color, "mc-histogram");

            mcSetMarket(1, `Most Wickets — ${teamName}`, data["top_bowler_" + bowlKey], "p_top", v.color);
            mcSetMarket(2, `2+ Wickets — ${teamName}`, data["bowler_2w_" + bowlKey], "p_hit", v.color);
            mcSetMarket(3, `3+ Wickets — ${teamName}`, data["bowler_3w_" + bowlKey], "p_hit", v.color);
        }

        document.querySelectorAll("#mc-subtabs .mc-subtab").forEach(b =>
            b.classList.toggle("active", b.dataset.mcSide === mcSide));
        document.querySelectorAll("#mc-roletabs .mc-subtab").forEach(b =>
            b.classList.toggle("active", b.dataset.mcRole === mcRole));
    }

    // Match-level markets: projected victory margins (runs if the side batting first wins,
    // wickets if the chasing side wins). The tie price is shown on the win bar's divider.
    // `m` is a /api/montecarlo (or /api/live/montecarlo) `margin` object.
    function matchMarketsHTML(m) {
        return `
            <div class="mm-chip">
                <span class="mm-label">Avg margin · bat-first win</span>
                <span class="mm-val">${m.mean_run_margin} <small>runs</small></span>
            </div>
            <div class="mm-chip">
                <span class="mm-label">Avg margin · chase win</span>
                <span class="mm-val">${m.mean_wkt_margin} <small>wkts</small></span>
            </div>`;
    }

    function renderMcMatchMarkets(m) {
        document.getElementById("mc-match-markets").innerHTML = matchMarketsHTML(m);
    }

    // Team A / Team B and Batting / Bowling tab switching for the exotic markets.
    document.querySelectorAll("#mc-subtabs .mc-subtab").forEach(btn =>
        btn.addEventListener("click", () => { mcSide = btn.dataset.mcSide; renderMcMarkets(); }));
    document.querySelectorAll("#mc-roletabs .mc-subtab").forEach(btn =>
        btn.addEventListener("click", () => { mcRole = btn.dataset.mcRole; renderMcMarkets(); }));

    function populateMCDropdowns() {
        if (!metadataCache) return;
        const mcFormatEl = document.getElementById("mc-format");
        if (!mcFormatEl) return;
        const formatVal = mcFormatEl.value;
        const teamsDict = teamsForFormat(formatVal);
        
        const teamASelect = document.getElementById("mc-team-a");
        const teamBSelect = document.getElementById("mc-team-b");
        
        const curA = teamASelect.value;
        const curB = teamBSelect.value;
        
        teamASelect.innerHTML = '';
        teamBSelect.innerHTML = '';
        
        for (const [team, elo] of Object.entries(teamsDict || {})) {
            teamASelect.add(new Option(`${team} (${elo} Elo)`, team));
            teamBSelect.add(new Option(`${team} (${elo} Elo)`, team));
        }
        
        if (curA && teamsDict && teamsDict[curA]) teamASelect.value = curA;
        if (curB && teamsDict && teamsDict[curB]) teamBSelect.value = curB;
        else if (teamBSelect.options.length > 1) teamBSelect.selectedIndex = 1;

        const mcSourceSel = document.getElementById("mc-source");
        mcSourceSel.value = ratingSource;
        refreshSourceSelectOptions(mcSourceSel, formatVal);
    }

    on("mc-format", "change", populateMCDropdowns);
    on("mc-source", "change", (e) => { setRatingSource(e.target.value); syncMcPanelToSelection(); });
    
    // Add to metadata fetching
    const origFetch = fetchMetadata;
    fetchMetadata = async function() {
        const res = await origFetch();
        populateMCDropdowns();
        return res;
    };
    
    // --------------------------------------------------
    // MC result cache + de-duplicated requests. No sim runs automatically — every entry only
    // exists because the user pressed "Run Simulation" at some point. This cache exists so that
    // switching tabs (or back to a previously-viewed matchup) never loses or re-triggers a result,
    // and so a still-running request is reused rather than duplicated if it's revisited mid-flight.
    // --------------------------------------------------
    const mcCache = new Map();
    const mcInFlight = new Map();
    function mcKey(format, teamA, teamB, source) { return `${format}:${teamA}::${teamB}:${source}`; }

    // Resolves with the /api/montecarlo response for (format, teamA, teamB, source): a cache
    // hit resolves immediately, an in-flight request is reused, otherwise a fresh POST is made.
    function getOrFetchMc(format, teamA, teamB, source) {
        const key = mcKey(format, teamA, teamB, source);
        if (mcCache.has(key)) return Promise.resolve(mcCache.get(key));
        if (mcInFlight.has(key)) return mcInFlight.get(key);
        const p = fetch("/api/montecarlo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ format, team_a: teamA, team_b: teamB, n_sims: 10000, source })
        })
            .then(res => res.json())
            .then(data => {
                mcInFlight.delete(key);
                if (!data.error) mcCache.set(key, data);
                return data;
            })
            .catch(err => { mcInFlight.delete(key); throw err; });
        mcInFlight.set(key, p);
        return p;
    }

    // The matchup key for whatever the MC form currently holds, so an async response can check
    // whether it's still relevant before touching the DOM.
    function currentMcSelectionKey() {
        return mcKey(
            document.getElementById("mc-format").value,
            document.getElementById("mc-team-a").value,
            document.getElementById("mc-team-b").value,
            document.getElementById("mc-source").value
        );
    }

    // Reflects whatever is already known about the current MC form selection: renders cached
    // results instantly, attaches to a still-running background prefetch (showing the loading
    // bar) so it appears the moment that finishes, or — if neither exists — leaves the panel
    // alone so the user presses "Run Simulation" themselves.
    let mcSurfacedKey = null;  // avoids attaching a duplicate progress-bar watcher to the same in-flight job
    function syncMcPanelToSelection() {
        const key = currentMcSelectionKey();

        if (mcCache.has(key)) {
            renderMcData(mcCache.get(key));
            document.getElementById("mc-loading").style.display = "none";
            document.getElementById("mc-results").style.display = "block";
            mcHistogramChart?.resize();
            return;
        }

        if (mcInFlight.has(key)) {
            document.getElementById("mc-results").style.display = "none";
            document.getElementById("mc-loading").style.display = "block";
            if (mcSurfacedKey !== key) {
                mcSurfacedKey = key;
                const progress = runMcProgress();
                mcInFlight.get(key).then(data => {
                    if (currentMcSelectionKey() !== key) return;  // user has since moved on
                    if (data.error) {
                        progress.cancel();
                        document.getElementById("mc-loading").style.display = "none";
                        return;
                    }
                    progress.finish();
                    document.getElementById("mc-loading").style.display = "none";
                    renderMcData(data);
                    document.getElementById("mc-results").style.display = "block";
                    mcHistogramChart?.resize();
                }).catch(() => {
                    document.getElementById("mc-loading").style.display = "none";
                });
            }
        }
        // Neither cached nor in-flight: this is the first panel loaded for this match — leave
        // the results panel exactly as-is and wait for an explicit "Run Simulation" click.
    }

    // Populates the win split, match markets, and leaderboards from an /api/montecarlo response.
    // Shared by the explicit "Run Simulation" flow and the background-prefetch flows above.
    function renderMcData(data) {
        setSourceBadge("mc-source-badge", data.source || "elo");
        document.getElementById("mc-experimental-warning").style.display = data.simulation_experimental ? "block" : "none";
        // Win prediction — proportional team-coloured split (1st vs 2nd batting side)
        const pA = data.margin.p_first_bat_win;
        const pB = data.margin.p_second_bat_win;
        renderWinSplit("mc-winsplit",
            { name: data.team_a, prob: pA, odds: data.margin.first_bat_odds },
            { name: data.team_b, prob: pB, odds: data.margin.second_bat_odds },
            { prob: data.margin.p_tie, odds: data.margin.tie_odds });

        // Match-level markets: tie price + projected victory margins
        renderMcMatchMarkets(data.margin);

        // Cache the response and expose both sides' exotic markets via the Team + Batting/Bowling tabs
        lastMcData = data;
        document.querySelector('#mc-subtabs .mc-subtab[data-mc-side="a"]').textContent = data.team_a;
        document.querySelector('#mc-subtabs .mc-subtab[data-mc-side="b"]').textContent = data.team_b;
        mcSide = "a";
        mcRole = "batting";
        renderMcMarkets();
    }

    // Explicit "Run Simulation" — the only way a sim starts when MC is the *first* panel loaded
    // for a match (or for a fully custom matchup). Reuses a cached/in-flight result if one
    // already exists rather than firing a duplicate request.
    on("mc-form", "submit", (e) => {
        e.preventDefault();
        runMcSimulation(
            document.getElementById("mc-format").value,
            document.getElementById("mc-team-a").value,
            document.getElementById("mc-team-b").value
        );
    });

    async function runMcSimulation(format, teamA, teamB) {
        document.getElementById("explorer-mc").style.display = "block";  // ensure the panel is visible
        document.getElementById("mc-results").style.display = "none";
        document.getElementById("mc-loading").style.display = "block";
        const progress = runMcProgress();
        const source = document.getElementById("mc-source").value;

        try {
            const data = await getOrFetchMc(format, teamA, teamB, source);

            if (data.error) {
                progress.cancel();
                document.getElementById("mc-loading").style.display = "none";
                showToast(data.error, "error");
                return;
            }
            progress.finish();
            document.getElementById("mc-loading").style.display = "none";
            renderMcData(data);
            document.getElementById("mc-results").style.display = "block";

        } catch(err) {
            console.error(err);
            progress.cancel();
            document.getElementById("mc-loading").style.display = "none";
            showToast("Simulation failed.", "error");
        }
    }

    // --------------------------------------------------
    // LIVE GAMES (scrape → select → live sims)
    // --------------------------------------------------
    let selectedLiveGame = null;

    async function loadLiveGames() {
        const statusEl = document.getElementById("live-games-status");
        const listEl = document.getElementById("live-games-list");
        if (!statusEl || !listEl) return;
        statusEl.textContent = "Loading live games…";
        listEl.innerHTML = "";
        try {
            const res = await fetch("/api/live/games");
            const games = await res.json();
            if (games.error) { statusEl.textContent = `Error: ${games.error}`; return; }
            const liveCount = games.filter(g => g.state === "in").length;
            statusEl.textContent = games.length
                ? `${liveCount} live · ${games.length - liveCount} upcoming`
                : "No cricket matches found right now.";
            listEl.innerHTML = games.map((g, i) => renderGameCard(g, i)).join("");
            games.forEach((g, i) => {
                const card = document.getElementById(`live-game-${i}`);
                if (card) card.addEventListener("click", () => selectLiveGame(g));
            });
        } catch (e) {
            console.error(e);
            statusEl.textContent = "Failed to fetch live games.";
        }
    }

    function renderGameCard(g, i) {
        const isLive = g.state === "in";
        const badge = isLive
            ? `<span style="color:#f43f5e;">● LIVE</span>`
            : `<span style="color:#94a3b8;">Upcoming</span>`;
        const scoreA = g.team_a_score ? ` — ${g.team_a_score}` : "";
        const scoreB = g.team_b_score ? ` — ${g.team_b_score}` : "";
        return `
            <div id="live-game-${i}" class="glass-card" style="cursor:pointer; padding:14px; display:flex; justify-content:space-between; align-items:center; gap:12px;">
                <div>
                    <div style="font-weight:600;">${g.team_a}${scoreA}</div>
                    <div style="font-weight:600;">${g.team_b}${scoreB}</div>
                    <div style="color:#94a3b8; font-size:0.8rem; margin-top:4px;">${g.league_name}</div>
                </div>
                <div style="text-align:right; white-space:nowrap;">
                    ${badge}
                    <div style="text-transform:uppercase; font-size:0.75rem; color:#4facfe; margin-top:4px;">${g.format}</div>
                </div>
            </div>`;
    }

    async function selectLiveGame(g) {
        selectedLiveGame = g;
        const detail = document.getElementById("live-detail");
        detail.style.display = "block";
        document.getElementById("live-mc-results").style.display = "none";
        document.getElementById("live-mc-loading").style.display = "none";
        document.getElementById("live-squad-preview").style.display = "none";
        document.getElementById("live-detail-title").textContent = `${g.team_a} v ${g.team_b}`;
        document.getElementById("live-detail-sub").textContent = `${g.league_name} · ${g.format.toUpperCase()}`;
        document.getElementById("live-scoreline").textContent = "Loading state…";
        document.getElementById("live-scoreline-meta").textContent = "";
        detail.scrollIntoView({ behavior: "smooth", block: "start" });

        const liveSourceSel = document.getElementById("live-source");
        liveSourceSel.value = ratingSource;
        refreshSourceSelectOptions(liveSourceSel, g.format);

        // ODI Monte Carlo is wired up but EXPERIMENTAL: the delivery model passes its own
        // per-ball check, but the full-match calibration battery (innings MAE, win-prob
        // agreement) has only ever been run for T20 — see od_delivery_model in app.py and
        // the "simulation_experimental" flag echoed back by /api/live/montecarlo.
        const mcBtn = document.getElementById("live-mc-btn");
        mcBtn.disabled = false;
        mcBtn.textContent = g.format === "od"
            ? "Run Monte Carlo (experimental — ODI unvalidated)"
            : "Run Monte Carlo (exotic markets)";
        document.getElementById("live-mc-experimental-warning").style.display = g.format === "od" ? "block" : "none";

        // Fires alongside the state fetch below rather than after it — an independent panel,
        // no reason to make the user wait for both sequentially.
        loadLiveSquadPreview(g);

        await refreshLiveState(g);
    }

    // Fetches and renders /api/live/state for `g` at the currently-selected rating source.
    // Split out of selectLiveGame so the source toggle can re-fetch without re-running that
    // function's full panel reset (scroll-into-view, clearing MC results, etc.).
    async function refreshLiveState(g) {
        const source = document.getElementById("live-source").value;
        try {
            const res = await fetch(`/api/live/state/${g.event_id}?format=${g.format}&league_id=${g.league_id}&source=${source}`);
            const st = await res.json();
            if (st.error) { document.getElementById("live-scoreline").textContent = `Error: ${st.error}`; return; }
            renderLiveState(st);
        } catch (e) {
            console.error(e);
            document.getElementById("live-scoreline").textContent = "Failed to load match state.";
        }
    }

    on("live-source", "change", (e) => {
        setRatingSource(e.target.value);
        if (selectedLiveGame) refreshLiveState(selectedLiveGame);
    });

    // Lineups & Elo preview — /api/live/squads resolves squads without running the (slower)
    // Monte Carlo simulation, so this shows up as soon as a match is selected. Also flags
    // sides where too many live-XI players have no historical rating (build_live_team()'s
    // per-player `rated` flag) so squad/effective Elo confidence is visible up front, before
    // the user spends a simulation run on a match we can't price well.
    async function loadLiveSquadPreview(g) {
        const preview = document.getElementById("live-squad-preview");
        try {
            const res = await fetch(`/api/live/squads?event_id=${g.event_id}&format=${g.format}&league_id=${g.league_id}`);
            const data = await res.json();
            if (data.error) { console.warn("Live squad preview:", data.error); return; }

            renderSquadCard("live-lineups-team-1", data.first_batting_squad, {
                overall: data.first_batting_overall_elo,
                battingSquad: data.first_batting_squad_elo,
                bowlingSquad: data.first_bowling_squad_elo,
                effective: data.first_effective_elo,
            });
            renderSquadCard("live-lineups-team-2", data.second_batting_squad, {
                overall: data.second_batting_overall_elo,
                battingSquad: data.second_batting_squad_elo,
                bowlingSquad: data.second_bowling_squad_elo,
                effective: data.second_effective_elo,
            });
            document.getElementById("live-mc-squad-note").textContent =
                `Squads: batting ${data.squad_source.batting}, bowling ${data.squad_source.bowling}`;
            renderSquadQualityWarning(data);
            preview.style.display = "block";
        } catch (e) {
            console.error("Failed to fetch live squad preview", e);
        }
    }

    // Shared by the pre-MC preview (/api/live/squads) and the post-MC results (/api/live/
    // montecarlo) — both responses carry the same {first,second}_batting_team/_squad shape.
    function renderSquadQualityWarning(data) {
        const el = document.getElementById("live-squad-quality-warning");
        const threshold = data.low_confidence_threshold != null ? data.low_confidence_threshold : 3;
        const sides = [
            { name: data.first_batting_team, quality: data.first_batting_squad && data.first_batting_squad.quality },
            { name: data.second_batting_team, quality: data.second_batting_squad && data.second_batting_squad.quality },
        ].filter(s => s.quality && s.quality.unrated_count >= threshold);

        if (!sides.length) { el.style.display = "none"; el.textContent = ""; return; }
        const parts = sides.map(s => `${s.name} (${s.quality.unrated_count}/${s.quality.total} players unrated)`);
        el.textContent = `⚠ Limited player data — ${parts.join(", ")}. Squad/effective Elo for `
            + `${sides.length > 1 ? "these sides" : "this side"} may be unreliable.`;
        el.style.display = "block";
    }

    function renderLiveState(st) {
        document.getElementById("live-scoreline").textContent =
            `${st.batting_team}: ${st.runs}/${st.wickets}` + (st.target ? `  (chasing ${st.target})` : "");
        const meta = [`Innings ${st.innings}`, `${st.overs} ov`, st.venue];
        if (st.required_run_rate != null) meta.push(`RRR ${st.required_run_rate}`);
        if (st.run_rate != null) meta.push(`RR ${st.run_rate}`);
        document.getElementById("live-scoreline-meta").textContent = meta.filter(Boolean).join("  ·  ");

        const fiEl = document.getElementById("live-first-innings");
        if (st.predicted_first_innings_score != null) {
            setSourceBadge("live-fi-source-badge", st.rating_source);
            document.getElementById("live-fi-predicted-display").textContent = st.predicted_first_innings_score;
            document.getElementById("live-fi-proj-display").textContent = st.first_innings_projected_score;
            document.getElementById("live-fi-crr-display").textContent = st.run_rate != null ? st.run_rate.toFixed(2) : "--";
            fiEl.style.display = "block";
        } else {
            fiEl.style.display = "none";
        }

        const splitEl = document.getElementById("live-winsplit");
        const sourceEl = document.getElementById("live-wp-source");
        // Prefer the Elo-matched names so Hundred franchise brand colours resolve
        // (raw ESPN names carry a "(Men)"/"(Women)" suffix that won't match).
        const batName = st.batting_team_matched && st.batting_team_matched !== "Unknown" ? st.batting_team_matched : st.batting_team;
        const bowlName = st.bowling_team_matched && st.bowling_team_matched !== "Unknown" ? st.bowling_team_matched : st.bowling_team;
        const ratingBadge = sourceBadgeHtml(st.rating_source);
        if (st.win_prob_batting != null) {
            const pB = st.win_prob_batting;
            splitEl.innerHTML = winSplitHTML(
                { name: batName, prob: pB },
                { name: bowlName, prob: 1 - pB });
            sourceEl.innerHTML = `Win probability — XGBoost chase model ${ratingBadge}`;
        } else {
            // 1st innings: no chase model yet — the Monte Carlo below produces the win prob.
            splitEl.innerHTML =
                '<div class="winsplit-empty">Win probability appears once the chase begins — or run the Monte Carlo below.</div>';
            sourceEl.innerHTML = "";
        }
    }

    async function runLiveMC() {
        if (!selectedLiveGame) return;
        const g = selectedLiveGame;
        const source = document.getElementById("live-source").value;
        document.getElementById("live-mc-results").style.display = "none";
        document.getElementById("live-mc-loading").style.display = "block";
        try {
            const res = await fetch("/api/live/montecarlo", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ event_id: g.event_id, format: g.format, league_id: g.league_id, n_sims: 5000, source })
            });
            const mc = await res.json();
            document.getElementById("live-mc-loading").style.display = "none";
            if (mc.error) { showToast(mc.error, "error"); return; }
            renderLiveMC(mc);
        } catch (e) {
            console.error(e);
            document.getElementById("live-mc-loading").style.display = "none";
            showToast("Live simulation failed.", "error");
        }
    }

    // Live MC market view state: mirrors mcSide/mcRole above, but "1"/"2" (first-/second-
    // batting side) instead of "a"/"b" — a live match has no fixed "Team A/B" identity, only
    // whichever side batted in inn1 vs inn2.
    let lastLiveMcData = null;
    let liveMcSide = "1";
    let liveMcRole = "batting";

    // Same shape as renderMcMarkets() (Match Explorer), reading from lastLiveMcData/liveMcSide/
    // liveMcRole instead. Kept as a separate function (rather than parameterizing one shared
    // function over both panels) since the two panels' state variables and default-selection
    // logic differ enough that forcing them through one call site would add more indirection
    // than it saves.
    function renderLiveMcMarkets() {
        const data = lastLiveMcData;
        if (!data) return;
        const isSecond = liveMcSide === "2";
        const teamName = isSecond ? data.second_batting_team : data.first_batting_team;
        const v = teamVisual(teamName);
        const batKey = isSecond ? "inn2" : "inn1";
        const bowlKey = isSecond ? "inn1" : "inn2";
        const battingBlock = document.getElementById("live-mc-batting-block");

        if (liveMcRole === "batting") {
            battingBlock.style.display = "";
            const dist = data[batKey];
            document.getElementById("live-mc-stat-tiles").innerHTML = statTilesHTML(dist, "5,000 sims");
            document.getElementById("live-mc-overunder").innerHTML = mcOverUnderHTML(dist);

            document.getElementById("live-mc-hist-title").textContent = `Projected final score — ${teamName}`;
            drawHistogramChart(data["hist_" + batKey], v.color, "live-mc-histogram");

            setMarketPanel("live-mc", 1, `Top Batsman — ${teamName}`, data["top_batsman_" + batKey], "p_top", v.color);
            setMarketPanel("live-mc", 2, `30+ Runs — ${teamName}`, data["props_30_" + batKey], "p_hit", v.color);
            setMarketPanel("live-mc", 3, `50+ Runs — ${teamName}`, data["props_50_" + batKey], "p_hit", v.color);
        } else {
            battingBlock.style.display = "none";
            document.getElementById("live-mc-hist-title").textContent = `Projected wickets taken — ${teamName}`;
            drawHistogramChart(data["wkts_hist_" + bowlKey], v.color, "live-mc-histogram");

            setMarketPanel("live-mc", 1, `Most Wickets — ${teamName}`, data["top_bowler_" + bowlKey], "p_top", v.color);
            setMarketPanel("live-mc", 2, `2+ Wickets — ${teamName}`, data["bowler_2w_" + bowlKey], "p_hit", v.color);
            setMarketPanel("live-mc", 3, `3+ Wickets — ${teamName}`, data["bowler_3w_" + bowlKey], "p_hit", v.color);
        }

        document.querySelectorAll("#live-mc-subtabs .mc-subtab").forEach(b =>
            b.classList.toggle("active", b.dataset.mcSide === liveMcSide));
        document.querySelectorAll("#live-mc-roletabs .mc-subtab").forEach(b =>
            b.classList.toggle("active", b.dataset.mcRole === liveMcRole));
    }

    document.querySelectorAll("#live-mc-subtabs .mc-subtab").forEach(btn =>
        btn.addEventListener("click", () => { liveMcSide = btn.dataset.mcSide; renderLiveMcMarkets(); }));
    document.querySelectorAll("#live-mc-roletabs .mc-subtab").forEach(btn =>
        btn.addEventListener("click", () => { liveMcRole = btn.dataset.mcRole; renderLiveMcMarkets(); }));

    function renderLiveMC(mc) {
        document.getElementById("live-squad-preview").style.display = "block";
        // Backend is the source of truth for experimental status (mirrors the pre-run check
        // in selectLiveGame, which only knows the format, not the actual response flag).
        document.getElementById("live-mc-experimental-warning").style.display = mc.simulation_experimental ? "block" : "none";
        document.getElementById("live-mc-squad-note").innerHTML =
            `Squads: batting ${mc.squad_source.batting}, bowling ${mc.squad_source.bowling}` +
            (mc.inn1_settled ? "  ·  1st-innings props settled (hidden)" : "") +
            `  ${sourceBadgeHtml(mc.source)}`;

        // Lineups & Elo — the exact squads simulate_match() ran on for this live game.
        renderSquadCard("live-lineups-team-1", mc.first_batting_squad, {
            overall: mc.first_batting_overall_elo,
            battingSquad: mc.first_batting_squad_elo,
            bowlingSquad: mc.first_bowling_squad_elo,
            effective: mc.first_effective_elo,
        });
        renderSquadCard("live-lineups-team-2", mc.second_batting_squad, {
            overall: mc.second_batting_overall_elo,
            battingSquad: mc.second_batting_squad_elo,
            bowlingSquad: mc.second_bowling_squad_elo,
            effective: mc.second_effective_elo,
        });
        renderSquadQualityWarning(mc);

        // Match-result win split (1st vs 2nd batting side), in team colours, tie on the divider.
        renderWinSplit("live-mc-winsplit",
            { name: mc.first_batting_team, prob: mc.margin.p_first_bat_win, odds: mc.margin.first_bat_odds },
            { name: mc.second_batting_team, prob: mc.margin.p_second_bat_win, odds: mc.margin.second_bat_odds },
            { prob: mc.margin.p_tie, odds: mc.margin.tie_odds });

        document.getElementById("live-mc-match-markets").innerHTML = matchMarketsHTML(mc.margin);

        lastLiveMcData = mc;
        document.querySelector('#live-mc-subtabs .mc-subtab[data-mc-side="1"]').textContent = mc.first_batting_team;
        document.querySelector('#live-mc-subtabs .mc-subtab[data-mc-side="2"]').textContent = mc.second_batting_team;
        // Default to whichever side is currently batting, so the panel opens on the live action
        // rather than the (possibly already-settled) 1st innings.
        liveMcSide = mc.inn1_settled ? "2" : "1";
        liveMcRole = "batting";
        renderLiveMcMarkets();

        document.getElementById("live-mc-results").style.display = "block";
    }

    on("live-refresh-btn", "click", loadLiveGames);
    on("live-mc-btn", "click", runLiveMC);

    // Load matches list on start
    loadMatchCatalog();


    // --------------------------------------------------
    // Analytics Tab Rendering
    // --------------------------------------------------
    let brierChart = null;
    let calCurveChart = null;
    let fipPhaseMaeChart = null;
    let analyticsLoaded = false;
    let analyticsCurrentView = "overview";
    const mcPerformanceLoadedFor = new Set();
    const featureImportanceCache = {};

    function currentAnalyticsFormat() {
        const formatSelect = document.getElementById("analytics-format-select");
        return formatSelect ? formatSelect.value : "t20";
    }

    async function loadAnalytics() {
        const formatVal = currentAnalyticsFormat();

        try {
            const res = await fetch(`/api/accuracy?format=${formatVal}`);
            const data = await res.json();

            // available===false means the accuracy report hasn't been generated for this format
            // yet (see scripts/build_accuracy_report.py). Bail out rather than feeding undefined
            // into the charts -- previously impossible, since every format fell back to t20's
            // hardcoded numbers, which silently mislabelled them as that format's own.
            if (data.available === false) {
                console.warn(`No accuracy report for '${formatVal}': ${data.message || ""}`);
                showToast(`Accuracy metrics not yet generated for ${formatVal.toUpperCase()}.`, "error");
            } else {
                plotBrierScores(data.phase_brier);
                plotCalibrationCurve(data.calibration);
                renderFirstInningsPerformance(data.first_innings, formatVal);
                analyticsLoaded = true;
            }
        } catch (err) {
            console.error("Failed to load analytics metrics", err);
        }
        // NB: deliberately outside the guard above -- Monte Carlo calibration exists for all six
        // formats even when the accuracy report for one doesn't, so an early return here would
        // hide the women's MC data this selector was widened to expose.

        loadFeatureImportance(formatVal);
        if (!mcPerformanceLoadedFor.has(formatVal)) loadMcPerformance(formatVal);
    }

    // Attach event listener for format selector in Analytics tab
    const analyticsFormatSelect = document.getElementById("analytics-format-select");
    if (analyticsFormatSelect) {
        analyticsFormatSelect.addEventListener("change", () => {
            loadAnalytics();
        });
    }

    // Sub-view switching (Accuracy Overview | Feature Importance | First Innings | Monte Carlo Sim)
    const analyticsViewTabs = document.getElementById("analytics-view-tabs");
    if (analyticsViewTabs) {
        analyticsViewTabs.addEventListener("click", (e) => {
            const btn = e.target.closest(".mc-subtab");
            if (!btn) return;
            const view = btn.dataset.analyticsView;
            analyticsCurrentView = view;
            document.querySelectorAll("#analytics-view-tabs .mc-subtab").forEach(b =>
                b.classList.toggle("active", b === btn));
            document.querySelectorAll(".analytics-view").forEach(v => {
                v.style.display = (v.id === `analytics-view-${view}`) ? "" : "none";
            });
        });
    }

    function plotBrierScores(phaseBrier) {
        if (brierChart) {
            brierChart.destroy();
        }
        const ctx = document.getElementById("brier-bar-chart").getContext("2d");
        
        brierChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Powerplay", "Middle Overs", "Death Overs"],
                datasets: [{
                    label: "Brier Score",
                    data: [phaseBrier.powerplay, phaseBrier.middle, phaseBrier.death],
                    backgroundColor: [
                        "rgba(244, 63, 94, 0.75)",  // Powerplay (Pink)
                        "rgba(59, 130, 246, 0.75)",  // Middle (Blue)
                        "rgba(16, 185, 129, 0.75)"   // Death (Green)
                    ],
                    borderColor: [
                        "#f43f5e",
                        "#3b82f6",
                        "#10b981"
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: "#94a3b8" }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8" },
                        min: 0,
                        max: 0.25,
                        title: { display: true, text: "Brier Score (Guessing = 0.25)", color: "#64748b" }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    function plotCalibrationCurve(calibration) {
        if (calCurveChart) {
            calCurveChart.destroy();
        }
        const ctx = document.getElementById("calibration-curve-chart").getContext("2d");
        
        const predictedVals = calibration.map(item => item.predicted * 100);
        const actualVals = calibration.map(item => item.actual * 100);

        calCurveChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: predictedVals,
                datasets: [
                    {
                        label: "Model Calibration",
                        data: actualVals,
                        borderColor: "#00f2fe",
                        backgroundColor: "rgba(0, 242, 254, 0.2)",
                        borderWidth: 3,
                        tension: 0.1,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        pointBackgroundColor: "#00f2fe"
                    },
                    {
                        label: "Perfect Calibration",
                        data: [5, 15, 25, 35, 45, 55, 65, 75, 85, 95], // Diagonal reference line
                        borderColor: "rgba(255, 255, 255, 0.25)",
                        borderDash: [5, 5],
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8", callback: (val) => `${val}%` },
                        title: { display: true, text: "Predicted Win Probability", color: "#64748b" }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8", callback: (val) => `${val}%` },
                        min: 0,
                        max: 100,
                        title: { display: true, text: "Actual Win Frequency", color: "#64748b" }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: "#94a3b8", boxWidth: 12 }
                    }
                }
            }
        });
    }

    // --------------------------------------------------
    // Analytics: Feature Importance
    // --------------------------------------------------
    async function loadFeatureImportance(formatVal) {
        const unavailableEl = document.getElementById("fi-importance-unavailable");
        const unavailableMsg = document.getElementById("fi-importance-unavailable-msg");
        const contentEl = document.getElementById("fi-importance-content");
        if (!unavailableEl || !contentEl) return;

        try {
            let data = featureImportanceCache[formatVal];
            if (!data) {
                const res = await fetch(`/api/feature-importance?format=${formatVal}`);
                data = await res.json();
                featureImportanceCache[formatVal] = data;
            }

            if (!data.available) {
                unavailableMsg.textContent = data.message || "Feature importance data is not available for this format.";
                unavailableEl.style.display = "";
                contentEl.style.display = "none";
                return;
            }

            unavailableEl.style.display = "none";
            contentEl.style.display = "";
            renderFeatureImportanceTable("fi-classifier-body", data.classifier.features);
            renderFeatureImportanceTable("fi-regressor-body", data.regressor.features);

            const clfSub = document.getElementById("fi-classifier-subtitle");
            if (clfSub) {
                clfSub.textContent = `Top ${data.classifier.features.length} of ${data.classifier.total_features} features `
                    + `(${data.classifier.folds_evaluated} CV folds, ${data.classifier.low_importance_count} below 1.0% gain).`;
            }
            const regSub = document.getElementById("fi-regressor-subtitle");
            if (regSub) {
                regSub.textContent = `Top ${data.regressor.features.length} of ${data.regressor.total_features} features `
                    + `(${data.regressor.folds_evaluated} CV folds, ${data.regressor.low_importance_count} below 1.0% gain).`;
            }
        } catch (err) {
            console.error("Failed to load feature importance", err);
            unavailableMsg.textContent = "Could not load feature importance data.";
            unavailableEl.style.display = "";
            contentEl.style.display = "none";
        }
    }

    function renderFeatureImportanceTable(bodyId, features) {
        const body = document.getElementById(bodyId);
        if (!body) return;
        if (!features || !features.length) {
            body.innerHTML = `<tr><td colspan="4" class="bet-empty">No feature importance data.</td></tr>`;
            return;
        }
        body.innerHTML = features.map(f => `
            <tr>
                <td>${f.rank}</td>
                <td><span class="match-cell">${escHtml(f.feature)}</span></td>
                <td class="num">${f.mean_gain_pct.toFixed(2)}%</td>
                <td class="num u-text-muted">${f.std_pct.toFixed(2)}%</td>
            </tr>`).join("");
    }

    // --------------------------------------------------
    // Analytics: First Innings Predictor Performance
    // --------------------------------------------------
    function renderFirstInningsPerformance(firstInnings, formatVal) {
        const unavailableEl = document.getElementById("fip-unavailable");
        const unavailableMsg = document.getElementById("fip-unavailable-msg");
        const contentEl = document.getElementById("fip-content");
        if (!unavailableEl || !contentEl) return;

        if (!firstInnings) {
            unavailableMsg.textContent = formatVal === "hnd"
                ? "First innings predictor performance data is not available for the Hundred format."
                : "First innings predictor performance data is not available.";
            unavailableEl.style.display = "";
            contentEl.style.display = "none";
            return;
        }

        unavailableEl.style.display = "none";
        contentEl.style.display = "";

        const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
        set("fip-kpi-mae", `${firstInnings.overall_mae.toFixed(1)} runs`);
        set("fip-kpi-rmse", `${firstInnings.overall_rmse.toFixed(1)} runs`);
        set("fip-kpi-samples", Number(firstInnings.sample_size).toLocaleString());

        if (fipPhaseMaeChart) fipPhaseMaeChart.destroy();
        const ctx = document.getElementById("fip-phase-mae-chart").getContext("2d");
        const phaseMae = firstInnings.per_phase_mae || {};
        fipPhaseMaeChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Powerplay", "Middle Overs", "Death Overs"],
                datasets: [{
                    label: "MAE (runs)",
                    data: [phaseMae.powerplay, phaseMae.middle, phaseMae.death],
                    backgroundColor: [
                        "rgba(244, 63, 94, 0.75)",
                        "rgba(59, 130, 246, 0.75)",
                        "rgba(16, 185, 129, 0.75)"
                    ],
                    borderColor: ["#f43f5e", "#3b82f6", "#10b981"],
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: "#94a3b8" } },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8" },
                        title: { display: true, text: "MAE (runs)", color: "#64748b" }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // --------------------------------------------------
    // Analytics: Monte Carlo Simulation Performance
    // --------------------------------------------------
    const MCP_FORMAT_LABELS = {
        t20: "T20", od: "One-Day", hnd: "The Hundred",
        t20w: "Women's T20", odw: "Women's ODI", hndw: "Women's Hundred",
    };

    async function loadMcPerformance(formatVal) {
        formatVal = formatVal || currentAnalyticsFormat();
        const unavailableEl = document.getElementById("mcp-unavailable");
        const unavailableMsg = document.getElementById("mcp-unavailable-msg");
        const contentEl = document.getElementById("mcp-content");
        if (!unavailableEl || !contentEl) return;

        try {
            const res = await fetch(`/api/mc-performance?format=${formatVal}`);
            const data = await res.json();
            mcPerformanceLoadedFor.add(formatVal);

            if (!data.available) {
                unavailableMsg.textContent = data.message || "Monte Carlo calibration metrics have not been generated yet.";
                unavailableEl.style.display = "";
                contentEl.style.display = "none";
                return;
            }

            unavailableEl.style.display = "none";
            contentEl.style.display = "";

            const set = (id, text, passed) => {
                const el = document.getElementById(id);
                if (!el) return;
                el.textContent = text;
                el.classList.remove("bet-pos", "bet-neg");
                if (passed === true) el.classList.add("bet-pos");
                if (passed === false) el.classList.add("bet-neg");
            };
            const ic = data.innings_calibration;
            set("mcp-kpi-coverage", `${(ic.coverage_rate * 100).toFixed(1)}%`, ic.coverage_passed);
            set("mcp-kpi-mae", `${ic.mae.toFixed(1)} runs`, ic.mae_passed);
            set("mcp-kpi-bias", `${ic.bias >= 0 ? "+" : ""}${ic.bias.toFixed(1)} runs`, ic.bias_passed);
            // Win-prob tile shows SKILL vs actual winners (gated) on snapshots that have it;
            // older snapshots predating the 2026-08-13 re-spec only carry the agreement
            // diagnostic, so fall back to that rather than rendering "--".
            const ws = data.winprob_skill;
            if (ws) {
                const skill = ws.skill_vs_baserate;
                set("mcp-kpi-agreement", `${skill >= 0 ? "+" : ""}${skill.toFixed(3)}`, ws.passed);
            } else {
                set("mcp-kpi-agreement", `${(data.winprob_agreement.mean_abs_diff * 100).toFixed(1)}%`, data.winprob_agreement.passed);
            }
            set("mcp-kpi-brier", data.over_under_reliability.brier_score.toFixed(4), data.over_under_reliability.passed);
            set("mcp-kpi-stddev", `${data.variance_sanity.avg_std_dev.toFixed(1)} runs`, data.variance_sanity.passed);

            const metaEl = document.getElementById("mcp-run-meta");
            if (metaEl) {
                const n = ic.sample_matches;
                const fmtLabel = MCP_FORMAT_LABELS[formatVal] || formatVal.toUpperCase();
                const gateSummary = (data.gates_total != null)
                    ? ` ${data.gates_passed}/${data.gates_total} GO/NO-GO gates passed.`
                    : "";
                metaEl.textContent = (data.generated_at
                    ? `Backtested against ${n} historical ${fmtLabel} matches. Snapshot generated ${new Date(data.generated_at).toLocaleString()}.`
                    : `Backtested against ${n} historical ${fmtLabel} matches.`) + gateSummary;
                metaEl.classList.toggle("bet-neg", data.gates_total != null && data.gates_passed < data.gates_total);
            }
        } catch (err) {
            console.error("Failed to load MC performance metrics", err);
            unavailableMsg.textContent = "Could not load Monte Carlo calibration metrics.";
            unavailableEl.style.display = "";
            contentEl.style.display = "none";
        }
    }

    // --------------------------------------------------
    // BETTING TAB — paper-bet dashboard + pre-match scanner
    // --------------------------------------------------
    let betEquityChart = null;

    const escHtml = (s) => String(s ?? "").replace(/[&<>"']/g, c =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    const money = (v) => "£" + Number(v || 0).toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const signedMoney = (v) => (Number(v) >= 0 ? "+" : "−") + money(Math.abs(Number(v) || 0));
    const signedPct = (frac) => (Number(frac) >= 0 ? "+" : "−") + (Math.abs(Number(frac) || 0) * 100).toFixed(1) + "%";
    const pnlClass = (v) => (Number(v) >= 0 ? "bet-pos" : "bet-neg");

    async function loadBetting() {
        try {
            const [summaryRes, betsRes] = await Promise.all([
                fetch("/api/betting/summary"),
                fetch("/api/betting/bets"),
            ]);
            const summary = await summaryRes.json();
            const betsPayload = await betsRes.json();
            if (summary.error) { showToast(summary.error, "error"); return; }
            if (betsPayload.error) { showToast(betsPayload.error, "error"); return; }

            renderBettingSummary(summary);
            renderBetLedger(betsPayload.bets || []);
            renderBankrollCurve(betsPayload.bets || [], betsPayload.starting_bankroll || 1000);
        } catch (e) {
            console.error(e);
            showToast("Could not load betting data.", "error");
        }
    }

    function renderBettingSummary(s) {
        const set = (id, text, cls) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.textContent = text;
            el.classList.remove("bet-pos", "bet-neg");
            if (cls) el.classList.add(cls);
        };
        set("bet-kpi-bankroll", money(s.current_bankroll));
        set("bet-kpi-pnl", signedMoney(s.total_pnl), pnlClass(s.total_pnl));
        set("bet-kpi-roi", (s.roi_pct >= 0 ? "+" : "−") + Math.abs(s.roi_pct).toFixed(1) + "%", pnlClass(s.roi_pct));
        set("bet-kpi-winrate", (s.win_rate * 100).toFixed(1) + "%");
        set("bet-kpi-record", `${s.wins}–${s.losses}–${s.pending}`);
        set("bet-kpi-edge", signedPct(s.avg_edge), pnlClass(s.avg_edge));
    }

    function renderBetLedger(bets) {
        const body = document.getElementById("bet-ledger-body");
        const status = document.getElementById("bet-ledger-status");
        if (!body) return;
        if (!bets.length) {
            body.innerHTML = `<tr><td colspan="10" class="bet-empty">No bets logged yet. Run a scan to find value bets.</td></tr>`;
            if (status) status.textContent = "No bets logged yet.";
            return;
        }
        if (status) status.textContent = `${bets.length} bet${bets.length === 1 ? "" : "s"} logged (newest first).`;
        body.innerHTML = bets.map(b => {
            const result = String(b.result || "pending").toLowerCase();
            const pnlCell = (b.pnl === null || b.pnl === undefined)
                ? '<span class="u-text-muted">—</span>'
                : `<span class="${pnlClass(b.pnl)}">${signedMoney(b.pnl)}</span>`;
            return `<tr>
                <td><span class="match-cell">${escHtml(b.match_description)}</span></td>
                <td><span class="bet-fmt">${escHtml(String(b.format || "").toUpperCase())}</span></td>
                <td>${escHtml(b.bet_on)}</td>
                <td class="num">${(b.model_prob * 100).toFixed(1)}%</td>
                <td class="num">${(b.fair_market_prob * 100).toFixed(1)}%</td>
                <td class="num ${pnlClass(b.edge)}">${signedPct(b.edge)}</td>
                <td class="num">${Number(b.decimal_odds).toFixed(2)}</td>
                <td class="num">${money(b.stake)}</td>
                <td><span class="bet-result ${result}">${escHtml(result)}</span></td>
                <td class="num">${pnlCell}</td>
            </tr>`;
        }).join("");
    }

    function renderBankrollCurve(bets, startingBankroll) {
        // Equity curve = running balance after each SETTLED bet, oldest -> newest.
        const settled = bets
            .filter(b => b.pnl !== null && b.pnl !== undefined && ["win", "loss", "void"].includes(String(b.result).toLowerCase()))
            .slice()
            .sort((a, b) => String(a.timestamp).localeCompare(String(b.timestamp)));

        const labels = ["Start", ...settled.map((_, i) => String(i + 1))];
        const data = [startingBankroll, ...settled.map(b => b.bankroll_after)];

        if (betEquityChart) betEquityChart.destroy();
        const ctx = document.getElementById("bet-equity-chart").getContext("2d");
        const grad = ctx.createLinearGradient(0, 0, 0, 300);
        grad.addColorStop(0, "rgba(0, 242, 254, 0.25)");
        grad.addColorStop(1, "rgba(0, 242, 254, 0.01)");

        betEquityChart = new Chart(ctx, {
            type: "line",
            data: {
                labels,
                datasets: [{
                    label: "Bankroll (£)",
                    data,
                    borderColor: "#00f2fe",
                    backgroundColor: grad,
                    borderWidth: 2.5,
                    tension: 0.25,
                    pointRadius: settled.length > 40 ? 0 : 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: "#00f2fe",
                    fill: true,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { grid: { display: false }, ticks: { color: "#94a3b8", maxTicksLimit: 12 },
                         title: { display: true, text: "Settled bet #", color: "#64748b" } },
                    y: { grid: { color: "rgba(255, 255, 255, 0.04)" },
                         ticks: { color: "#94a3b8", callback: (v) => "£" + v } },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: (c) => "Bankroll: " + money(c.parsed.y) } },
                }
            }
        });
    }

    // Betting sub-tab switching (Dashboard | Scanner)
    document.querySelectorAll("#betting-subtabs .mc-subtab").forEach(btn =>
        btn.addEventListener("click", () => {
            const view = btn.dataset.betView;
            document.querySelectorAll("#betting-subtabs .mc-subtab").forEach(b =>
                b.classList.toggle("active", b === btn));
            document.getElementById("betting-dashboard").style.display = view === "dashboard" ? "" : "none";
            document.getElementById("betting-scanner").style.display = view === "scanner" ? "" : "none";
        }));

    // Pre-match scanner submit (PAPER mode; hits the live Betfair API server-side)
    on("betting-scan-form", "submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("betting-scan-btn");
        const loading = document.getElementById("betting-scan-loading");
        const results = document.getElementById("betting-scan-results");

        const payload = {
            format_filter: document.getElementById("scan-format").value,
            hours_ahead: parseInt(document.getElementById("scan-hours").value, 10) || 48,
            threshold: (parseFloat(document.getElementById("scan-edge").value) || 3) / 100,
            kelly_fraction: parseFloat(document.getElementById("scan-kelly").value) || 0.25,
            bankroll: parseFloat(document.getElementById("scan-bankroll").value) || 1000,
        };

        btn.disabled = true;
        loading.style.display = "block";
        results.style.display = "none";
        try {
            const res = await fetch("/api/betting/scan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (data.error) { showToast(data.error, "error"); return; }
            renderScanResults(data.results || []);
            results.style.display = "block";
            // Paper bets may have been logged -> refresh the dashboard ledger + KPIs.
            if (data.summary) renderBettingSummary(data.summary);
            loadBetting();
            showToast("Scan complete.", "success");
        } catch (err) {
            console.error(err);
            showToast("Scan failed.", "error");
        } finally {
            loading.style.display = "none";
            btn.disabled = false;
        }
    });

    function renderScanResults(results) {
        const body = document.getElementById("betting-scan-body");
        const status = document.getElementById("betting-scan-status");
        if (!body) return;
        if (!results.length) {
            body.innerHTML = `<tr><td colspan="7" class="bet-empty">No upcoming events matched, or no value found.</td></tr>`;
            if (status) status.textContent = "";
            return;
        }
        const valueCount = results.filter(r => r.is_value_a || r.is_value_b).length;
        if (status) status.textContent = `${results.length} match${results.length === 1 ? "" : "es"} scanned · ${valueCount} value bet${valueCount === 1 ? "" : "s"} found (paper bets logged).`;

        body.innerHTML = results.map(r => {
            // Show whichever side carries the larger edge (mirrors the CLI summary table).
            const preferA = (r.edge_a ?? -1) >= (r.edge_b ?? -1);
            const modelP = preferA ? r.model_prob_a : r.model_prob_b;
            const fairP = preferA ? r.fair_prob_a : r.fair_prob_b;
            const edge = preferA ? r.edge_a : r.edge_b;
            const odds = preferA ? r.back_odds_a : r.back_odds_b;
            const isValue = r.is_value_a || r.is_value_b;
            return `<tr class="${isValue ? "bet-value-row" : ""}">
                <td><span class="match-cell">${escHtml(r.team_a)} v ${escHtml(r.team_b)}</span></td>
                <td><span class="bet-fmt">${escHtml(String(r.format || "").toUpperCase())}</span></td>
                <td class="num">${(modelP * 100).toFixed(1)}%</td>
                <td class="num">${(fairP * 100).toFixed(1)}%</td>
                <td class="num ${pnlClass(edge)}">${signedPct(edge)}</td>
                <td class="num">${Number(odds).toFixed(2)}</td>
                <td>${escHtml(r.recommended_action)}</td>
            </tr>`;
        }).join("");
    }

    // --------------------------------------------------
    // Rating-source toggle (markdowns/win_predictor/rating_source_toggle_plan.md Phase 4).
    // Global, persisted choice between "elo" (deployed default) and "icc" (the ICC-style +
    // franchise-bridge engine); read by every /api/elo/* fetch below via ?source=. Prediction
    // endpoints don't consume this yet (Phase 6/7) -- it only affects the Player & Team Data
    // tab's display for now.
    // --------------------------------------------------
    const RATING_SOURCE_KEY = "ratingSource";
    let ratingSource = localStorage.getItem(RATING_SOURCE_KEY) || "elo";

    const POOL_BADGE_LABELS = {
        intl: "International", bridge_a: "Franchise (bridged)",
        bridge_bc: "Franchise (multi-hop)", uncovered: "Uncovered",
    };

    function poolBadgeHtml(pool) {
        if (!pool) return "";
        const cls = String(pool).replace(/[^a-z_]/g, "");
        return `<span class="pool-badge pool-badge--${cls}">${escHtml(POOL_BADGE_LABELS[pool] || pool)}</span>`;
    }

    // Sets a prediction panel's source-indicator badge element to match `src` ("elo"/"icc") --
    // used by every wired prediction endpoint's result renderer, not just the display tab.
    function setSourceBadge(elId, src) {
        const el = document.getElementById(elId);
        if (!el) return;
        el.textContent = src === "icc" ? "ICC-style" : "Elo";
        el.className = `source-badge source-badge--${src === "icc" ? "icc" : "elo"}`;
    }

    // Same as setSourceBadge, but returns markup for inline injection rather than updating one
    // fixed element by id -- for panels (like the Live Games win-prob line) that rebuild their
    // whole innerHTML on every render instead of mutating a persistent badge node.
    function sourceBadgeHtml(src) {
        const cls = src === "icc" ? "icc" : "elo";
        return `<span class="source-badge source-badge--${cls}">${cls === "icc" ? "ICC-style" : "Elo"}</span>`;
    }

    // Formats not listed in metadataCache.rating_sources[src] have no coverage for that source
    // (e.g. icc for hnd/hndw) -- fails OPEN (treats as supported) before metadata has loaded,
    // since the backend itself is the actual authority and will 400 a genuinely bad combination.
    function formatSupportsSource(fmt, src) {
        const byFmt = metadataCache?.rating_sources;
        if (!byFmt || !byFmt[fmt]) return true;
        return byFmt[fmt].includes(src);
    }

    // Disables the "ICC-style" <option> in a rating-source <select> for formats with no
    // coverage, and silently falls the select back to Elo if it was sitting on the now-disabled
    // option -- called whenever the paired format <select> changes.
    function refreshSourceSelectOptions(sourceSelectEl, fmt) {
        const iccOption = sourceSelectEl.querySelector('option[value="icc"]');
        if (!iccOption) return;
        const supported = formatSupportsSource(fmt, "icc");
        iccOption.disabled = !supported;
        iccOption.title = supported ? "" : `No ICC-style coverage for ${fmt.toUpperCase()} yet`;
        if (!supported && sourceSelectEl.value === "icc") {
            sourceSelectEl.value = "elo";
            setRatingSource("elo");
        }
    }

    function setRatingSource(src) {
        ratingSource = src;
        localStorage.setItem(RATING_SOURCE_KEY, src);
        document.querySelectorAll(".rating-source-select").forEach(sel => {
            if (sel.value !== src) sel.value = src;
        });
    }

    // --------------------------------------------------
    // TAB 7: PLAYER & TEAM DATA (Elo progression explorer — up to 8-way comparison)
    // --------------------------------------------------
    let eloExplorerInit = false;
    let eloHistoryChart = null;
    let eloEntityRoles = {};    // player name -> [available roles]
    let eloCompareItems = [];   // [{type, format, role, name, key, colorIdx}]
    let eloRangeStart = null;   // "YYYY-MM-DD" or null (open-ended)
    let eloRangeEnd = null;     // "YYYY-MM-DD" or null (open-ended)
    let eloLastValid = [];      // last-rendered [{r, item}] rows, kept so re-sorting needs no refetch
    let eloSortCol = null;      // "current" | "peak" | "low" | "matches" | null (insertion order)
    let eloSortDir = "desc";
    const ELO_MAX_COMPARE = 8;

    // Fixed-order categorical palette (colorblind-validated), dark-surface steps —
    // matches this dashboard's dark theme. Assigned by slot index, never cycled/reused
    // while a series is on screen, so removing one item never repaints the others.
    const ELO_PALETTE = [
        "#3987e5", // blue
        "#d95926", // orange
        "#199e70", // aqua
        "#c98500", // yellow
        "#d55181", // magenta
        "#008300", // green
        "#9085e9", // violet
        "#e66767", // red
    ];

    // Turn any CSS colour (hex or the hsl() strings teamVisual emits) into a translucent rgba/hsla.
    function withAlpha(color, a) {
        if (!color) return `rgba(0, 242, 254, ${a})`;
        if (color[0] === "#") {
            let h = color.slice(1);
            if (h.length === 3) h = h.split("").map(c => c + c).join("");
            const n = parseInt(h, 16);
            return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
        }
        const m = color.match(/hsl\(\s*([\d.]+)[,\s]+([\d.]+)%[,\s]+([\d.]+)%/i);
        if (m) return `hsla(${m[1]}, ${m[2]}%, ${m[3]}%, ${a})`;
        return color;
    }

    function eloItemKey(type, fmt, role, name) {
        return `${type}|${fmt}|${type === "player" ? role : ""}|${name}`;
    }

    // Players have no brand colour, so they draw from the fixed palette (assigned at add-time,
    // held for the item's lifetime). Teams never consume a slot — they always use their real
    // franchise/derived colour (teamVisual), same as the Live Games / Live Simulator tabs.
    function eloNextColorIdx() {
        const used = new Set(eloCompareItems.filter(it => it.type === "player").map(it => it.colorIdx));
        for (let i = 0; i < ELO_PALETTE.length; i++) if (!used.has(i)) return i;
        return 0;
    }

    function eloItemColor(item) {
        return item.type === "team" ? teamVisual(item.name).color : ELO_PALETTE[item.colorIdx];
    }

    function eloItemLabel(item) {
        const fmtLabel = item.format.toUpperCase();
        if (item.type === "player") {
            return `${item.name} — ${item.role === "bowler" ? "Bowling" : "Batting"} (${fmtLabel})`;
        }
        return `${item.name} (${fmtLabel})`;
    }

    // Computes {from, to} ISO date strings for a preset ("all" = open-ended both sides).
    function eloPresetRange(preset) {
        if (preset === "all") return { from: "", to: "" };
        const years = { "1y": 1, "3y": 3, "5y": 5, "10y": 10 }[preset] || 1;
        const to = new Date();
        const from = new Date(to);
        from.setFullYear(from.getFullYear() - years);
        const iso = (d) => d.toISOString().slice(0, 10);
        return { from: iso(from), to: iso(to) };
    }

    function setEloRangeActivePreset(preset) {
        document.querySelectorAll("#elo-range-presets .mc-subtab")
            .forEach(b => b.classList.toggle("active", b.dataset.range === preset));
    }

    async function initEloExplorer() {
        if (eloExplorerInit) return;
        eloExplorerInit = true;
        await fetchMetadata();

        const typeSel = document.getElementById("elo-type");
        const fmtSel = document.getElementById("elo-format");
        const sourceSel = document.getElementById("elo-source");
        const input = document.getElementById("elo-entity-input");

        sourceSel.value = ratingSource;
        refreshSourceSelectOptions(sourceSel, fmtSel.value);

        typeSel.addEventListener("change", () => { updateEloModeUI(); loadEloEntities(); });
        fmtSel.addEventListener("change", () => { refreshSourceSelectOptions(sourceSel, fmtSel.value); loadEloEntities(); });
        sourceSel.addEventListener("change", () => { setRatingSource(sourceSel.value); loadEloEntities(); refreshEloComparison(); });
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter") { e.preventDefault(); addEloCompareItem(); }
        });
        document.getElementById("elo-add-btn").addEventListener("click", () => addEloCompareItem());
        document.getElementById("elo-clear-btn").addEventListener("click", () => {
            eloCompareItems = [];
            refreshEloComparison();
        });

        // Time-period controls: quick presets (relative to today) + free-form From/To pickers.
        // Applies to the whole comparison at once — it's a viewing window, not a per-entity setting.
        const fromInput = document.getElementById("elo-range-from");
        const toInput = document.getElementById("elo-range-to");
        document.querySelectorAll("#elo-range-presets .mc-subtab").forEach(btn => {
            btn.addEventListener("click", () => {
                setEloRangeActivePreset(btn.dataset.range);
                const { from, to } = eloPresetRange(btn.dataset.range);
                fromInput.value = from;
                toInput.value = to;
                eloRangeStart = from || null;
                eloRangeEnd = to || null;
                refreshEloComparison();
            });
        });
        [fromInput, toInput].forEach(el => {
            el.addEventListener("change", () => {
                setEloRangeActivePreset(null); // manual edit: no preset stays highlighted
                eloRangeStart = fromInput.value || null;
                eloRangeEnd = toInput.value || null;
                refreshEloComparison();
            });
        });

        // Comparison table: click a stat column to sort by it; click again to flip direction.
        // Purely client-side against the already-fetched rows, so it needs no refetch.
        document.querySelectorAll("#elo-compare-table th[data-sort]").forEach(th => {
            th.addEventListener("click", () => {
                if (eloSortCol === th.dataset.sort) {
                    eloSortDir = eloSortDir === "desc" ? "asc" : "desc";
                } else {
                    eloSortCol = th.dataset.sort;
                    eloSortDir = "desc";
                }
                renderEloCompareTable(eloLastValid);
            });
        });

        updateEloModeUI();
        await loadEloEntities();
    }

    function updateEloModeUI() {
        const type = document.getElementById("elo-type").value;
        document.getElementById("elo-role-wrapper").style.display = type === "player" ? "" : "none";
        document.getElementById("elo-entity-label").textContent = type === "player" ? "Select Player" : "Select Team";
        document.getElementById("elo-entity-input").placeholder =
            type === "player" ? "Type a player name…" : "Type a team name…";
    }

    async function loadEloEntities() {
        const type = document.getElementById("elo-type").value;
        const fmt = document.getElementById("elo-format").value;
        const input = document.getElementById("elo-entity-input");
        const list = document.getElementById("elo-entity-list");

        input.value = "";
        list.innerHTML = "";
        eloEntityRoles = {};

        try {
            const resp = await fetch(`/api/elo/entities?type=${type}&format=${fmt}&source=${ratingSource}`);
            const data = await resp.json();
            if (!resp.ok) { showToast(data.error || "Failed to load list.", "error"); return; }
            const entities = data.entities || [];
            list.innerHTML = entities.map(e => `<option value="${escHtml(e.name)}"></option>`).join("");
            if (type === "player") {
                entities.forEach(e => { eloEntityRoles[e.name] = e.roles || ["batter", "bowler"]; });
            }
            input.disabled = entities.length === 0;
            if (!entities.length) showToast(`No ${type} rating data available for this format/source.`, "error");
        } catch (e) {
            console.error("Error loading Elo entities:", e);
            showToast("Failed to load list.", "error");
        }
    }

    function addEloCompareItem() {
        const type = document.getElementById("elo-type").value;
        const fmt = document.getElementById("elo-format").value;
        const input = document.getElementById("elo-entity-input");
        const name = input.value.trim();
        if (!name) return;

        // For players, default to a role they actually have data for.
        let role = document.getElementById("elo-role").value;
        if (type === "player" && eloEntityRoles[name] && !eloEntityRoles[name].includes(role)) {
            role = eloEntityRoles[name][0];
        }

        const key = eloItemKey(type, fmt, role, name);
        if (eloCompareItems.some(it => it.key === key)) {
            showToast(`${name} is already in the comparison.`, "error");
            return;
        }
        if (eloCompareItems.length >= ELO_MAX_COMPARE) {
            showToast(`You can compare up to ${ELO_MAX_COMPARE} teams/players at once.`, "error");
            return;
        }

        eloCompareItems.push({ type, format: fmt, role, name, key, colorIdx: type === "player" ? eloNextColorIdx() : null });
        input.value = "";
        refreshEloComparison();
    }

    function removeEloCompareItem(key) {
        eloCompareItems = eloCompareItems.filter(it => it.key !== key);
        refreshEloComparison();
    }

    function renderEloChips() {
        const bar = document.getElementById("elo-compare-bar");
        const chips = document.getElementById("elo-chips");
        if (!eloCompareItems.length) { bar.style.display = "none"; return; }
        bar.style.display = "flex";
        chips.innerHTML = eloCompareItems.map(it => `
            <span class="elo-chip" style="--chip-color:${eloItemColor(it)}">
                <i class="elo-chip-dot"></i>
                <span class="elo-chip-label">${escHtml(eloItemLabel(it))}</span>
                <button type="button" class="elo-chip-remove" data-key="${escHtml(it.key)}" aria-label="Remove ${escHtml(it.name)}">&times;</button>
            </span>`).join("");
        chips.querySelectorAll(".elo-chip-remove").forEach(btn =>
            btn.addEventListener("click", () => removeEloCompareItem(btn.dataset.key)));
    }

    async function refreshEloComparison() {
        renderEloChips();

        if (!eloCompareItems.length) {
            document.getElementById("elo-chart-container").style.display = "none";
            document.getElementById("elo-empty").style.display = "";
            return;
        }

        try {
            const resp = await fetch("/api/elo/compare", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    items: eloCompareItems.map(({ type, format, role, name }) =>
                        ({ type, format, role, name, source: ratingSource })),
                    start: eloRangeStart,
                    end: eloRangeEnd,
                }),
            });
            const data = await resp.json();
            const results = data.results || [];

            // Surface per-item lookup failures without derailing the ones that succeeded.
            results.forEach((r, i) => {
                if (r.error) showToast(`${eloCompareItems[i]?.name || "Item"}: ${r.error}`, "error");
            });

            renderEloComparisonChart(results);
        } catch (e) {
            console.error("Error fetching Elo comparison:", e);
            showToast("Failed to load Elo history.", "error");
        }
    }

    function renderEloComparisonChart(results) {
        const valid = results
            .map((r, i) => ({ r, item: eloCompareItems[i] }))
            .filter(({ r }) => r && !r.error && (r.series || []).length);

        if (!valid.length) {
            document.getElementById("elo-chart-container").style.display = "none";
            document.getElementById("elo-empty").style.display = "";
            return;
        }

        document.getElementById("elo-empty").style.display = "none";
        document.getElementById("elo-chart-container").style.display = "block";
        const sourceLabel = ratingSource === "icc" ? "ICC-style rating" : "Elo progression";
        document.getElementById("elo-chart-title").textContent =
            valid.length === 1 ? `${eloItemLabel(valid[0].item)} — ${sourceLabel}` : `Progression Comparison (${sourceLabel})`;

        const rangeLabel = document.getElementById("elo-range-label");
        if (eloRangeStart && eloRangeEnd) rangeLabel.textContent = `${eloRangeStart} to ${eloRangeEnd}`;
        else if (eloRangeStart) rangeLabel.textContent = `From ${eloRangeStart} to present`;
        else if (eloRangeEnd) rangeLabel.textContent = `Up to ${eloRangeEnd}`;
        else rangeLabel.textContent = "All-time";

        // A single series gets the original polished treatment (gradient fill down to the
        // axis); once there's more than one line, filled areas would overlap into a muddy
        // mess, so comparisons render as plain strokes only (same convention as the existing
        // ball-by-ball worm chart, which overlays two teams' lines the same way).
        const isSingle = valid.length === 1;
        const ctx = document.getElementById("elo-history-chart").getContext("2d");
        const datasets = valid.map(({ r, item }) => {
            const color = eloItemColor(item);
            const points = r.series.map(pt => ({ x: Date.parse(pt.date), y: pt.elo }));
            let bg = "transparent";
            if (isSingle) {
                const grad = ctx.createLinearGradient(0, 0, 0, 320);
                grad.addColorStop(0, withAlpha(color, 0.28));
                grad.addColorStop(1, withAlpha(color, 0.01));
                bg = grad;
            }
            return {
                label: eloItemLabel(item),
                data: points,
                borderColor: color,
                backgroundColor: bg,
                borderWidth: 2,
                tension: 0.2,
                pointRadius: points.length > 60 ? 0 : 2,
                pointHoverRadius: 5,
                pointBackgroundColor: color,
                fill: isSingle,
            };
        });

        if (eloHistoryChart) eloHistoryChart.destroy();
        eloHistoryChart = new Chart(ctx, {
            type: "line",
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "nearest", intersect: false, axis: "x" },
                scales: {
                    x: {
                        type: "linear",
                        grid: { display: false },
                        ticks: {
                            color: "#94a3b8", maxTicksLimit: 10,
                            callback: (v) => new Date(v).toISOString().slice(0, 7),
                        },
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.04)" },
                        ticks: { color: "#94a3b8" },
                        title: { display: true, text: ratingSource === "icc" ? "ICC-style rating" : "Elo rating", color: "#64748b" },
                    },
                },
                plugins: {
                    legend: {
                        display: valid.length > 1,
                        position: "top",
                        labels: { color: "#cbd5e1", boxWidth: 12, boxHeight: 12 },
                    },
                    tooltip: {
                        callbacks: {
                            title: (items) => items.length ? new Date(items[0].parsed.x).toISOString().slice(0, 10) : "",
                            label: (c) => `${c.dataset.label}: ${c.parsed.y}`,
                        },
                    },
                }
            }
        });

        renderEloCompareTable(valid);
    }

    function renderEloCompareTable(valid) {
        eloLastValid = valid;

        const sorted = eloSortCol
            ? [...valid].sort((a, b) => {
                const av = a.r.stats?.[eloSortCol] ?? -Infinity;
                const bv = b.r.stats?.[eloSortCol] ?? -Infinity;
                return eloSortDir === "asc" ? av - bv : bv - av;
            })
            : valid;

        const body = document.getElementById("elo-compare-table-body");
        body.innerHTML = sorted.map(({ r, item }) => {
            const s = r.stats || {};
            const color = eloItemColor(item);
            return `<tr>
                <td><i class="elo-chip-dot" style="background:${color}"></i></td>
                <td>${escHtml(eloItemLabel(item))}</td>
                <td class="num">${s.current ?? "--"}</td>
                <td class="num">${s.peak ?? "--"}</td>
                <td class="num">${s.low ?? "--"}</td>
                <td class="num">${s.matches ?? "--"}</td>
            </tr>`;
        }).join("");

        document.querySelectorAll("#elo-compare-table th[data-sort]").forEach(th => {
            const isActive = th.dataset.sort === eloSortCol;
            th.classList.toggle("sorted", isActive);
            th.querySelector(".sort-arrow").textContent = isActive ? (eloSortDir === "asc" ? "▲" : "▼") : "";
        });
    }

    // --------------------------------------------------
    // Player & Team Data — Leaderboard sub-view (top N teams/players by current Elo)
    // --------------------------------------------------
    let lbInit = false;
    let lbLastRows = [];
    let lbSortCol = "current";
    let lbSortDir = "desc";

    // Elo Explorer | Leaderboard sub-tab switching
    document.querySelectorAll("#elo-view-tabs .mc-subtab").forEach(btn =>
        btn.addEventListener("click", () => {
            const view = btn.dataset.eloView;
            document.querySelectorAll("#elo-view-tabs .mc-subtab").forEach(b =>
                b.classList.toggle("active", b === btn));
            document.getElementById("elo-view-explorer").style.display = view === "explorer" ? "" : "none";
            document.getElementById("elo-view-leaderboard").style.display = view === "leaderboard" ? "" : "none";
            if (view === "leaderboard") initEloLeaderboard();
        }));

    function updateLbModeUI() {
        const type = document.getElementById("lb-type").value;
        document.getElementById("lb-role-wrapper").style.display = type === "player" ? "" : "none";
    }

    async function initEloLeaderboard() {
        if (lbInit) return;
        lbInit = true;
        await fetchMetadata();

        const fmtSel = document.getElementById("lb-format");
        const sourceSel = document.getElementById("lb-source");
        sourceSel.value = ratingSource;
        refreshSourceSelectOptions(sourceSel, fmtSel.value);

        document.getElementById("lb-type").addEventListener("change", () => { updateLbModeUI(); loadEloLeaderboard(); });
        fmtSel.addEventListener("change", () => { refreshSourceSelectOptions(sourceSel, fmtSel.value); loadEloLeaderboard(); });
        document.getElementById("lb-role").addEventListener("change", () => loadEloLeaderboard());
        document.getElementById("lb-limit").addEventListener("change", () => loadEloLeaderboard());
        sourceSel.addEventListener("change", () => { setRatingSource(sourceSel.value); loadEloLeaderboard(); });

        // Ranked-table columns: click to sort, click again to flip direction (client-side only).
        document.querySelectorAll("#lb-table th[data-sort]").forEach(th => {
            th.addEventListener("click", () => {
                if (lbSortCol === th.dataset.sort) {
                    lbSortDir = lbSortDir === "desc" ? "asc" : "desc";
                } else {
                    lbSortCol = th.dataset.sort;
                    lbSortDir = "desc";
                }
                renderLbTable(lbLastRows);
            });
        });

        updateLbModeUI();
        await loadEloLeaderboard();
    }

    async function loadEloLeaderboard() {
        const type = document.getElementById("lb-type").value;
        const fmt = document.getElementById("lb-format").value;
        const role = document.getElementById("lb-role").value;
        const limit = document.getElementById("lb-limit").value;

        const sourceTag = ratingSource === "icc" ? " — ICC-style" : "";
        document.getElementById("lb-table-title").textContent = (type === "player"
            ? `Player ${role === "bowler" ? "Bowling" : "Batting"} Elo Leaderboard (${fmt.toUpperCase()})`
            : `Team Elo Leaderboard (${fmt.toUpperCase()})`) + sourceTag;

        document.querySelectorAll("#lb-pool-th").forEach(th => { th.style.display = ratingSource === "icc" ? "" : "none"; });

        try {
            const resp = await fetch(`/api/elo/leaderboard?type=${type}&format=${fmt}&role=${role}&limit=${limit}&source=${ratingSource}`);
            const data = await resp.json();
            if (!resp.ok) {
                lbLastRows = [];
                showToast(data.error || "Failed to load leaderboard.", "error");
                renderLbTable(lbLastRows);
                return;
            }
            lbLastRows = data.rows || [];
            document.getElementById("lb-table-subtitle").textContent = lbLastRows.length
                ? `Showing top ${lbLastRows.length} of ${data.total} ranked ${type === "player" ? "players" : "teams"}.`
                : "";
            renderLbTable(lbLastRows);
        } catch (e) {
            console.error("Error loading Elo leaderboard:", e);
            showToast("Failed to load leaderboard.", "error");
        }
    }

    function renderLbTable(rows) {
        const empty = document.getElementById("lb-empty");
        const container = document.getElementById("lb-table-container");
        const type = document.getElementById("lb-type").value;

        if (!rows.length) {
            document.getElementById("lb-empty-msg").textContent =
                `No ranked ${type === "player" ? "players" : "teams"} found for this format yet.`;
            empty.style.display = "";
            container.style.display = "none";
            return;
        }
        empty.style.display = "none";
        container.style.display = "";

        const sorted = [...rows].sort((a, b) => {
            const av = a[lbSortCol] ?? -Infinity;
            const bv = b[lbSortCol] ?? -Infinity;
            return lbSortDir === "asc" ? av - bv : bv - av;
        });

        const showPool = ratingSource === "icc";
        const body = document.getElementById("lb-table-body");
        body.innerHTML = sorted.map((r, i) => `<tr>
            <td>${i + 1}</td>
            <td>${escHtml(r.name)}</td>
            <td class="lb-pool-col" style="${showPool ? "" : "display:none;"}">${poolBadgeHtml(r.rating_pool)}</td>
            <td class="num">${r.current}</td>
            <td class="num">${r.peak}</td>
            <td class="num">${r.low}</td>
            <td class="num">${r.matches}</td>
            <td>${escHtml(r.last_date || "--")}</td>
        </tr>`).join("");

        document.querySelectorAll("#lb-table th[data-sort]").forEach(th => {
            const isActive = th.dataset.sort === lbSortCol;
            th.classList.toggle("sorted", isActive);
            th.querySelector(".sort-arrow").textContent = isActive ? (lbSortDir === "asc" ? "▲" : "▼") : "";
        });
    }
});
