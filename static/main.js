function $(sel) { return document.querySelector(sel); }

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;",
        '"': "&quot;", "'": "&#39;"
    }[c]));
}

const demoForm = $("#demo-form");

if (demoForm) {
    const ranges = demoForm.querySelectorAll('input[type="range"]');
    const predictBtn = $("#predict-btn");
    const randomBtn = $("#random-btn");
    const resultBox = $("#result");
    const nameInput = $("#client-name");

    function syncOutputs() {
        ranges.forEach((r) => {
            const out = demoForm.querySelector('output[data-for="' + r.id + '"]');
            if (out) out.textContent = r.value;
        });
    }
    ranges.forEach((r) => r.addEventListener("input", syncOutputs));
    syncOutputs();

    function collectFeatures() {
        const values = [];
        ranges.forEach((r) => values.push(parseFloat(r.value)));
        return values;
    }

    const labels = window.DEMO_FIELDS.map((f) => f.name);

    function renderResult(d) {
        const level = d.advice.level;
        const levelText = { high: "зона риска", mid: "средний риск", low: "всё спокойно" }[level];
        let fact = "";
        if (typeof d.real_churned === "boolean") {
            const guessed = (d.risk >= 0.5) === d.real_churned;
            fact = '<div class="fact-line ' + (guessed ? "ok" : "fail") + '">' +
                (guessed
                    ? "На самом деле клиент " + (d.real_churned ? "ушёл" : "остался") + " — сеть угадала!"
                    : "На самом деле клиент " + (d.real_churned ? "ушёл" : "остался") + " — сеть ошиблась (100% угадываний не бывает)") +
                "</div>";
        }
        let mini = '<div class="mini-grid">';
        labels.forEach((label, i) => {
            mini += "<span>" + escapeHtml(label) + ": <b>" + escapeHtml(d.features[i]) + "</b></span>";
        });
        mini += "</div>";
        const savedNote = d.saved
            ? '<div class="saved-note">Проверка сохранена в базу (таблица predictions) — она есть на странице «История»</div>'
            : "";
        let why = "";
        if (d.explain && d.explain.length > 0) {
            const whyTitle = level === "low"
                ? "Что выделяется на фоне обычных клиентов:"
                : "Сеть насторожилась из-за:";
            why = '<div class="why-block"><h4>' + whyTitle + "</h4>";
            d.explain.forEach((e) => {
                why += '<div class="why-row"><b>' + escapeHtml(e.name) + "</b><span>у клиента — " +
                    escapeHtml(e.value) + ", у среднего беглеца — " + escapeHtml(e.churner) + "</span></div>";
            });
            why += "</div>";
        }
        resultBox.innerHTML =
            '<div class="result-head">' +
                '<span class="result-name">' + escapeHtml(d.name) + "</span>" +
                '<span class="verdict-badge ' + level + '">' + levelText + "</span>" +
            "</div>" +
            '<div class="risk-big ' + level + '">' + d.risk_pct + "%</div>" +
            '<div class="gauge"><div class="gauge-fill ' + level + '" style="width: ' + d.risk_pct + '%"></div></div>' +
            '<p class="advice">' + escapeHtml(d.advice.text) + "</p>" +
            why + mini + fact + savedNote;
    }

    function showError(text) {
        resultBox.textContent = text;
    }

    async function predict() {
        predictBtn.disabled = true;
        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    features: collectFeatures(),
                    name: nameInput ? nameInput.value : "",
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Что-то пошло не так");
            renderResult({ features: collectFeatures(), ...data });
        } catch (e) {
            showError("Ошибка: " + e.message);
        }
        predictBtn.disabled = false;
    }

    async function randomClient() {
        randomBtn.disabled = true;
        try {
            const res = await fetch("/api/random-client");
            const d = await res.json();
            ranges.forEach((r, i) => { r.value = d.features[i]; });
            syncOutputs();
            renderResult(d);
        } catch (e) {
            showError("Ошибка: " + e.message);
        }
        randomBtn.disabled = false;
    }

    predictBtn.addEventListener("click", predict);
    randomBtn.addEventListener("click", randomClient);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

const sigSlider = $("#sig-x");

if (sigSlider) {
    const W = 400, H = 240, X0 = 20, X1 = 380, Y0 = 10, Y1 = 230;
    const toPx = (x) => X0 + ((x + 6) / 12) * (X1 - X0);
    const toPy = (y) => Y1 - y * (Y1 - Y0);

    let points = [];
    for (let x = -6; x <= 6.01; x += 0.1) {
        points.push(toPx(x).toFixed(1) + "," + toPy(sigmoid(x)).toFixed(1));
    }
    $("#sig-curve").setAttribute("points", points.join(" "));

    function updateSig() {
        const x = parseFloat(sigSlider.value);
        const y = sigmoid(x);
        $("#sig-dot").setAttribute("cx", toPx(x));
        $("#sig-dot").setAttribute("cy", toPy(y));
        $("#sig-x-value").textContent = x.toFixed(1);
        $("#sig-y-value").textContent = y.toFixed(3);
    }
    sigSlider.addEventListener("input", updateSig);
    updateSig();
}

const netSvg = $("#net-svg");

if (netSvg) {
    const NS = "http://www.w3.org/2000/svg";
    const layers = [
        { n: 8, x: 80, color: "#3B82F6", label: "8 признаков" },
        { n: 16, x: 360, color: "#F97316", label: "16 нейронов" },
        { n: 1, x: 640, color: "#10B981", label: "риск ухода" },
    ];

    const coords = layers.map((layer) => {
        if (layer.n === 1) return [250];
        const step = 440 / (layer.n - 1);
        const pts = [];
        for (let i = 0; i < layer.n; i++) pts.push(30 + i * step);
        return pts;
    });

    for (let li = 0; li < 2; li++) {
        for (const y1 of coords[li]) {
            for (const y2 of coords[li + 1]) {
                const line = document.createElementNS(NS, "line");
                line.setAttribute("x1", layers[li].x);
                line.setAttribute("y1", y1);
                line.setAttribute("x2", layers[li + 1].x);
                line.setAttribute("y2", y2);
                line.setAttribute("stroke", "#CBD5E1");
                line.setAttribute("stroke-width", "0.7");
                line.setAttribute("opacity", (0.12 + Math.random() * 0.25).toFixed(2));
                netSvg.appendChild(line);
            }
        }
    }

    layers.forEach((layer, li) => {
        coords[li].forEach((y, i) => {
            const c = document.createElementNS(NS, "circle");
            c.setAttribute("cx", layer.x);
            c.setAttribute("cy", y);
            c.setAttribute("r", li === 2 ? 16 : 11);
            c.setAttribute("fill", layer.color);
            netSvg.appendChild(c);
        });
        const text = document.createElementNS(NS, "text");
        text.setAttribute("x", layer.x);
        text.setAttribute("y", 500);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", "14");
        text.setAttribute("font-weight", "bold");
        text.setAttribute("fill", "#64748B");
        text.textContent = layer.label;
        netSvg.appendChild(text);
    });
}

const errorSvg = $("#error-svg");

if (errorSvg && window.TRAIN_HISTORY && window.TRAIN_HISTORY.length > 0) {
    drawErrorChart(window.TRAIN_HISTORY);
}

function drawErrorChart(hist) {
    const NS = "http://www.w3.org/2000/svg";
    const W = 720, H = 280, M = { left: 50, right: 16, top: 18, bottom: 34 };
    errorSvg.innerHTML = "";

    const maxVal = Math.max(...hist) * 1.1;
    const toX = (i) => M.left + (i / (hist.length - 1)) * (W - M.left - M.right);
    const toY = (v) => H - M.bottom - (v / maxVal) * (H - M.top - M.bottom);

    for (let g = 0; g <= 4; g++) {
        const v = (maxVal / 4) * g;
        const line = document.createElementNS(NS, "line");
        line.setAttribute("x1", M.left);
        line.setAttribute("x2", W - M.right);
        line.setAttribute("y1", toY(v));
        line.setAttribute("y2", toY(v));
        line.setAttribute("stroke", "#E4E9F3");
        line.setAttribute("stroke-width", "1");
        errorSvg.appendChild(line);
        const label = document.createElementNS(NS, "text");
        label.setAttribute("x", M.left - 8);
        label.setAttribute("y", toY(v) + 4);
        label.setAttribute("text-anchor", "end");
        label.setAttribute("font-size", "11");
        label.setAttribute("fill", "#64748B");
        label.textContent = v.toFixed(2);
        errorSvg.appendChild(label);
    }

    let points = [];
    hist.forEach((v, i) => points.push(toX(i).toFixed(1) + "," + toY(v).toFixed(1)));
    const curve = document.createElementNS(NS, "polyline");
    curve.setAttribute("points", points.join(" "));
    curve.setAttribute("fill", "none");
    curve.setAttribute("stroke", "#EF4444");
    curve.setAttribute("stroke-width", "2.5");
    errorSvg.appendChild(curve);

    const xlabel = document.createElementNS(NS, "text");
    xlabel.setAttribute("x", (M.left + W - M.right) / 2);
    xlabel.setAttribute("y", H - 8);
    xlabel.setAttribute("text-anchor", "middle");
    xlabel.setAttribute("font-size", "12");
    xlabel.setAttribute("fill", "#64748B");
    xlabel.textContent = "эпохи обучения (0 → " + hist.length + ")";
    errorSvg.appendChild(xlabel);
}

const retrainBtn = $("#retrain-btn");

if (retrainBtn) {
    retrainBtn.addEventListener("click", async () => {
        const status = $("#train-status");
        retrainBtn.disabled = true;
        status.textContent = "Сеть учится... это займёт секунд 20, подожди.";
        try {
            const res = await fetch("/api/train", { method: "POST" });
            const info = await res.json();
            $("#m-accuracy").textContent = info.accuracy + "%";
            $("#m-found").textContent = info.found + "/" + info.total_churners;
            $("#m-false").textContent = info.false_alarms;
            $("#m-seconds").textContent = info.seconds + " сек";
            $("#m-trained").textContent = info.trained_at;
            $("#m-seed").textContent = info.seed;
            $("#m-error-note").textContent =
                "Ошибка — «среднеквадратичная»: разница между ответом сети и правдой, " +
                "возведённая в квадрат. Было " + info.error_start + ", стало " + info.error_end + ".";
            drawErrorChart(info.history);
            status.textContent = "Готово! Сеть переобучена, риски клиентов тоже пересчитаны.";
            $("#retrain-note").textContent =
                "Новый запуск: точность " + info.accuracy + "%, беглецов найдено " +
                info.found + " из " + info.total_churners +
                " (каждый раз чуть по-другому, потому что стартовые веса случайные).";
        } catch (e) {
            status.textContent = "Не получилось: " + e.message;
        }
        retrainBtn.disabled = false;
    });
}

const themeBtn = $("#theme-btn");

function applyMetaTheme() {
    const meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) return;
    meta.setAttribute("content",
        document.documentElement.getAttribute("data-theme") === "dark" ? "#0B1220" : "#F2F5FA");
}

if (themeBtn) {
    themeBtn.addEventListener("click", () => {
        const root = document.documentElement;
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try {
            localStorage.setItem("theme", next);
        } catch (e) {}
        applyMetaTheme();
    });
}

applyMetaTheme();

const thrSlider = $("#thr-slider");

if (thrSlider) {
    const thrValue = $("#thr-value");
    const cells = {
        tp: $("#mx-tp"),
        fp: $("#mx-fp"),
        fn: $("#mx-fn"),
        tn: $("#mx-tn"),
    };
    const summary = $("#mx-summary");

    async function updateMatrix() {
        const thr = thrSlider.value;
        thrValue.textContent = Math.round(thr * 100) + "%";
        try {
            const res = await fetch("/api/matrix?thr=" + thr);
            const m = await res.json();
            cells.tp.textContent = m.tp;
            cells.fp.textContent = m.fp;
            cells.fn.textContent = m.fn;
            cells.tn.textContent = m.tn;
            summary.textContent = "Поймано беглецов: " + m.found + " из " + m.total_churners +
                " (" + m.recall + "%), ложных тревог: " + m.false_alarms +
                ", точность: " + m.accuracy + "%";
        } catch (e) {
            summary.textContent = "Не получилось посчитать: " + e.message;
        }
    }
    thrSlider.addEventListener("input", updateMatrix);
    updateMatrix();
}

const gameBoard = $("#game-board");

if (gameBoard) {
    const nameEl = $("#game-name");
    const statsEl = $("#game-stats");
    const chooseEl = $("#game-choose");
    const resultEl = $("#game-result");
    const youScore = $("#score-you");
    const netScore = $("#score-net");

    function loadScore() {
        try {
            const s = JSON.parse(localStorage.getItem("gameScore"));
            if (s && s.you && s.net) return s;
        } catch (e) {}
        return { you: { right: 0, total: 0 }, net: { right: 0, total: 0 } };
    }

    let score = loadScore();
    let current = null;

    function renderScore() {
        youScore.textContent = score.you.right + " из " + score.you.total;
        netScore.textContent = score.net.right + " из " + score.net.total;
    }

    function saveScore() {
        try {
            localStorage.setItem("gameScore", JSON.stringify(score));
        } catch (e) {}
    }

    function statRows(features) {
        let html = "";
        window.GAME_FIELDS.forEach((name, i) => {
            html += "<div><span>" + escapeHtml(name) + "</span><b>" + escapeHtml(features[i]) + "</b></div>";
        });
        return html;
    }

    async function newRound() {
        chooseEl.hidden = false;
        resultEl.hidden = true;
        resultEl.innerHTML = "";
        try {
            const res = await fetch("/api/game/new", { method: "POST" });
            const d = await res.json();
            if (!res.ok) throw new Error(d.error || "Что-то пошло не так");
            current = d;
            nameEl.textContent = d.name;
            statsEl.innerHTML = statRows(d.features);
        } catch (e) {
            nameEl.textContent = "Ошибка: " + e.message;
        }
    }

    async function answer(guess) {
        chooseEl.hidden = true;
        try {
            const res = await fetch("/api/game/answer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: current.id, guess: guess }),
            });
            const d = await res.json();
            if (!res.ok) throw new Error(d.error || "Что-то пошло не так");
            score.you.total += 1;
            score.net.total += 1;
            resultEl.hidden = false;
            if (d.player_right) score.you.right += 1;
            if (d.net_right) score.net.right += 1;
            saveScore();
            renderScore();
            const real = d.real_churned ? "ушёл" : "остался";
            const youLine = '<div class="fact-line ' + (d.player_right ? "ok" : "fail") + '">Ты сказал: «' +
                (guess ? "уйдёт" : "останется") + "». На самом деле клиент " + real +
                (d.player_right ? " — угадал!" : " — не угадал") + "</div>";
            const netLine = '<div class="fact-line ' + (d.net_right ? "ok" : "fail") + '">Нейросеть дала риск ' +
                d.risk_pct + "% — " + (d.net_right ? "тоже угадала" : "ошиблась (стопроцентных сетей не бывает)") + "</div>";
            const who = d.player_right && d.net_right
                ? "Ничья — правы оба!"
                : (d.player_right ? "Раунд твой!" : (d.net_right ? "Раунд забрала нейросеть!" : "Забавно — оба промахнулись!"));
            resultEl.innerHTML =
                '<h3 class="game-verdict">' + who + "</h3>" +
                '<p class="advice">Совет сети менеджеру: ' + escapeHtml(d.advice.text.toLowerCase()) + "</p>" +
                youLine + netLine +
                '<div class="game-next"><button class="btn btn-primary" id="game-next-btn" type="button">Следующий клиент</button></div>';
            $("#game-next-btn").addEventListener("click", newRound);
        } catch (e) {
            chooseEl.hidden = false;
            nameEl.textContent = "Ошибка: " + e.message;
        }
    }

    $("#btn-stay").addEventListener("click", () => answer(false));
    $("#btn-leave").addEventListener("click", () => answer(true));
    renderScore();
    newRound();
}
