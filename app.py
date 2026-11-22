import csv
import io
import math
import os

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, abort
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from dataset import FEATURES, FEATURE_HINTS, Normalizer, generate_clients
from models import db, Client, Prediction, TrainingRun, TrainedModel, DB_PATH
from neuronet import NeuralNetwork
from training import run_training

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + DB_PATH.replace("\\", "/")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


@app.before_request
def check_origin():
    
    
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
            abort(403)
    if net is None:
        prepare_app()



def load_model():
    row = TrainedModel.query.order_by(TrainedModel.id.desc()).first()
    if row is None:
        raise RuntimeError("Нет обученной модели в базе данных")
    loaded = NeuralNetwork.from_dict({"sizes": row.sizes, "layers": row.layers})
    norm = Normalizer()
    norm.mins = row.mins
    norm.maxs = row.maxs
    return loaded, norm


def setup_database():
    db.create_all()
    if Client.query.count() == 0:
        print("База пустая — заселяю клиентами...")
        for c in generate_clients():
            db.session.add(Client(**c))
        db.session.commit()
        print("Заселил {} клиентов".format(Client.query.count()))
    if TrainedModel.query.count() == 0:
        print("Обученной сети в базе нет — обучаю (секунд 20)...")
        run_training(verbose=False)


def last_run_info():
    run = TrainingRun.query.order_by(TrainingRun.id.desc()).first()
    if run is None:
        return None
    return {
        "id": run.id,
        "epochs": run.epochs,
        "lr": run.lr,
        "seed": run.seed,
        "seconds": run.seconds,
        "weights": run.weights,
        "hidden": run.hidden,
        "n_clients": run.n_clients,
        "n_test": run.n_test,
        "accuracy": run.accuracy,
        "found": run.found,
        "total_churners": run.total_churners,
        "recall": run.recall,
        "false_alarms": run.false_alarms,
        "error_start": run.error_start,
        "error_end": run.error_end,
        "history": run.history,
        "trained_at": run.trained_at,
    }


def verdict_for(risk):
    if risk >= 0.7:
        return {"text": "Срочно звонить и предлагать скидку", "level": "high"}
    if risk >= 0.4:
        return {"text": "Отправить письмо с промокодом", "level": "mid"}
    return {"text": "Клиент доволен, трогать не надо", "level": "low"}


feature_stats = None


def get_feature_stats():
    global feature_stats
    if feature_stats is not None:
        return feature_stats
    churners = Client.query.filter_by(churned=True).all()
    stayers = Client.query.filter_by(churned=False).all()
    stats = []
    for i in range(len(FEATURES)):
        ch = [c.features()[i] for c in churners]
        st = [c.features()[i] for c in stayers]
        ch_mean = sum(ch) / len(ch) if ch else 0.0
        st_mean = sum(st) / len(st) if st else 0.0
        st_var = sum((v - st_mean) ** 2 for v in st) / len(st) if st else 0.0
        stats.append({
            "churner_mean": ch_mean,
            "stayer_mean": st_mean,
            "stayer_std": st_var ** 0.5,
        })
    feature_stats = stats
    return feature_stats


def fmt_feat(i, v):
    name = FEATURES[i]
    if name == "средний_чек":
        return "{:,}".format(int(round(v))).replace(",", " ") + " руб"
    if name.startswith("доля_"):
        return "{:.0f}%".format(v * 100)
    if float(v).is_integer():
        return "{:.0f}".format(v)
    return "{:.1f}".format(v)


def explain_client(raw):
    reasons = []
    for i, f in enumerate(FEATURES):
        s = get_feature_stats()[i]
        span = s["churner_mean"] - s["stayer_mean"]
        std = s["stayer_std"] or 1e-9
        if abs(span) < 0.25 * std:
            continue
        bad = (raw[i] - s["stayer_mean"]) if span > 0 else (s["stayer_mean"] - raw[i])
        z = bad / std
        if z >= 0.7:
            reasons.append({
                "name": FEATURE_HINTS[f]["text"],
                "value": fmt_feat(i, raw[i]),
                "churner": fmt_feat(i, s["churner_mean"]),
                "z": round(z, 2),
            })
    reasons.sort(key=lambda r: r["z"], reverse=True)
    return reasons[:3]


def feature_importance():
    rows = Client.query.all()
    if not rows:
        return []
    med = []
    for i in range(len(FEATURES)):
        col = sorted(c.features()[i] for c in rows)
        med.append(col[len(col) // 2])
    out = []
    for i, f in enumerate(FEATURES):
        hint = FEATURE_HINTS[f]
        low, high = list(med), list(med)
        low[i], high[i] = hint["min"], hint["max"]
        delta = net.predict(norm.transform(high)) - net.predict(norm.transform(low))
        out.append({"name": hint["text"], "delta": round(delta * 100), "abs": abs(delta)})
    out.sort(key=lambda r: r["abs"], reverse=True)
    return out


def compute_all_risks():
    rows = []
    for c in Client.query.order_by(Client.id).all():
        risk = net.predict(norm.transform(c.features()))
        rows.append({
            "id": c.id,
            "name": c.name,
            "risk": risk,
            "risk_pct": round(risk * 100),
            "churned": c.churned,
        })
    rows.sort(key=lambda r: r["risk"], reverse=True)
    return rows


net = None
norm = None


def prepare_app():
    global net, norm
    with app.app_context():
        setup_database()
        net, norm = load_model()


@app.route("/")
def index():
    info = last_run_info()
    at_risk = sum(1 for c in compute_all_risks() if c["risk"] >= 0.5)
    checks = Prediction.query.count()
    return render_template("index.html", info=info, total=Client.query.count(),
                           at_risk=at_risk, checks=checks)


@app.route("/demo")
def demo_page():
    fields = []
    for f in FEATURES:
        h = FEATURE_HINTS[f]
        fields.append({
            "name": h["text"],
            "min": h["min"],
            "max": h["max"],
            "step": h["step"],
            "default": round(h["min"] + (h["max"] - h["min"]) * 0.3, 2),
        })
    return render_template("demo.html", fields=fields)


@app.route("/how")
def how_page():
    return render_template("how.html", importance=feature_importance())


@app.route("/clients")
def clients_page():
    q = request.args.get("q", "").strip()
    only_risk = request.args.get("risk") == "1"
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    per_page = 50

    rows = compute_all_risks()
    if q:
        rows = [r for r in rows if q.lower() in r["name"].lower()]
    if only_risk:
        rows = [r for r in rows if r["risk"] >= 0.5]

    total_pages = max(1, math.ceil(len(rows) / per_page))
    page = min(page, total_pages)
    chunk = rows[(page - 1) * per_page: page * per_page]
    for r in chunk:
        r["verdict"] = verdict_for(r["risk"])
    return render_template("clients.html", rows=chunk, page=page,
                           total_pages=total_pages, shown=len(rows), q=q,
                           only_risk=only_risk, total=Client.query.count(),
                           page_nums=page_numbers(page, total_pages))


def page_numbers(current, total):
    if total <= 9:
        return list(range(1, total + 1))
    window = set(range(max(1, current - 2), min(total, current + 2) + 1))
    nums = sorted(window | {1, total})
    out, prev = [], 0
    for n in nums:
        if n - prev > 1:
            out.append(None)
        out.append(n)
        prev = n
    return out


@app.route("/clients/export")
def clients_export():
    q = request.args.get("q", "").strip()
    only_risk = request.args.get("risk") == "1"

    rows = []
    for c in Client.query.order_by(Client.id).all():
        risk = net.predict(norm.transform(c.features()))
        rows.append({
            "id": c.id,
            "name": c.name,
            "risk": risk,
            "risk_pct": round(risk * 100),
            "churned": c.churned,
            "features": c.features(),
            "verdict": verdict_for(risk)["text"],
        })
    rows.sort(key=lambda r: r["risk"], reverse=True)
    if q:
        rows = [r for r in rows if q.lower() in r["name"].lower()]
    if only_risk:
        rows = [r for r in rows if r["risk"] >= 0.5]

    wb = Workbook()
    ws = wb.active
    ws.title = "Клиенты"
    headers = (["№", "Клиент"]
               + [f.replace("_", " ") for f in FEATURES]
               + ["Риск, %", "Что делать менеджеру", "На самом деле"])
    ws.append(headers)

    thin = Side(style="thin", color="D8DFEC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="0F1B33")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = head_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws.row_dimensions[1].height = 30

    risk_col = 2 + len(FEATURES) + 1
    real_text = {1: "ушёл", 0: "остался"}
    risk_color = {"high": "DC2626", "mid": "C2410C", "low": "047857"}
    for i, r in enumerate(rows, 1):
        safe_name = r["name"]
        if safe_name.lstrip().startswith(("=", "+", "-", "@")):
            safe_name = "'" + safe_name
        ws.append([i, safe_name, *r["features"], r["risk_pct"],
                   r["verdict"], real_text.get(r["churned"], "неизвестно")])
        level = "high" if r["risk"] >= 0.7 else ("mid" if r["risk"] >= 0.4 else "low")
        cell = ws.cell(row=i + 1, column=risk_col)
        cell.font = Font(bold=True, color=risk_color[level])
        cell.alignment = Alignment(horizontal="center")

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.border = border
    widths = [6, 22] + [14] * len(FEATURES) + [10, 36, 14]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="клиенты.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/clients/import-sample")
def clients_import_sample():
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow(["имя", "месяцев_в_магазине", "покупок_в_месяц", "средний_чек",
                     "дней_с_последней_покупки", "жалоб_за_год", "использованных_скидок",
                     "доля_брошенных_корзин", "доля_открытых_писем", "факт"])
    writer.writerow(["Мария К.", 24, 4.5, 7800, 12, 0, 2, 0.1, 0.8, "остался"])
    writer.writerow(["Олег Т.", 3, 0.5, 5200, 150, 4, 0, 0.85, 0.05, "ушёл"])
    writer.writerow(["Новый клиент", 10, 2, 6000, 60, 1, 1, 0.4, 0.5, ""])
    data = buf.getvalue().encode("utf-8-sig")
    return send_file(io.BytesIO(data), as_attachment=True,
                     download_name="образец_клиенты.csv", mimetype="text/csv")


@app.route("/clients/import", methods=["POST"])
def clients_import():
    file = request.files.get("csvfile")
    if file is None or file.filename == "":
        flash("Файл не выбран — выбери CSV-файл и попробуй ещё раз.")
        return redirect("/clients")
    raw = file.read(2_000_001)
    if len(raw) > 2_000_000:
        flash("Файл слишком большой — нужно не больше 2 МБ.")
        return redirect("/clients")
    text = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        flash("Не смог прочитать файл: кодировка не UTF-8 и не Windows-1251.")
        return redirect("/clients")
    head = text[:2000]
    delim = ";" if head.count(";") >= head.count(",") else ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim)
            if any(cell.strip() for cell in r)]
    if not rows:
        flash("Файл пустой.")
        return redirect("/clients")

    canon = {
        "имя": "name", "месяцев_в_магазине": "months", "покупок_в_месяц": "purchases",
        "средний_чек": "avg_check", "дней_с_последней_покупки": "days_since",
        "жалоб_за_год": "complaints", "использованных_скидок": "discounts",
        "доля_брошенных_корзин": "abandoned", "доля_открытых_писем": "opens",
        "ушёл": "churned", "ушел": "churned", "факт": "churned",
    }
    header = [cell.strip().lower() for cell in rows[0]]
    colmap = {}
    for idx, cell_name in enumerate(header):
        if cell_name in canon:
            colmap[canon[cell_name]] = idx
    if "name" in colmap:
        data_rows = rows[1:]
    else:
        colmap = {k: i for i, k in enumerate(
            ["name", "months", "purchases", "avg_check", "days_since",
             "complaints", "discounts", "abandoned", "opens", "churned"])}
        data_rows = rows

    imported = skipped = 0
    for row in data_rows[:2000]:
        def cell(key):
            i = colmap.get(key)
            return row[i].strip() if i is not None and i < len(row) else ""
        if not cell("name"):
            skipped += 1
            continue
        try:
            values = [float(cell(k).replace(",", ".")) for k in
                      ("months", "purchases", "avg_check", "days_since",
                       "complaints", "discounts", "abandoned", "opens")]
        except ValueError:
            skipped += 1
            continue
        if (not all(math.isfinite(v) for v in values) or
                any(v < FEATURE_HINTS[f]["min"] or v > FEATURE_HINTS[f]["max"]
                    for v, f in zip(values, FEATURES))):
            skipped += 1
            continue
        churn_raw = cell("churned").lower()
        churned = None
        if churn_raw in ("ушёл", "ушел", "да", "true", "yes", "1"):
            churned = True
        elif churn_raw in ("остался", "нет", "false", "no", "0"):
            churned = False
        db.session.add(Client(name=cell("name")[:100], months=values[0],
                              purchases=values[1], avg_check=values[2],
                              days_since=values[3], complaints=values[4],
                              discounts=values[5], abandoned=values[6],
                              opens=values[7], churned=churned))
        imported += 1
    db.session.commit()
    global feature_stats
    feature_stats = None
    msg = "Загрузил {} новых клиентов в базу.".format(imported)
    if skipped:
        msg += " Пропустил {} строк — в них нет имени или числа.".format(skipped)
    flash(msg)
    return redirect("/clients")


@app.route("/clients/<int:client_id>")
def client_page(client_id):
    c = db.get_or_404(Client, client_id)
    risk = net.predict(norm.transform(c.features()))
    verdict = verdict_for(risk)
    fields = []
    for name, value in zip(FEATURES, c.features()):
        fields.append({
            "name": name.replace("_", " "),
            "hint": FEATURE_HINTS[name]["text"],
            "value": value,
        })
    guessed = None
    if c.churned is not None:
        guessed = (risk >= 0.5) == c.churned
    return render_template("client_detail.html", c=c, risk=risk,
                           risk_pct=round(risk * 100), verdict=verdict,
                           fields=fields, guessed=guessed,
                           explain=explain_client(c.features()))


@app.route("/game")
def game_page():
    texts = [FEATURE_HINTS[f]["text"] for f in FEATURES]
    return render_template("game.html", feature_texts=texts)


@app.route("/api/game/new", methods=["POST"])
def api_game_new():
    c = (Client.query
         .filter(Client.churned.is_not(None))
         .order_by(db.func.random())
         .first())
    if c is None:
        return jsonify({"error": "В базе нет клиентов с известным исходом"}), 400
    return jsonify({"id": c.id, "name": c.name, "features": c.features()})


@app.route("/api/game/answer", methods=["POST"])
def api_game_answer():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("id"), int) or isinstance(data.get("id"), bool):
        return jsonify({"error": "Неверные данные игрока"}), 400
    if not isinstance(data.get("guess"), bool):
        return jsonify({"error": "Выбери один из двух вариантов"}), 400
    c = db.session.get(Client, data["id"])
    if c is None or c.churned is None:
        return jsonify({"error": "Такого клиента с известным исходом нет"}), 400
    guess = data.get("guess") is True
    risk = net.predict(norm.transform(c.features()))
    return jsonify({
        "name": c.name,
        "real_churned": c.churned,
        "risk": risk,
        "risk_pct": round(risk * 100),
        "advice": verdict_for(risk),
        "player_right": guess == bool(c.churned),
        "net_right": (risk >= 0.5) == bool(c.churned),
    })


@app.route("/history")
def history_page():
    preds = Prediction.query.order_by(Prediction.id.desc()).limit(100).all()
    return render_template("history.html", preds=preds)


@app.route("/history/clear", methods=["POST"])
def history_clear():
    Prediction.query.delete()
    db.session.commit()
    return render_template("history.html", preds=[])


@app.route("/train")
def train_page():
    rows = Client.query.filter(Client.churned.is_not(None)).all()
    total = len(rows)
    churners = sum(1 for c in rows if c.churned)
    rule_right = rule_found = net_right = net_found = 0
    for c in rows:
        f = c.features()
        truth = bool(c.churned)
        pred_rule = f[3] > 140 or (f[4] >= 3 and f[6] >= 0.5)
        pred_net = net.predict(norm.transform(f)) >= 0.5
        if pred_rule == truth:
            rule_right += 1
        if truth and pred_rule:
            rule_found += 1
        if pred_net == truth:
            net_right += 1
        if truth and pred_net:
            net_found += 1
    baselines = {
        "total": total,
        "churners": churners,
        "always_acc": round(100.0 * (total - churners) / max(1, total), 1),
        "rule_acc": round(100.0 * rule_right / max(1, total), 1),
        "rule_found": rule_found,
        "net_acc": round(100.0 * net_right / max(1, total), 1),
        "net_found": net_found,
    }
    runs = TrainingRun.query.order_by(TrainingRun.id.desc()).limit(10).all()
    return render_template("train.html", info=last_run_info(),
                           baselines=baselines, runs=runs)


@app.route("/api/matrix")
def api_matrix():
    try:
        thr = float(request.args.get("thr", 0.5))
    except (ValueError, TypeError):
        thr = 0.5
    if not math.isfinite(thr):
        thr = 0.5
    thr = min(0.95, max(0.05, thr))
    tp = fp = fn = tn = 0
    for c in Client.query.filter(Client.churned.is_not(None)).all():
        pred = net.predict(norm.transform(c.features())) >= thr
        if c.churned and pred:
            tp += 1
        elif c.churned:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    return jsonify({
        "thr": round(thr, 2),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total,
        "accuracy": round(100.0 * (tp + tn) / max(1, total), 1),
        "recall": round(100.0 * tp / max(1, tp + fn), 1),
        "precision": round(100.0 * tp / max(1, tp + fp), 1),
        "found": tp,
        "total_churners": tp + fn,
        "false_alarms": fp,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Нужен JSON с данными клиента"}), 400
    values = data.get("features")
    if not isinstance(values, list) or len(values) != len(FEATURES):
        return jsonify({"error": "Нужно передать ровно {} чисел".format(len(FEATURES))}), 400
    raw = []
    try:
        for v in values:
            raw.append(float(v))
    except (TypeError, ValueError):
        return jsonify({"error": "Все признаки должны быть числами"}), 400
    if (not all(math.isfinite(v) for v in raw) or
            any(v < FEATURE_HINTS[f]["min"] or v > FEATURE_HINTS[f]["max"]
                for v, f in zip(raw, FEATURES))):
        return jsonify({"error": "Числа должны быть в допустимых пределах"}), 400
    name = str(data.get("name") or "").strip()[:100] or "Клиент без имени"

    risk = net.predict(norm.transform(raw))
    verdict = verdict_for(risk)

    prediction = Prediction(
        client_name=name,
        features=raw,
        risk=round(risk, 4),
        verdict=verdict["text"],
    )
    db.session.add(prediction)
    db.session.commit()

    return jsonify({
        "name": name,
        "risk": risk,
        "risk_pct": round(risk * 100),
        "advice": verdict,
        "features": raw,
        "explain": explain_client(raw),
        "saved": True,
    })


@app.route("/api/random-client", methods=["GET"])
def api_random_client():
    c = (Client.query
         .filter(Client.churned.is_not(None))
         .order_by(db.func.random())
         .first())
    if c is None:
        return jsonify({"error": "В базе нет клиентов с известным исходом"}), 404
    risk = net.predict(norm.transform(c.features()))
    return jsonify({
        "name": c.name,
        "features": c.features(),
        "real_churned": c.churned,
        "risk": risk,
        "risk_pct": round(risk * 100),
        "advice": verdict_for(risk),
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    global net, norm, feature_stats
    info = run_training(seed=None, verbose=False)
    net, norm = load_model()
    feature_stats = None
    return jsonify(info)


@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", code=404,
                           message="Такой страницы нет. Может, опечатка в адресе?"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500,
                           message="Что-то сломалось на сервере. Уже разбираюсь!"), 500


if __name__ == "__main__":
    prepare_app()
    app.run(debug=False, use_reloader=False,
            host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
