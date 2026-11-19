import io
import math
import os

from flask import Flask, jsonify, render_template, request, send_file
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


def load_model():
    row = TrainedModel.query.order_by(TrainedModel.id.desc()).first()
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
    return render_template("how.html")


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
        ws.append([i, r["name"], *r["features"], r["risk_pct"],
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
                           fields=fields, guessed=guessed)


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
    return render_template("train.html", info=last_run_info())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    values = data.get("features")
    if not isinstance(values, list) or len(values) != len(FEATURES):
        return jsonify({"error": "Нужно передать ровно {} чисел".format(len(FEATURES))}), 400
    raw = []
    try:
        for v in values:
            raw.append(float(v))
    except (TypeError, ValueError):
        return jsonify({"error": "Все признаки должны быть числами"}), 400
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
        "saved": True,
    })


@app.route("/api/random-client", methods=["GET"])
def api_random_client():
    c = (Client.query
         .filter(Client.churned.is_not(None))
         .order_by(db.func.random())
         .first())
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
    global net, norm
    info = run_training(seed=None, verbose=False)
    net, norm = load_model()
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
    app.run(debug=True, use_reloader=False,
            host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
