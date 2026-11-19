import os
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "neuro.db")

db = SQLAlchemy()


class Client(db.Model):

    __tablename__ = "clients"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    months = db.Column(db.Float, nullable=False)
    purchases = db.Column(db.Float, nullable=False)
    avg_check = db.Column(db.Float, nullable=False)
    days_since = db.Column(db.Float, nullable=False)
    complaints = db.Column(db.Float, nullable=False)
    discounts = db.Column(db.Float, nullable=False)
    abandoned = db.Column(db.Float, nullable=False)
    opens = db.Column(db.Float, nullable=False)
    churned = db.Column(db.Boolean)
    created_at = db.Column(db.DateTime, default=datetime.now)

    FEATURE_FIELDS = ["months", "purchases", "avg_check", "days_since",
                      "complaints", "discounts", "abandoned", "opens"]

    def features(self):
        return [getattr(self, field) for field in self.FEATURE_FIELDS]


class Prediction(db.Model):

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    client_name = db.Column(db.String(100), nullable=False)
    features = db.Column(db.JSON, nullable=False)
    risk = db.Column(db.Float, nullable=False)
    verdict = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)


class TrainingRun(db.Model):

    __tablename__ = "training_runs"

    id = db.Column(db.Integer, primary_key=True)
    epochs = db.Column(db.Integer, nullable=False)
    lr = db.Column(db.Float, nullable=False)
    seed = db.Column(db.String(20), nullable=False)
    hidden = db.Column(db.Integer, nullable=False)
    weights = db.Column(db.Integer, nullable=False)
    n_clients = db.Column(db.Integer, nullable=False)
    n_test = db.Column(db.Integer, nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    right = db.Column(db.Integer, nullable=False)
    found = db.Column(db.Integer, nullable=False)
    total_churners = db.Column(db.Integer, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    false_alarms = db.Column(db.Integer, nullable=False)
    error_start = db.Column(db.Float, nullable=False)
    error_end = db.Column(db.Float, nullable=False)
    seconds = db.Column(db.Float, nullable=False)
    history = db.Column(db.JSON, nullable=False)
    trained_at = db.Column(db.String(20), nullable=False)


class TrainedModel(db.Model):

    __tablename__ = "trained_models"

    id = db.Column(db.Integer, primary_key=True)
    sizes = db.Column(db.JSON, nullable=False)
    layers = db.Column(db.JSON, nullable=False)
    mins = db.Column(db.JSON, nullable=False)
    maxs = db.Column(db.JSON, nullable=False)
    features = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    run_id = db.Column(db.Integer)
