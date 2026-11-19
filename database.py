import sys

from app import app
from dataset import generate_clients
from models import db, Client, Prediction, TrainingRun, TrainedModel
from training import run_training

if __name__ == "__main__":
    fresh = "--fresh" in sys.argv

    with app.app_context():
        if fresh:
            print("Сношу старую базу (все таблицы)...")
            db.drop_all()
        db.create_all()

        if Client.query.count() == 0:
            print("Заселяю базу клиентами...")
            for c in generate_clients():
                db.session.add(Client(**c))
            db.session.commit()
            gone = Client.query.filter_by(churned=True).count()
            print("Заселил {} клиентов, из них ушли {} ({:.0f}%)".format(
                Client.query.count(), gone,
                100.0 * gone / max(1, Client.query.count())))
        else:
            print("Клиенты в базе уже есть — не трогаю.")

        if fresh or TrainedModel.query.count() == 0:
            run_training(verbose=True)

    print("\nБаза готова: neuro.db (рядом с файлами проекта).")
    print("Запускай сайт: python3 app.py и открывай http://127.0.0.1:5000")
