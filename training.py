import random
import time

from models import db, Client, TrainingRun, TrainedModel
from dataset import FEATURES, Normalizer
from neuronet import NeuralNetwork

EPOCHS = 400
LEARNING_RATE = 0.3
SEED = 42


def run_training(epochs=EPOCHS, lr=LEARNING_RATE, seed=SEED, verbose=True):

    def say(text):
        if verbose:
            print(text)

    random.seed(seed)

    clients = (Client.query
               .filter(Client.churned.is_not(None))
               .order_by(Client.id)
               .all())
    say("Достал из базы {} клиентов".format(len(clients)))
    gone = sum(1 for c in clients if c.churned)
    say("Из них ушли: {} ({:.0f}%)".format(gone, 100.0 * gone / len(clients)))

    X_raw = [c.features() for c in clients]
    y = [1.0 if c.churned else 0.0 for c in clients]

    norm = Normalizer()
    norm.fit(X_raw)
    X = [norm.transform(row) for row in X_raw]

    indices = list(range(len(clients)))
    random.shuffle(indices)
    cut = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:cut], indices[cut:]

    train_data = []
    for i in train_idx:
        train_data.append((X[i], [y[i]]))
        if y[i] == 1.0:
            train_data.append((X[i], [y[i]]))

    net = NeuralNetwork([len(FEATURES), 16, 1])
    say("Архитектура сети: {} -> {} -> 1, всего весов: {}".format(
        net.sizes[0], net.sizes[1], net.count_weights()))

    start = time.time()
    history = net.train(train_data, epochs=epochs, lr=lr)
    seconds = time.time() - start
    say("Обучение заняло {:.1f} сек. Ошибка упала с {:.4f} до {:.4f}.".format(
        seconds, history[0], history[-1]))

    right = 0
    found_churners = 0
    total_churners = 0
    false_alarms = 0
    for i in test_idx:
        risk = net.predict(X[i])
        predicted = 1.0 if risk >= 0.5 else 0.0
        if predicted == y[i]:
            right += 1
        if y[i] == 1.0:
            total_churners += 1
            if predicted == 1.0:
                found_churners += 1
        elif predicted == 1.0:
            false_alarms += 1

    n_test = len(test_idx)
    accuracy = 100.0 * right / n_test
    recall = 100.0 * found_churners / max(1, total_churners)
    say("Экзамен: угадано {}/{} ({:.0f}%), беглецов найдено {}/{} ({:.0f}%), ложных тревог: {}".format(
        right, n_test, accuracy, found_churners, total_churners, recall, false_alarms))

    run = TrainingRun(
        epochs=epochs,
        lr=lr,
        seed=str(seed) if seed is not None else "случайный",
        hidden=net.sizes[1],
        weights=net.count_weights(),
        n_clients=len(clients),
        n_test=n_test,
        accuracy=round(accuracy, 1),
        right=right,
        found=found_churners,
        total_churners=total_churners,
        recall=round(recall, 1),
        false_alarms=false_alarms,
        error_start=round(history[0], 4),
        error_end=round(history[-1], 4),
        seconds=round(seconds, 1),
        history=[round(h, 4) for h in history],
        trained_at=time.strftime("%d.%m.%Y %H:%M"),
    )
    db.session.add(run)
    db.session.flush()

    exported = net.export()
    model_row = TrainedModel(
        sizes=exported["sizes"],
        layers=exported["layers"],
        mins=norm.mins,
        maxs=norm.maxs,
        features=FEATURES,
        run_id=run.id,
    )
    db.session.add(model_row)
    db.session.commit()
    say("Сеть и статистика сохранены в базу neuro.db")

    info = {
        "id": run.id,
        "epochs": epochs,
        "lr": lr,
        "seed": run.seed,
        "seconds": run.seconds,
        "weights": run.weights,
        "hidden": run.hidden,
        "n_clients": run.n_clients,
        "n_test": run.n_test,
        "accuracy": run.accuracy,
        "right": right,
        "found": run.found,
        "total_churners": run.total_churners,
        "recall": run.recall,
        "false_alarms": run.false_alarms,
        "error_start": run.error_start,
        "error_end": run.error_end,
        "history": run.history,
        "trained_at": run.trained_at,
    }
    return info
