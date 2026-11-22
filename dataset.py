import math
import random

FEATURES = [
    "месяцев_в_магазине",
    "покупок_в_месяц",
    "средний_чек",
    "дней_с_последней_покупки",
    "жалоб_за_год",
    "использованных_скидок",
    "доля_брошенных_корзин",
    "доля_открытых_писем",
]

FEATURE_HINTS = {
    "месяцев_в_магазине": {"text": "сколько месяцев клиент с нами", "min": 1, "max": 48, "step": 1},
    "покупок_в_месяц": {"text": "покупок в месяц в среднем", "min": 0, "max": 10, "step": 0.5},
    "средний_чек": {"text": "средний чек, рублей", "min": 500, "max": 15000, "step": 100},
    "дней_с_последней_покупки": {"text": "дней с последней покупки", "min": 0, "max": 180, "step": 5},
    "жалоб_за_год": {"text": "жалоб в поддержку за год", "min": 0, "max": 5, "step": 1},
    "использованных_скидок": {"text": "сколько раз воспользовался скидками", "min": 0, "max": 10, "step": 1},
    "доля_брошенных_корзин": {"text": "доля брошенных корзин", "min": 0.0, "max": 1.0, "step": 0.05},
    "доля_открытых_писем": {"text": "доля открытых писем рассылки", "min": 0.0, "max": 1.0, "step": 0.05},
}

N_CLIENTS = 1000

NAMES = ["Алексей", "Аня", "Артём", "Борис", "Вера", "Влад", "Гоша", "Даша",
         "Дима", "Егор", "Игорь", "Инна", "Катя", "Кирилл", "Лиза", "Максим",
         "Марина", "Настя", "Олег", "Полина", "Рома", "Света", "Тимур", "Юля"]

INITIALS = "АБВГДЕЗИКМНОПРСТУЯ"


def _make_client(rnd):
    name = rnd.choice(NAMES) + " " + rnd.choice(INITIALS) + "."
    months = rnd.randint(1, 48)
    purchases = round(rnd.uniform(0.0, 10.0), 1)
    avg_check = rnd.randint(500, 15000)
    days_since = int(rnd.triangular(0, 180, 15))
    complaints = int(rnd.random() ** 2 * 6)
    discounts = rnd.randint(0, 10)
    abandon = min(1.0, max(0.0, rnd.gauss(0.4, 0.25) + 0.2 * days_since / 180.0))
    opens = min(1.0, max(0.0, rnd.gauss(0.55, 0.25) - 0.25 * days_since / 180.0))
    return {
        "name": name,
        "months": months,
        "purchases": purchases,
        "avg_check": avg_check,
        "days_since": days_since,
        "complaints": complaints,
        "discounts": discounts,
        "abandoned": round(abandon, 2),
        "opens": round(opens, 2),
    }


def _churn_probability(client, rnd):
    score = (
        1.6 * client["days_since"] / 180.0
        + 1.0 * client["complaints"] / 5.0
        + 0.9 * client["abandoned"]
        - 1.2 * client["purchases"] / 10.0
        - 1.6 * client["opens"]
        - 0.7 * client["months"] / 48.0
        + rnd.gauss(0.0, 0.35)
    )
    return 1.0 / (1.0 + math.exp(-6.0 * (score - 0.15)))


def generate_clients(n=N_CLIENTS, seed=7):
    rnd = random.Random(seed)
    clients = []
    for _ in range(n):
        client = _make_client(rnd)
        client["churned"] = rnd.random() < _churn_probability(client, rnd)
        clients.append(client)
    return clients


class Normalizer:

    def fit(self, rows):
        self.mins = []
        self.maxs = []
        for i in range(len(rows[0])):
            column = [row[i] for row in rows]
            self.mins.append(min(column))
            self.maxs.append(max(column))

    def transform(self, row):
        result = []
        for i in range(len(row)):
            span = self.maxs[i] - self.mins[i]
            if span == 0:
                span = 1.0
            result.append((row[i] - self.mins[i]) / span)
        return result
