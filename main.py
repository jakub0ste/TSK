import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# from pmdarima import auto_arima


class KalkulatorPKB:
    def __init__(self):
        pass

    def pkb_metoda_dochodowa(self, wynagrodzenia, kapital, podatki, dotacje):
        pkb = wynagrodzenia + kapital + podatki - dotacje
        return pkb

    def pkb_metoda_wydatkowa(self, konsumpcja, inwestycje, wydatki_rzadowe, eksport, importy):
        pkb = konsumpcja + inwestycje + wydatki_rzadowe + (eksport - importy)
        return pkb

    def pkb_metoda_produkcji(self, produkcja, zuzycie):
        pkb = produkcja - zuzycie
        return pkb


class PredykcjaPKB:

    def __init__(self, p, d, q):
        self.errors = None
        self.p = p
        self.d = d
        self.q = q
        self.coef_AR = np.random.normal(0, 0.1, p)
        self.coef_MA = np.random.normal(0, 0.1, q)

    def difference(self, series):
        diff = series
        for _ in range(self.d):
            diff = np.diff(diff, n=1)
        return diff

    def fit(self, series):
        series_diff = self.difference(series)
        n = len(series_diff)

        # Autoregresja (AR)
        X_AR = np.array([series_diff[i - self.p:i] for i in range(self.p, n)])
        y_AR = series_diff[self.p:]

        # Dopasowanie współczynników AR za pomocą pseudoodwrotności
        if len(X_AR) > 0:
            self.coef_AR = np.linalg.pinv(X_AR).dot(y_AR)

        # Przygotowanie miejsca na błędy dla modelu MA
        self.errors = np.zeros(self.q)

    def predict(self, series, n_steps):
        predictions = []
        series_diff = self.difference(series)
        last_values = series_diff[-self.p:].copy()  # Ostatnie wartości dla części AR

        for _ in range(n_steps):
            pred_AR = np.dot(self.coef_AR, last_values[-self.p:]) if self.p > 0 else 0

            pred_MA = np.dot(self.coef_MA, self.errors) if self.q > 0 else 0

            pred = pred_AR + pred_MA
            predictions.append(pred)

            true_value = series[len(series) + len(predictions) - 1] if len(series) + len(predictions) - 1 < len(
                series) else pred
            error = true_value - pred

            last_values = np.append(last_values, pred)[-self.p:]
            self.errors = np.roll(self.errors, -1)
            self.errors[-1] = error

        return predictions


def main():
    kalkulator = KalkulatorPKB()
    # wartości głównie przypadkowe
    pkb_dochodowa = kalkulator.pkb_metoda_dochodowa(100801, 457823, 574000, 50000)

    pkb_wydatkowa = kalkulator.pkb_metoda_wydatkowa(140000, 60000, 40000, 30000, 20000)

    pkb_produkcja = kalkulator.pkb_metoda_produkcji(1200000, 40000)

    print(f"PKB metodą dochodową: {pkb_dochodowa} mln zł")
    print(f"PKB metodą wydatkową: {pkb_wydatkowa} mln zł")
    print(f"PKB metodą produkcyjną: {pkb_produkcja} mln zł")

    np.random.seed(0)
    data = np.cumsum(np.random.normal(0, 1, 100)) + 50  # Losowe dane symulujące PKB

    arima_model = PredykcjaPKB(p=2, d=1, q=2)
    arima_model.fit(data)

    predictions = arima_model.predict(data, n_steps=10)

    plt.plot(range(len(data)), data, label='Dane rzeczywiste')
    plt.plot(range(len(data), len(data) + len(predictions)), predictions, label='Prognoza ARIMA', color='red')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
