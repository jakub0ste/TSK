import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler


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
        self.p = p
        self.d = d
        self.q = q
        self.coef_AR = np.zeros(p)
        self.coef_MA = np.zeros(q)
        self.errors = np.zeros(q)

    def difference(self, series):
        diff = series.copy()
        for _ in range(self.d):
            diff = np.diff(diff, n=1)
        return diff

    def fit(self, series):
        series_diff = self.difference(series)
        n = len(series_diff)

        # Autoregresja (AR)
        X_AR = np.array([series_diff[i - self.p:i] for i in range(self.p, n)])
        y_AR = series_diff[self.p:]

        if len(X_AR) > 0:
            self.coef_AR = np.linalg.pinv(X_AR).dot(y_AR)

    def predict(self, series, n_steps):
        predictions = []
        series_diff = self.difference(series)
        last_values = series_diff[-self.p:].copy()

        for _ in range(n_steps):
            # Predykcja AR
            pred_AR = np.dot(self.coef_AR, last_values[-self.p:]) if self.p > 0 else 0

            # Predykcja MA
            pred_MA = np.dot(self.coef_MA, self.errors) if self.q > 0 else 0

            # Final prediction
            pred = pred_AR + pred_MA
            predictions.append(pred)

            # Update for next iteration
            true_value_index = len(series) + len(predictions) - 1
            true_value = series[true_value_index] if true_value_index < len(series) else pred
            error = true_value - pred

            # Update last values and errors
            last_values = np.append(last_values, pred)[-self.p:]
            self.errors = np.roll(self.errors, -1)
            self.errors[-1] = error

        return predictions

    def create_lags_p(self, series):
        n = len(series)
        lags = np.zeros((n - self.p, self.p + 1))

        for i in range(self.p + 1):
            lags[:, i] = series[self.p - i: n - i]

        return lags

    def create_lags_q(self, residuals):
        n = len(residuals)
        lags = np.zeros((n - self.q, self.q + 1))

        for i in range(self.q + 1):
            lags[:, i] = residuals[self.q - i: n - i]

        return lags

    def AR(self, series):
        # lags = self.create_lags_p(series)
        # train_size = int(0.8 * lags.shape[0])
        # train_data = lags[:train_size, :]
        # test_data = lags[train_size:, :]
        #
        # # Ustalanie X (przesunięte wartości) i y (wartości docelowe)
        # X_train = train_data[:, 1:]
        # y_train = train_data[:, 0].reshape(-1, 1)
        # X_test = test_data[:, 1:]
        # y_test = test_data[:, 0].reshape(-1, 1)
        #
        # theta = linear_regression(y_train, X_train)
        #
        # # Obliczanie wartości przewidywanych dla zbioru treningowego
        # y_train_pred = X_train @ theta
        # # Obliczanie wartości przewidywanych dla zbioru testowego
        # y_test_pred = X_test @ theta
        #
        # residuals_train = y_train.flatten() - y_train_pred.flatten()
        # residuals_test = y_test.flatten() - y_test_pred.flatten()
        #
        # # Łącz reszty do dalszego użycia
        # residuals = np.concatenate((residuals_train, residuals_test))
        #
        # # Obliczanie błędu średniokwadratowego (RMSE)
        # rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        #
        # return {
        #     "train_predictions": y_train_pred.flatten(),
        #     "test_predictions": y_test_pred.flatten(),
        #     "coefficients": theta,
        #     "residuals": residuals,
        #     "rmse": rmse
        # }
        self.fit(series)
        return self.predict(series, n_steps=len(series) - self.p)

    def MA(self, residuals):
        # lags = self.create_lags_q(res)
        # train_size = int(0.8 * lags.shape[0])
        # train_data = lags[:train_size, :]
        # test_data = lags[train_size:, :]
        # # Ustalanie X (przesunięte wartości) i y (wartości docelowe)
        # X_train = train_data[:, 1:]
        # y_train = train_data[:, 0].reshape(-1, 1)
        # X_test = test_data[:, 1:]
        # y_test = test_data[:, 0].reshape(-1, 1)
        # theta = linear_regression(y_train, X_train)
        # # Obliczanie wartości przewidywanych dla zbioru treningowego
        # y_train_pred = X_train @ theta
        # # Obliczanie wartości przewidywanych dla zbioru testowego
        # y_test_pred = X_test @ theta
        #
        # rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        #
        # return {
        #     "train_predictions": y_train_pred,
        #     "test_predictions": y_test_pred,
        #     "coefficients": theta,
        #     "rmse": rmse
        # }
        n = len(residuals)
        if n < self.q:
            raise ValueError("Not enough residuals for MA model.")

        self.errors = residuals[-self.q:]  # Initialize the errors

        # We need to predict the next values based on errors
        return self.predict(residuals, n_steps=len(residuals) - self.q)


def linear_regression(y, X):
    # X.T * X - macierz kowariancji
    XtX_inv = np.linalg.inv(X.T @ X)
    # (X.T * X)^(-1) * X.T * y - estymacja współczynników
    return XtX_inv @ X.T @ y


def adf_test(series, max_lag=1):
    n = len(series)
    # 1. Różnicowanie szeregu czasowego
    y_diff = np.diff(series)
    y_diff = y_diff[max_lag:]

    # 2. Tworzenie regresorów z opóźnionych wartości
    lagged_series = series[:-1]
    lagged_series = lagged_series[max_lag:]

    X = np.column_stack([lagged_series] + [np.roll(y_diff, i) for i in range(1, max_lag + 1)])
    X = np.column_stack((np.ones(len(X)), X))  # Dodajemy stałą

    # 3. Regresja OLS dla obliczenia statystyki ADF
    beta = linear_regression(y_diff, X)

    # 4. Statystyka testowa ADF: beta[1] / błędy standardowe
    y_pred = X @ beta
    residuals = y_diff - y_pred

    sse = np.sum(residuals ** 2)
    sigma = np.sqrt(sse / (len(y_diff) - len(beta)))
    se_beta1 = sigma / np.sqrt(np.sum((lagged_series - np.mean(lagged_series)) ** 2))

    adf_statistic = beta[1] / se_beta1

    # 5. P-wartość i wartości krytyczne (szacowane manualnie)
    p_value = 2 * (1 - stats.norm.cdf(abs(adf_statistic)))
    critical_values = {
        "1%": -3.430,
        "5%": -2.860,
        "10%": -2.570
    }

    return adf_statistic, p_value, max_lag, critical_values


def find_d(series):
    d = 0
    while True:
        p_value = adf_test(series)[1]
        if p_value < 0.05:  # Test stacjonarności jest istotny
            break
        series = np.diff(series)
        d += 1
    return d


def find_p_q(series, d):
    # Najpierw różnicujemy dane d razy
    for _ in range(d):
        series = np.diff(series)

    # Rysujemy ACF i PACF
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_acf(series, lags=20, ax=plt.gca())
    plt.title('ACF')

    plt.subplot(1, 2, 2)
    plot_pacf(series, lags=20, ax=plt.gca())
    plt.title('PACF')

    plt.tight_layout()
    plt.show()

    # Wartości p i q można określić na podstawie wizualizacji ACF i PACF
    p = int(input("Wprowadź wartość p na podstawie wykresu PACF: "))
    q = int(input("Wprowadź wartość q na podstawie wykresu ACF: "))

    return p, q


def shift(array, num_shift):
    """Przesuwa wartości w tablicy o `num_shift` miejsc."""
    result = np.empty_like(array)
    if num_shift > 0:
        result[:num_shift] = np.nan
        result[num_shift:] = array[:-num_shift]
    elif num_shift < 0:
        result[num_shift:] = np.nan
        result[:num_shift] = array[-num_shift:]
    else:
        result[:] = array
    return result


def diff(array, lag):
    return array - shift(array, lag)


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

    d = find_d(data)
    p, q = find_p_q(data, d)
    arima_model = PredykcjaPKB(p=p, d=d, q=q)
    AR_predictions = arima_model.AR(data)

    # Calculate residuals for MA model
    residuals = data[arima_model.p:] - AR_predictions

    MA_predictions = arima_model.MA(residuals)

    # Combine AR and MA predictions
    predictions = AR_predictions + MA_predictions

    predictions = np.exp(predictions)
    predictions *= max(data)

    plt.plot(range(len(data)), data, label='Dane rzeczywiste')
    plt.plot(range(len(data), len(data) + len(predictions)), predictions, label='Prognoza ARIMA', color='red')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
