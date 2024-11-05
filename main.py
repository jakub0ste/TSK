import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
import tkinter as tk
from tkinter import filedialog
import openpyxl


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

    def AR(self, data):
        df = pd.DataFrame(data, columns=['Value'])
        for i in range(1, self.p + 1):
            df[f'Lags_{i}'] = df['Value'].shift(i)

        # Drop rows with NaN values created by lagging
        df.dropna(inplace=True)

        X = df.iloc[:, 1:].values  # Select lagged values as features
        y = df['Value'].values  # Original values as target

        model = LinearRegression()
        model.fit(X, y)
        return model

    def MA(self, residuals):
        res_df = pd.DataFrame(residuals, columns=['Residuals'])
        for i in range(1, self.q + 1):
            res_df[f'Lags_{i}'] = res_df['Residuals'].shift(i)

        # Drop NaN values created by lagging
        res_df.dropna(inplace=True)

        X = res_df.iloc[:, 1:].values
        y = res_df['Residuals'].values

        model = LinearRegression()
        model.fit(X, y)
        return model

    def calculate(self, data, forecast_steps=5):
        diff_data = difference(data['Value'].values, self.d)
        if len(diff_data) <= self.p:
            raise ValueError(f"Insufficient data for {self.p} AR lags after differencing.")

        # AR model fitting
        ar_model = self.AR(diff_data['Value'].values)

        last_lags = diff_data[-self.p:].values.reshape(1, -1)

        ar_pred = ar_model.predict(last_lags)

        # Calculate residuals
        residuals = diff_data['Value'].values - ar_pred

        # MA model fitting
        if self.q > 0:
            ma_model = self.MA(residuals)
            last_q_lags = residuals[-self.q:] if self.q > 1 else [residuals[-1]]
            last_q_lags = np.array(last_q_lags).reshape(1, -1)
            ma_pred = ma_model.predict(last_q_lags)
        else:
            ma_pred = 0

        # Sum AR and MA predictions
        predictions = ar_pred + ma_pred

        reverted_predictions = np.zeros_like(predictions)
        reverted_predictions[0] = predictions[0] + data['Value'].iloc[-(self.d + 1)]

        for i in range(1, len(predictions)):
            reverted_predictions[i] = predictions[i] + reverted_predictions[i - 1]

        # Future predictions (can be further adapted for iterative forecasting)
        forecast = [reverted_predictions[-1] + (i + 1) * np.mean(predictions) for i in range(forecast_steps)]

        return reverted_predictions, forecast


def adf_test(series, max_lag=1):
    adf_result = adfuller(series)
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]

    return adf_statistic, p_value, critical_values


def find_d(series):
    d = 0
    p_value = adf_test(series)[1]

    while p_value >= 0.05:
        series = np.diff(series)
        d += 1
        p_value = adf_test(series)[1]

    return d


def find_p_q(series, d):
    diff_series = series.copy()
    for _ in range(d):
        diff_series = np.diff(diff_series)

    n = len(diff_series)
    max_lags = min(20, n // 2)

    # Rysowanie wykresów ACF i PACF
    plt.figure(figsize=(12, 6))

    # ACF
    plt.subplot(1, 2, 1)
    plot_acf(diff_series, lags=max_lags, ax=plt.gca())
    plt.title('ACF')

    # PACF
    plt.subplot(1, 2, 2)
    plot_pacf(diff_series, lags=max_lags, ax=plt.gca())
    plt.title('PACF')

    plt.tight_layout()
    plt.show()

    # Wartości p i q można określić na podstawie wykresów ACF i PACF
    p = int(input("Wprowadź wartość p na podstawie wykresu PACF: "))
    q = int(input("Wprowadź wartość q na podstawie wykresu ACF: "))

    return p, q


def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    return file_path


def difference(data, d):
    diff_data = data.copy()
    for _ in range(d):
        diff_data = np.diff(diff_data)
    return pd.DataFrame(diff_data, columns=['Value'])


def main():
    pd.options.mode.copy_on_write = True
    kalkulator = KalkulatorPKB()
    # wartości głównie przypadkowe
    pkb_dochodowa = kalkulator.pkb_metoda_dochodowa(100801, 457823, 574000, 50000)

    pkb_wydatkowa = kalkulator.pkb_metoda_wydatkowa(140000, 60000, 40000, 30000, 20000)

    pkb_produkcja = kalkulator.pkb_metoda_produkcji(1200000, 40000)

    print(f"PKB metodą dochodową: {pkb_dochodowa} mln zł")
    print(f"PKB metodą wydatkową: {pkb_wydatkowa} mln zł")
    print(f"PKB metodą produkcyjną: {pkb_produkcja} mln zł")

    file_path = choose_file()
    if file_path:
        df = pd.read_excel(file_path, header=None)

        df.columns = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)

        df = df.T
        df.columns = ['Value']
        df.index.name = 'Year'

        df['Value'] = pd.to_numeric(df['Value'], errors='coerce') * 1_000_000

        df_Values = df['Value']
        d = find_d(df_Values.dropna())
        p, q = find_p_q(df_Values.dropna(), d)
        arima_model = PredykcjaPKB(p, d, q)
        predictions, forecast = arima_model.calculate(df)
        print(df)
        print(predictions)
        print(forecast)

        # Wyświetlenie wyników
        plt.figure(figsize=(20, 6))
        forecast_years = np.arange(df.index[-1] + 1, df.index[-1] + 1 + len(forecast))
        combined_years = np.concatenate([df.index, forecast_years])
        combined_values = np.concatenate([df['Value'].values, forecast])

        # Plot everything together for a continuous line
        plt.plot(combined_years, combined_values, label='Połączone prognozy', color='red')
        plt.plot(df.index, df['Value'], label='Oryginalne wartości')
        plt.plot(df.index[-len(predictions):], predictions, label='Prognozy ARIMA', linestyle='--')
        plt.plot(forecast_years, forecast, label='Prognozy ARIMA', linestyle=':')
        # plt.xlim(df.index.min(), df_c.index.max() + n_lat)
        plt.grid()
        plt.legend()
        plt.title("Porównanie oryginalnych wartości i prognoz ARIMA")
        plt.xlabel("Rok")
        plt.ylabel("PKB(w mln zł)")
        plt.show()

    else:
        print("Nie wybrano pliku.")


if __name__ == '__main__':
    main()
