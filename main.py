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

    def AR(self, df, n_przewidywan = 5):

        df_temp = df

        # Generating the lagged p terms
        for i in range(1, self.p + 1):
            df_temp['Shifted_values_%d' % i] = df_temp['Value'].shift(i)

        df_temp = df_temp.ffill()

        train_size = int(0.8 * df_temp.shape[0])

        # Breaking data set into test and training
        df_train = pd.DataFrame(df_temp[0:train_size])
        df_test = pd.DataFrame(df_temp[train_size:df.shape[0]])

        df_train_2 = df_train.dropna()
        # X contains the lagged values ,hence we skip the first column
        X_train = df_train_2.iloc[:, 1:].values.reshape(-1, self.p)
        # Y contains the value,it is the first column
        y_train = df_train_2.iloc[:, 0].values.reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_
        df_train_2['Predicted_Values'] = X_train.dot(theta) + intercept

        X_test = df_test.iloc[:, 1:].values.reshape(-1, self.p)
        df_test['Predicted_Values'] = X_test.dot(theta) + intercept

        last_values = df['Value'].iloc[-self.p:].values
        future_predictions = []
        for _ in range(n_przewidywan):
            future_pred = lr.predict(last_values.reshape(1, -1))[0, 0]
            #future_pred = max(future_pred, 0)
            future_predictions.append(future_pred)

            # Update last_values for next prediction
            last_values = np.roll(last_values, -1)
            last_values[-1] = future_pred

        offset = df_test.index.max() - df.shape[0] + 1
        df_future = pd.DataFrame({'Predicted_Values': future_predictions}, index=range(df.shape[0]+offset, df.shape[0] + n_przewidywan+offset))
        return [df_train_2, df_test, df_future]

    def MA(self, res):

        for i in range(1, self.q + 1):
            res['Shifted_values_%d' % i] = res['Residuals'].shift(i)

        train_size = int(0.8 * res.shape[0])

        res_train = pd.DataFrame(res[0:train_size])
        res_test = pd.DataFrame(res[train_size:res.shape[0]])

        res_train_2 = res_train.dropna()
        X_train = res_train_2.iloc[:, 1:].values.reshape(-1, self.q)
        y_train = res_train_2.iloc[:, 0].values.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_
        res_train_2['Predicted_Values'] = X_train.dot(theta) + intercept

        X_test = res_test.iloc[:, 1:].values.reshape(-1, self.q)
        res_test['Predicted_Values'] = X_test.dot(theta) + intercept
        return [res_train_2, res_test]


def linear_regression(y, X):
    # X.T * X - macierz kowariancji
    XtX_inv = np.linalg.inv(X.T @ X)
    # (X.T * X)^(-1) * X.T * y - estymacja współczynników
    return XtX_inv @ X.T @ y


def adf_test(series, max_lag=1):
    # n = len(series)
    # # 1. Różnicowanie szeregu czasowego
    # y_diff = np.diff(series)
    # y_diff = y_diff[max_lag:]
    #
    # # 2. Tworzenie regresorów z opóźnionych wartości
    # lagged_series = series[:-1]
    # lagged_series = lagged_series[max_lag:]
    #
    # X = np.column_stack([lagged_series] + [np.roll(y_diff, i) for i in range(1, max_lag + 1)])
    # X = np.column_stack((np.ones(len(X)), X))  # Dodajemy stałą
    #
    # # 3. Regresja OLS dla obliczenia statystyki ADF
    # beta = linear_regression(y_diff, X)
    #
    # # 4. Statystyka testowa ADF: beta[1] / błędy standardowe
    # y_pred = X @ beta
    # residuals = y_diff - y_pred
    #
    # sse = np.sum(residuals ** 2)
    # sigma = np.sqrt(sse / (len(y_diff) - len(beta)))
    # se_beta1 = sigma / np.sqrt(np.sum((lagged_series - np.mean(lagged_series)) ** 2))
    #
    # adf_statistic = beta[1] / se_beta1
    #
    # # 5. P-wartość i wartości krytyczne (szacowane manualnie)
    # p_value = 2 * (1 - stats.norm.cdf(abs(adf_statistic)))
    # critical_values = {
    #     "1%": -3.430,
    #     "5%": -2.860,
    #     "10%": -2.570
    # }
    #
    # return adf_statistic, p_value, max_lag, critical_values
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

    # np.random.seed(0)
    # data = np.cumsum(np.random.normal(0, 1, 100)) + 50  # Losowe dane symulujące PKB

    file_path = choose_file()
    if file_path:
        df = pd.read_excel(file_path, header=None)

        df.columns = df.iloc[0]
        df = df.drop(0).reset_index(drop=True)

        df = df.T
        df.columns = ['Value']
        df.index.name = 'Year'

        df['Value'] = pd.to_numeric(df['Value'], errors='coerce') * 1_000_000

        df_testing = pd.DataFrame(np.log(df.Value).diff().diff(1))

        d = find_d(df_testing.Value.dropna())
        p, q = find_p_q(df_testing.Value.dropna(), d)
        arima_model = PredykcjaPKB(p, d, q)
        n_lat = int(input("Podaj ile lat przewidzieć: "))
        [df_train, df_test, df_future] = arima_model.AR(pd.DataFrame(df_testing.Value), n_lat)
        df_c = pd.concat([df_train, df_test, df_future])

        res = pd.DataFrame()
        res['Residuals'] = df_c['Value'] - df_c['Predicted_Values']
        [res_train, res_test] = arima_model.MA(pd.DataFrame(res.Residuals))
        res_c = pd.concat([res_train, res_test])
        if 'Predicted_Values' in res_c.columns:
            df_c['Predicted_Values'] = df_c['Predicted_Values'].add(res_c['Predicted_Values'], fill_value=0)

        # df_c['Value'] = df_c['Value'].replace(0, np.nan)
        # df_c['Predicted_Values'] = df_c['Predicted_Values'].replace(0, np.nan)
        # df_c['Value'] = np.log(df_c['Value']).shift(1)
        # df_c['Predicted_Values'] = np.log(df_c['Predicted_Values']).shift(1)
        # df_c['Value'] = df_c['Value'].bfill()
        # df_c['Predicted_Values'] = df_c['Predicted_Values'].bfill()
        # df_c['Value'] = np.exp(df_c['Value'])
        # df_c['Predicted_Values'] = np.exp(df_c['Predicted_Values'])

        #df_c.Predicted_Values += res_c.Predicted_Values
        df_c.Value += np.log(df).shift(1).Value
        df_c.Value += np.log(df).diff().shift(1).Value
        df_c.Predicted_Values += np.log(df).shift(1).Value
        df_c.Predicted_Values += np.log(df).diff().shift(1).Value
        last_valid_value = df_c['Predicted_Values'].iloc[-n_lat - 1]
        df['Growth_Rate'] = df['Value'].pct_change()
        growth_rate = df['Growth_Rate'].mean()
        counter = df['Growth_Rate'].count()
        print(growth_rate)
        for year in range(df.index.max() + 1, df_c.index.max()):
            df_c.loc[year, 'Predicted_Values'] = last_valid_value * (1 + growth_rate)
            last_valid_value = df_c.loc[year, 'Predicted_Values']
            #growth_rate = (growth_rate * counter + (df_c.loc[year, 'Predicted_Values'] / df_c.loc[year - 1, 'Predicted_Values'])) / (counter + 1)
            print(growth_rate)
        print(df_c['Predicted_Values'])
        df_c.Value = np.exp(df_c.Value)
        df_c.Predicted_Values = np.exp(df_c.Predicted_Values)

        # Wyświetlenie wyników
        plt.figure(figsize=(20, 6))
        plt.plot(df_c.index, df_c['Value'], label='Oryginalne wartości')
        plt.plot(df_c.index, df_c['Predicted_Values'], label='Prognozy ARIMA', linestyle='--')
        print(df_c.Predicted_Values)
        #plt.xlim(df.index.min(), df_c.index.max() + n_lat)
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
