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
        residuals = diff_data['Value'].values - ar_pred
        if self.q > 0:
            ma_model = self.MA(residuals)
            ma_pred = ma_model.predict(residuals[-self.q:].reshape(1, -1))
        else:
            ma_pred = np.zeros_like(ar_pred)
        predictions = ar_pred + ma_pred

        # reverted_predictions = np.zeros_like(predictions)
        # reverted_predictions[0] = predictions[0] + data['Value'].iloc[-(self.d + 1)]
        initial_value = data['Value'].iloc[-1]  # Last actual observed value
        reverted_predictions = np.concatenate(([initial_value], predictions)).cumsum()

        # for i in range(1, len(predictions)):
        #     reverted_predictions[i] = predictions[i] + reverted_predictions[i - 1]

        forecast = [reverted_predictions[-1] + (i + 1) * np.mean(predictions) for i in range(forecast_steps)]

        # reverted_predictions_rec = diff_data.copy()
        # forecast = []
        # for _ in range(forecast_steps):
        #     forecast_step = self.recursive_forecast(reverted_predictions_rec['Value'].values)
        #
        #     # Append the forecasted step to the overall forecast and use it for the next step
        #     forecast.append(forecast_step[0])
        #     reverted_predictions_rec.loc[reverted_predictions_rec.index.max()+1, 'Value'] = forecast_step[0]

        # forecast = []
        # last_values = diff_data
        # last_values = last_values._append(pd.Series(predictions[-1]), ignore_index=True)
        # for _ in range(forecast_steps):
        #     next_pred = ar_model.predict(last_values[-self.p:].values.reshape(1, -1))
        #     forecast.append(next_pred[0])
        #     last_values = last_values._append(next_pred[0])  # Update for next iteration
        #
        # forecast = np.concatenate(([reverted_predictions[-1]], forecast)).cumsum()

        return [reverted_predictions[-1]], forecast

    def recursive_forecast(self, data_in):
        data = data_in
        ar_model = self.AR(data)
        last_lags = data[-self.p:].reshape(1, -1)
        ar_pred = ar_model.predict(last_lags)
        residuals = data - ar_pred
        if self.q > 0:
            ma_model = self.MA(residuals)
            last_q_lags = residuals[-self.q:] if self.q > 1 else [residuals[-1]]
            last_q_lags = np.array(last_q_lags).reshape(1, -1)
            ma_pred = ma_model.predict(last_q_lags)
        else:
            ma_pred = 0
        predictions = ar_pred + ma_pred

        reverted_predictions = np.zeros_like(predictions)
        reverted_predictions[0] = predictions[0] + data[-(self.d + 1)]

        for i in range(1, len(predictions)):
            reverted_predictions[i] = predictions[i] + reverted_predictions[i - 1]

        return reverted_predictions


    def AR_overlap(self, df, n_przewidywan = 5):

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

    def MA_overlap(self, res):

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
        df_testing = pd.DataFrame(np.log(df.Value).diff().diff(1))
        d = find_d(df_Values.dropna())
        p, q = find_p_q(df_Values.dropna(), d)
        arima_model = PredykcjaPKB(p, d, q)
        predictions, forecast = arima_model.calculate(df)
        #print(df)
        # print(predictions)
        # print(forecast)
        n_lat = 5
        [df_train, df_test, df_future] = arima_model.AR_overlap(pd.DataFrame(df_testing.Value), n_lat)
        df_c = pd.concat([df_train, df_test])
        res = pd.DataFrame()
        res['Residuals'] = df_c['Value'] - df_c['Predicted_Values']
        [res_train, res_test] = arima_model.MA_overlap(pd.DataFrame(res.Residuals))
        res_c = pd.concat([res_train, res_test])
        if 'Predicted_Values' in res_c.columns:
            df_c['Predicted_Values'] = df_c['Predicted_Values'].add(res_c['Predicted_Values'], fill_value=0)

        df_c.Value += np.log(df).shift(1).Value
        df_c.Value += np.log(df).diff().shift(1).Value
        df_c.Predicted_Values += np.log(df).shift(1).Value
        df_c.Predicted_Values += np.log(df).diff().shift(1).Value

        df_c.Value = np.exp(df_c.Value)
        df_c.Predicted_Values = np.exp(df_c.Predicted_Values)
        # Wyświetlenie wyników
        plt.figure(figsize=(20, 6))
        forecast_years = np.arange(df.index[-1] + 1, df.index[-1] + 1 + len(forecast))
        combined_years = np.concatenate([df.index, forecast_years])
        combined_values = np.concatenate([df['Value'].values, forecast])

        test_index = np.arange(df.index[-1], df.index[-1] + 1 + len(predictions))
        testList = list()
        testList.append(df['Value'].iloc[-1])
        testList.append(predictions[-1])
        test_values = pd.Series(testList, index=test_index[0:])
        print(test_values)
        print(df['Value'].iloc[-1])
        print(predictions)
        print(forecast)

        #plt.plot(combined_years, combined_values, label='Połączone prognozy', color='red')
        plt.plot(df.index, df['Value'], label='Oryginalne wartości')
        #plt.plot(test_index[-1], predictions, label='Prognozy ARIMA', linestyle='-.')
        plt.plot(test_index, test_values, linestyle='-.')

        #plt.plot(forecast_years, forecast, label='Prognozy ARIMA', linestyle=':')
        plt.plot(df_c.index, df_c['Predicted_Values'], label='Prognozy ARIMA', linestyle='--')
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
