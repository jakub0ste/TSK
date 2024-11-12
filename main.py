import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

        model = CustomLinearRegression()
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

        model = CustomLinearRegression()
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

        initial_value = data['Value'].iloc[-1]
        reverted_predictions = np.concatenate(([initial_value], predictions)).cumsum()

        forecast = []
        last_values = list(diff_data)
        last_values.append(predictions[-1])
        for _ in range(forecast_steps):
            last_values = np.nan_to_num(pd.to_numeric(last_values, errors='coerce').astype(float)).tolist()
            if len(last_values) < self.p:
                padding = [last_values[0]] * (self.p - len(last_values))  # Repeat the first value as padding
                last_values = padding + last_values  # Extend the list
            else:
                last_values = last_values[-self.p:]
            next_pred = ar_model.predict(np.array(last_values).reshape(1, -1))
            forecast.append(next_pred[0])
            last_values.append(next_pred[0])

        forecast = np.concatenate(([reverted_predictions[-1]], forecast)).cumsum()

        return [reverted_predictions[-1]], forecast

    def calculate_overlap(self, data, forecast_steps=5):
        result =[]
        for i in range(self.p + self.d + 1, len(data)):
            diff_data = difference(data['Value'].values[:i], self.d)

            if len(diff_data) <= self.p:
                raise ValueError(f"Insufficient data of len {len(diff_data)} for {self.p} AR lags after differencing.")

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
            initial_value = data['Value'].iloc[i]
            reverted_predictions = np.concatenate(([initial_value], predictions)).cumsum()
            result.append(reverted_predictions[-1])
        return result

    # def AR_overlap(self, df, n_przewidywan=5):
    #     df_temp = df.copy()
    #
    #     # Generating the lagged p terms
    #     for i in range(1, self.p + 1):
    #         df_temp[f'Shifted_values_{i}'] = df_temp['Value'].shift(i)
    #
    #     df_temp = df_temp.ffill()
    #
    #     train_size = int(0.8 * df_temp.shape[0])
    #
    #     # Breaking data set into training and testing
    #     df_train = df_temp.iloc[:train_size].dropna()
    #     df_test = df_temp.iloc[train_size:].copy()
    #
    #     # X and Y preparation
    #     X_train = df_train.iloc[:, 1:self.p + 1].values
    #     y_train = df_train.iloc[:, 0].values
    #
    #     # Train the model
    #     lr = CustomLinearRegression()
    #     lr.fit(X_train, y_train)
    #
    #     # Coefficients and intercept
    #     theta = lr.coefficients
    #     intercept = lr.intercept
    #
    #     # Calculate predictions on training data
    #     df_train['Predicted_Values'] = X_train.dot(theta) + intercept
    #
    #     # Test set predictions
    #     X_test = df_test.iloc[:, 1:self.p + 1].values
    #     df_test['Predicted_Values'] = X_test.dot(theta) + intercept
    #
    #     # Future predictions
    #     last_values = df['Value'].iloc[-self.p:].values
    #     future_predictions = []
    #     for _ in range(n_przewidywan):
    #         future_pred = lr.predict(last_values.reshape(1, -1))[0]
    #         future_predictions.append(future_pred)
    #
    #         # Update last_values for next prediction
    #         last_values = np.roll(last_values, -1)
    #         last_values[-1] = future_pred
    #
    #     # Offset and future DataFrame
    #     offset = df_test.index.max() - df.shape[0] + 1
    #     df_future = pd.DataFrame({'Predicted_Values': future_predictions},
    #                              index=range(df.shape[0] + offset, df.shape[0] + n_przewidywan + offset))
    #
    #     return [df_train, df_test, df_future]
    #
    # def MA_overlap(self, res):
    #     # Generating the lagged q terms
    #     for i in range(1, self.q + 1):
    #         res[f'Shifted_values_{i}'] = res['Residuals'].shift(i)
    #
    #     train_size = int(0.8 * res.shape[0])
    #
    #     # Split data into training and testing sets
    #     res_train = res.iloc[:train_size].dropna().copy()
    #     res_test = res.iloc[train_size:].copy()
    #
    #     # Prepare X and y for training
    #     X_train = res_train.iloc[:, 1:self.q + 1].values
    #     y_train = res_train.iloc[:, 0].values
    #
    #     # Train the model
    #     lr = CustomLinearRegression()
    #     lr.fit(X_train, y_train)
    #
    #     # Extract coefficients and intercept
    #     theta = lr.coefficients  # Should be shape (self.q,)
    #     intercept = lr.intercept
    #
    #     # Training predictions
    #     res_train['Predicted_Values'] = X_train.dot(theta) + intercept
    #
    #     # Testing predictions
    #     X_test = res_test.iloc[:, 1:self.q + 1].values
    #     res_test['Predicted_Values'] = X_test.dot(theta) + intercept
    #
    #     return [res_train, res_test]


class CustomLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add a column of ones to X to account for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]  # Shape (n_samples, n_features + 1)

        # Calculate coefficients using the normal equation: (X^T * X)^-1 * X^T * y
        X_transpose = X.T
        # XTX_inv = np.linalg.pinv(X_transpose @ X)
        U, s, Vt = np.linalg.svd(X_transpose @ X)
        epsilon = 1e-10
        S_inv = np.diag(1 / (s+epsilon))
        XTX_inv = Vt.T @ S_inv @ U.T
        XTy = X_transpose @ y
        params = XTX_inv @ XTy  # Calculate (n_features + 1,) array of params

        # Separate intercept and coefficients
        self.intercept = params[0]
        self.coefficients = params[1:]

    def predict(self, X):
        # Check if the model is fitted
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with training data first.")

        # Add a column of ones to X to account for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # Predict: y = X @ params, where params includes intercept and coefficients
        predictions = X @ np.concatenate(([self.intercept], self.coefficients))
        return predictions

    def __str__(self):
        return f"CustomLinearRegression(intercept={self.intercept}, coefficients={self.coefficients})"


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
        overlap = arima_model.calculate_overlap(df)
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

        plt.plot(combined_years, combined_values, label='Połączone prognozy', color='red')
        plt.plot(df.index, df['Value'], label='Oryginalne wartości')
        #plt.plot(test_index[-1], predictions, label='Prognozy ARIMA', linestyle='-.')
        plt.plot(test_index, test_values, linestyle='-.')

        plt.plot(forecast_years, forecast, label='Prognozy ARIMA', linestyle=':')
        overlap_index = np.arange(df.index[arima_model.p + arima_model.d + 1], df.index[arima_model.p + arima_model.d + 1] + len(overlap))
        plt.plot(overlap_index, overlap, label='Prognozy ARIMA', linestyle='--')
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
