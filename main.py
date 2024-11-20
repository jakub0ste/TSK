import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

import itertools
from statsmodels.tsa.arima.model import ARIMA

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

        df.dropna(inplace=True)
        X = df.iloc[:, 1:].values
        y = df['Value'].values

        model = CustomLinearRegression()
        model.fit(X, y)
        return model

    def MA(self, residuals):
        res_df = pd.DataFrame(residuals, columns=['Residuals'])
        for i in range(1, self.q + 1):
            res_df[f'Lags_{i}'] = res_df['Residuals'].shift(i)

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
                padding = [last_values[0]] * (self.p - len(last_values))
                last_values = padding + last_values
            else:
                last_values = last_values[-self.p:]
            next_pred = ar_model.predict(np.array(last_values).reshape(1, -1))
            forecast.append(next_pred[0])
            last_values.append(next_pred[0])

        forecast = np.concatenate(([reverted_predictions[-1]], forecast)).cumsum()
        return [reverted_predictions[-1]], forecast

    def calculate_overlap(self, data, forecast_steps=5):
        result = []
        for i in range(self.p + self.d + 1, len(data)):
            diff_data = difference(data['Value'].values[:i], self.d)
            if len(diff_data) <= self.p:
                raise ValueError(f"Insufficient data of len {len(diff_data)} for {self.p} AR lags after differencing.")

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

class CustomLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        X_transpose = X.T
        U, s, Vt = np.linalg.svd(X_transpose @ X)
        epsilon = 1e-10
        S_inv = np.diag(1 / (s + epsilon))
        XTX_inv = Vt.T @ S_inv @ U.T
        XTy = X_transpose @ y
        params = XTX_inv @ XTy

        self.intercept = params[0]
        self.coefficients = params[1:]

    def predict(self, X):
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not fitted yet. Call 'fit' with training data first.")
        X = np.c_[np.ones(X.shape[0]), X]
        predictions = X @ np.concatenate(([self.intercept], self.coefficients))
        return predictions

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

def plot_acf_pacf(diff_series, max_lags):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_acf(diff_series, lags=max_lags, ax=axes[0])
    axes[0].set_title('ACF')
    plot_pacf(diff_series, lags=max_lags, ax=axes[1])
    axes[1].set_title('PACF')
    plt.tight_layout()
    return fig

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
    return file_path

def difference(data, d):
    diff_data = data.copy()
    for _ in range(d):
        diff_data = np.diff(diff_data)
    return pd.DataFrame(diff_data, columns=['Value'])

def calculate_forecast(p, q, df):
    p = int(p)
    q = int(q)
    df_Values = df['Value']
    d = find_d(df_Values.dropna())
    arima_model = PredykcjaPKB(p, d, q)
    predictions, forecast = arima_model.calculate(df)
    overlap = arima_model.calculate_overlap(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    forecast_years = np.arange(df.index[-1] + 1, df.index[-1] + 1 + len(forecast))
    combined_years = np.concatenate([df.index, forecast_years])
    combined_values = np.concatenate([df['Value'].values, forecast])

    test_index = np.arange(df.index[-1], df.index[-1] + 1 + len(predictions))
    testList = list()
    testList.append(df['Value'].iloc[-1])
    testList.append(predictions[-1])
    test_values = pd.Series(testList, index=test_index[0:])

    ax.plot(combined_years, combined_values, label='Combined Forecasts', color='red')
    ax.plot(df.index, df['Value'], label='Original Values')
    ax.plot(test_index, test_values, linestyle='-.')

    ax.plot(forecast_years, forecast, label='ARIMA Forecasts', linestyle=':')
    overlap_index = np.arange(df.index[arima_model.p + arima_model.d + 1],
                              df.index[arima_model.p + arima_model.d + 1] + len(overlap))
    ax.plot(overlap_index, overlap, label='ARIMA Overlap Forecasts', linestyle='--')
    ax.grid()
    ax.legend()
    ax.set_title("Comparison of Original Values and ARIMA Forecasts")
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP (in million PLN)")

    # Add points for each forecasted value
    scatter = ax.scatter(forecast_years, forecast, color='red')

    # Use mplcursors to display annotations on hover
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f'{forecast_years[sel.index]}: {forecast[sel.index]:.2f} zł'))

    for widget in frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Display predicted values in a table
    tree = ttk.Treeview(frame, columns=('Year', 'Value'), show='headings')
    tree.heading('Year', text='Year')
    tree.heading('Value', text='Value (zł)')
    for year, value in zip(forecast_years, forecast):
        tree.insert('', 'end', values=(year, f'{value:.2f} zł'))
    tree.pack(side=tk.BOTTOM, pady=10)

def auto_select_params(df):
    best_aic = float('inf')
    best_params = (0, 0)
    d = find_d(df['Value'].dropna())
    with open('params_aic_log.txt', 'w') as file:
        file.write("p, q, AIC\n")
        for p, q in itertools.product(range(6), repeat=2):
            try:
                model = ARIMA(df['Value'], order=(p, d, q))
                results = model.fit()
                aic = results.aic
                file.write(f"{p}, {q}, {aic}\n")
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, q)
            except:
                continue
    return best_params

def on_file_select():
    def validate_input(new_value):
        return new_value.isdigit() and 0 <= int(new_value) <= 9

    def check_entries(*args):
        if p_entry.get() and q_entry.get():
            calculate_button.config(state=tk.NORMAL)
        else:
            calculate_button.config(state=tk.DISABLED)

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
        diff_series = np.diff(df_Values.dropna(), n=d)
        max_lags = min(20, len(diff_series) // 2)
        fig = plot_acf_pacf(diff_series, max_lags)

        for widget in frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        vcmd = (frame.register(validate_input), '%P')
        p_label = ttk.Label(frame, text="Enter p:")
        p_label.pack(side=tk.LEFT, padx=5)
        p_entry = ttk.Entry(frame, validate='key', validatecommand=vcmd)
        p_entry.pack(side=tk.LEFT, padx=5)
        p_entry.bind('<KeyRelease>', check_entries)

        q_label = ttk.Label(frame, text="Enter q:")
        q_label.pack(side=tk.LEFT, padx=5)
        q_entry = ttk.Entry(frame, validate='key', validatecommand=vcmd)
        q_entry.pack(side=tk.LEFT, padx=5)
        q_entry.bind('<KeyRelease>', check_entries)

        calculate_button = ttk.Button(frame, text="Calculate", command=lambda: calculate_forecast(p_entry.get(), q_entry.get(), df))
        calculate_button.pack(side=tk.LEFT, padx=5)
        calculate_button.config(state=tk.DISABLED)

        auto_button = ttk.Button(frame, text="Auto", command=lambda: auto_select_and_calculate(df))
        auto_button.pack(side=tk.LEFT, padx=5)

def auto_select_and_calculate(df):
    p, q = auto_select_params(df)
    print(f"Selected parameters: p = {p}, q = {q}")
    calculate_forecast(p, q, df)

def main():
    window = tk.Tk()
    window.title("ACF and PACF Plotter")
    window.geometry("1920x900")

    global frame
    frame = ttk.Frame(window, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)

    button = ttk.Button(frame, text="Select File", command=on_file_select)
    button.pack(side=tk.TOP, pady=10)

    window.mainloop()

if __name__ == '__main__':
    main()