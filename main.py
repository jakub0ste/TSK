import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
#from pmdarima import auto_arima


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
    def __int__(self):
        pass


def main():
    kalkulator = KalkulatorPKB()
    # wartości głównie przypadkowe
    pkb_dochodowa = kalkulator.pkb_metoda_dochodowa(100801, 457823, 574000, 50000)

    pkb_wydatkowa = kalkulator.pkb_metoda_wydatkowa(140000, 60000, 40000, 30000, 20000)

    pkb_produkcja = kalkulator.pkb_metoda_produkcji(1200000, 40000)

    print(f"PKB metodą dochodową: {pkb_dochodowa} mln zł")
    print(f"PKB metodą wydatkową: {pkb_wydatkowa} mln zł")
    print(f"PKB metodą produkcyjną: {pkb_produkcja} mln zł")

    data = {
        'Date': pd.date_range(start='2000-01-01', periods=80, freq='Q'),
        'GDP': np.random.normal(5000, 500, 80)
    }

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['GDP'], label='PKB', color='blue')
    plt.title('Historyczny PKB')
    plt.xlabel('Data')
    plt.ylabel('PKB')
    plt.legend()
    plt.show()

    train = df.iloc[:-8]
    test = df.iloc[-8:]
    # stepwise_model = auto_arima(train['GDP'], start_p=1, start_q=1,
    #                             max_p=3, max_q=3, m=4,
    #                             start_P=0, seasonal=False,
    #                             d=1, trace=True,
    #                             error_action='ignore',
    #                             suppress_warnings=True,
    #                             stepwise=True)
    # print(stepwise_model.summary())
    # model = ARIMA(train['GDP'], order=stepwise_model.order)
    # model_fit = model.fit()
    # forecast = model_fit.forecast(steps=8)
    # test['Forecast'] = forecast
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train['GDP'], label='Dane uczące')
    plt.plot(test.index, test['GDP'], label='Dane rzeczywiste')
    #plt.plot(test.index, test['Forecast'], label='Prognoza', color='red')
    #plt.fill_between(test.index, test['Forecast'] - 1.96 * model_fit.bse,
    #                 test['Forecast'] + 1.96 * model_fit.bse, color='red', alpha=0.2)
    plt.title('Prognoza PKB za pomocą ARIMA')
    plt.xlabel('Data')
    plt.ylabel('PKB')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
