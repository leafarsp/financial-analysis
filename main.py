import pandas as pd
import matplotlib.pyplot as plt
import FinancialAnalysis


def main():
    prices = pd.read_csv('testDataBases\\DataBasePython - Price.csv', index_col="Date", parse_dates=True)
    categories = pd.read_csv('testDataBases\\DataBasePython-Names.csv', index_col='Código')
    fa = FinancialAnalysis.StockAnalysis(prices, categories)

    #norm_prices = fa.getNormalizedPrices(startDate='2021')
    #print(norm_prices)
    #exit(1)
    #fa.plotAllMarketByCategory(startDate='2021')
    #fa.plotAllTopStocks(start_date='2021')

    fa.plotHistogram('WEGE3',start_date='2020',end_date='2021')
    #returns=fa.getReturns()
    #returns['WEGE3'][pd.Timestamp('2020-01-01'):pd.Timestamp('2020-12-31')].hist(bins=100)
    #returns.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2020-12-31'),'WEGE3'].hist(bins=100)

    buttonPressed = False
    while (not buttonPressed):
        buttonPressed = plt.waitforbuttonpress()




if __name__ == '__main__':
    main()

# exec(open('E:\Programacao\Python\FinancialAnalysis\FinancialAnalysis.py').read())
