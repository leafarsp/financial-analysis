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
    fa.plotAllMarketByCategory(startDate='2021')
    fa.plotAllTopStocks(start_date='2021')

    buttonPressed = False
    while (not buttonPressed):
        buttonPressed = plt.waitforbuttonpress()




if __name__ == '__main__':
    main()

# exec(open('E:\Programacao\Python\FinancialAnalysis\FinancialAnalysis.py').read())
