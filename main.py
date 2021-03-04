import pandas as pd
import matplotlib.pyplot as plt
import FinancialAnalysis


def main():
    prices = pd.read_csv('DataBasePython - Price.csv', index_col="Date", parse_dates=True)
    categories = pd.read_csv('DataBasePython-Names.csv', index_col='Código')
    fa = FinancialAnalysis.StockAnalysis(prices, categories)

    #fa.getStockDataByCategory(setor='Financeiro', subsetor='Exploração de Imóveis',
                              #segmento='Exploração de Imóveis').plot()
    #fa.getStockDataByCategory(setor='Utilidade Pública', subsetor='Energia Elétrica', segmento='Energia Elétrica').plot()
    #fa.getStockDataByCategory(setor='Utilidade Pública').head()
   # fa.getStockDataByCategory(setor='Bens Industriais').plot()
    #buttonPressed = False
    #while (not buttonPressed):
    #    buttonPressed = plt.waitforbuttonpress()
    #exit(1)
    # print(fa.getSetores())
    # print(fa.getSubsetores())
    # print(fa.getSegmentos())
    # print(fa.getStocksByCategory(setor='Bens Industriais'))
    #fa.plotAllMarketByCategory(initialAmount=1,startDate='2021-01-01')

    #fa.getStockDataByCategory(setor='Saúde', subsetor='Análises e Diagnósticos', startDate='2021-01').plot()
    #mktAvg = fa.getMarketAvgData(startDate='2021')
    #mktAvg = fa.getMarketAvgData(startDate='2021', setores='Consumo não Cíclico')

    setor = 'Consumo não Cíclico'
    #subsetor = ''

    #keywordSetor = (fa.stockCategories.loc[:, 'SETOR']==setor)



    #keywordSubsetor = (fa.stockCategories.loc[:, 'SUBSETOR'])


    #keywordSegmento = (fa.stockCategories.loc[:, 'SEGMENTO'])


    #keyword = keywordSetor & keywordSubsetor & keywordSegmento


    #stocksByCat = fa.stockCategories[keywordSetor & keywordSubsetor & keywordSegmento].index

    #print(stocksByCat)
    #exit(1)
    #teste=fa.getStockDataByCategory(subsetor='Bebidas',startDate='2021')
    #mktAvg=fa.plotMktAvg(startDate='2021',setores='Consumo não Cíclico')
    #fa.plotAllMarketByCategory(startDate='2021')
    #fa.plotMktAvgBySetor(startDate='2021',folder_path='Averages')
    #print(teste.head())
    #print(mktAvg.head())
    #exit(1)
    #print(fa.getSetores())
    #mktAvg.plot()
    #fa.plotMktAvg(startDate='2021')
    #exit(1)
    #buttonPressed = False
    #while (not buttonPressed):
    #    buttonPressed = plt.waitforbuttonpress()


    #fa.plotMktAvgBySegmento(startDate='2021',folder_path='Averages',segmento='Comércio')

    stocksByCat = fa.getStocksByCategory(setor = 'Bens Industriais')
    #stocksByCat = fa.stockCategories[(fa.stockCategories.loc[:, 'SETOR'] == 'Bens industriais')].index
    #stocksByCat = fa.stockCategories[(fa.stockCategories.loc[:, 'SUBSETOR'] == 'Comércio')].index
    print(stocksByCat)
        #if i > 20:
            #break
    #plt.show()


if __name__ == '__main__':
    main()

# exec(open('E:\Programacao\Python\FinancialAnalysis\FinancialAnalysis.py').read())
