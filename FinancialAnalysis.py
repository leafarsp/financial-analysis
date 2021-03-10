import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

#08/03/2021

# from google.colab import files

class StockAnalysis:
    '''
    @param self:
    @param priceData: pandas.core.frame.DataFrame
    @param stocksCategories: pandas.core.frame.DataFrame
    @return: None
    '''

    def __init__(self, priceData, stocksCategories):
        self.price = priceData
        self.stockCategories = stocksCategories
        self.returns = None
        self.tempNormPrices=None
        self.tempGrouth = None

    def getStockList(self):
        return self.price.columns

    def plotStocks(self, stock, initialDateYYYY_MM_DD='2016-01-01', \
                   finalDateYYYY_MM_DD='today'):
        if finalDateYYYY_MM_DD == 'today':
            finalDateYYYY_MM_DD = pd.to_datetime("today")
        self.__getStockByDate(stock, initialDateYYYY_MM_DD, \
                              finalDateYYYY_MM_DD).plot()

    def __getStockByDate(self, stock, initialDateYYYY_MM_DD='2016-01-01', \
                         finalDateYYYY_MM_DD='today'):
        if finalDateYYYY_MM_DD == 'today':
            finalDateYYYY_MM_DD = pd.to_datetime("today")
        tempstock = self.price[stock].dropna(how='all')
        tempstock[pd.Timestamp(initialDateYYYY_MM_DD): \
                  pd.Timestamp(finalDateYYYY_MM_DD)]
        return tempstock

    def plotStockByCategory(self, initialDateYYYY_MM_DD='2016-01-01', \
                            finalDateYYYY_MM_DD='today', stock='all', \
                            setores='all', subsetores='all', segmentos='all'):
        pass

    def getSegmentos(self):
        return self.stockCategories['SEGMENTO'].unique()

    def getSetores(self):
        self.stockCategories['SETOR'].head()
        return self.stockCategories['SETOR'].unique()

    def getSubsetores(self):
        return self.stockCategories['SUBSETOR'].unique()

    def getStockList(self):
        return self.getPrice().columns

    def getStocksByCategory(self, setor='all', subsetor='all', segmento='all'):
        # print('Getting stocks by categories')
        # print(f'Setor = {setor}, Subsetor = {subsetor}, Segmento = {segmento}')
        #TODO: Melhorar essa função, se o keywordSetor for igual a 'all', não dá certo
        """
        if setor == 'all':
            keywordSetor = (self.stockCategories.loc[:, 'SETOR'])
        else:
            keywordSetor = (self.stockCategories.loc[:, 'SETOR'] == setor)

        if subsetor == 'all':
            keywordSubsetor = (self.stockCategories.loc[:, 'SUBSETOR'])
        else:
            keywordSubsetor = (self.stockCategories.loc[:, 'SUBSETOR'] == subsetor)

        if (segmento == 'all'):
            keywordSegmento = (self.stockCategories.loc[:, 'SEGMENTO'])
        else:
            keywordSegmento = (self.stockCategories.loc[:, 'SEGMENTO'] == segmento)
        # stocksByCat = self.stockCategories[keyword].index
        """

        if setor == 'all':
            if subsetor == 'all':
                if (segmento == 'all'):
                    stocksByCat = self.stockCategories.index

                else:
                    stocksByCat = self.stockCategories[(self.stockCategories.loc[:, 'SEGMENTO'] == segmento)].index
            else:
                if (segmento == 'all'):
                    stocksByCat = self.stockCategories[(self.stockCategories.loc[:, 'SUBSETOR'] == subsetor)].index
                else:
                    stocksByCat = self.stockCategories[(self.stockCategories.loc[:, 'SUBSETOR'] == subsetor) & \
                                                       (self.stockCategories.loc[:, 'SEGMENTO'] == segmento)].index
        else:
            if subsetor == 'all':
                if (segmento == 'all'):
                    stocksByCat = self.stockCategories[(self.stockCategories.loc[:, 'SETOR'] == setor)].index
                else:
                    stocksByCat = self.stockCategories[(self.stockCategories.loc[:, 'SETOR'] == setor) & \
                                                       (self.stockCategories.loc[:, 'SEGMENTO'] == segmento)].index
            else:
                if (segmento == 'all'):
                    stocksByCat = self.stockCategories[(self.stockCategories.loc[:, 'SETOR'] == setor) & \
                                                       (self.stockCategories.loc[:, 'SUBSETOR'] == subsetor)].index
                else:
                    stocksByCat = self.stockCategories[(self.stockCategories.loc[:, 'SETOR'] == setor) & \
                                                       (self.stockCategories.loc[:, 'SUBSETOR'] == subsetor) & \
                                                       (self.stockCategories.loc[:, 'SEGMENTO'] == segmento)].index

        return stocksByCat

    def getReturns(self):
        if self.returns is None:
            self.__calculateReturns()
        return self.returns

    # local

    def getNormalizedPrices(self, initialAmount=1, stocks='all', startDate='2016-01-01',
                            endDate='today'):
        if endDate == 'today':
            endDate = pd.to_datetime("today")
        # print (f'Getting Normalized Prices, initial Amount: {initialAmount}, start date on {startDate}, end date on {endDate} ')
        if (type(stocks) == str):
            if (stocks == 'all'):
                stocks = self.getStockList()
            else:
                stocks = [stocks]

        temp_returns_Full = self.getReturns()
        temp_returns = temp_returns_Full[stocks][pd.Timestamp(startDate):pd.Timestamp(endDate)]

        temp_norm_prices = pd.DataFrame(data=None, \
                                        index=temp_returns.index, columns=stocks)
        temp_norm_prices.iloc[0] = initialAmount
        for stock in stocks:
            prev_date = temp_norm_prices.iloc[0].name
            # print(prev_date)
            for date in temp_norm_prices.index:
                if pd.Timestamp(date) > pd.Timestamp(temp_norm_prices.iloc[0].name):
                    temp_norm_prices.loc[date, stock] = \
                        temp_norm_prices.loc[prev_date, stock] * (1 + temp_returns.loc[date, stock])
                prev_date = date

        #np.polyfit(xspace,y,1)[0] -- ax+b, esse seria o "a"
        growths = self.__calculateGrowth(temp_norm_prices)
        #print(growths.columns)
        #print(temp_norm_prices)
        #temp_norm_prices_organized = temp_norm_prices.reindex(columns=growths.columns)
        temp_norm_prices_organized= temp_norm_prices[growths.columns]
        #print(growths)
        #print(temp_norm_prices_organized.columns)

        return temp_norm_prices_organized

    def __calculateGrowth(self,normalized_prices):
        temp_df = pd.DataFrame(data=None,index = ['growth'], columns=normalized_prices.columns)
        size_prices = normalized_prices.index.size
        xspace = np.linspace(1, size_prices, size_prices)
        for column in normalized_prices.columns:
            yspace = normalized_prices[column].astype('float32').to_numpy()
            temp_df.loc['growth',column] = np.polyfit(x=xspace,y=yspace,deg=1)[0]

        self.tempGrouth = temp_df[temp_df.iloc[-1,:].sort_values(ascending=False).index]
        return self.tempGrouth

    def getPrice(self):
        return self.price

    # local

    def __calculateReturns(self):
        print('Calculating returns')
        tempprice = self.getPrice().dropna(how='all', axis=0)
        temp_returns = pd.DataFrame(data=None, index=tempprice.index, \
                                    columns=self.getStockList())
        for stock in self.getStockList():
            temp_returns[stock] = tempprice[stock].dropna(how='all').pct_change(1)
        self.returns = temp_returns.fillna(0)

    def plotStockList(self, initialAmount=1, stocks='all', startDate='2016-01-01',
                      endDate='today'):
        pass

    def getStockDataByCategory(self, setor='all', subsetor='all', segmento='all', \
                               initialAmount=1, \
                               startDate='2016-01-01', endDate='today'):
        if endDate == 'today':
            endDate = pd.to_datetime("today")

        stockList = self.getStocksByCategory(setor, subsetor, segmento)

        return self.getNormalizedPrices(initialAmount=initialAmount, stocks=stockList, startDate=startDate, \
                                        endDate=endDate)

    def plotAllMarketByCategory(self, folder_path='charts', initialAmount=1, startDate='2016-01-01',
                                endDate='today'):
        if endDate == 'today':
            endDate = pd.to_datetime("today")
        numValidCats = self.__getNumberOfValidSetoresAndSubsetores()
        i = 1
        folder = folder_path
        for setor in self.getSetores():
            for subsetor in self.getSubsetores():
                stocks = None
                stocks = self.getStocksByCategory(setor, subsetor).copy()
                if (stocks.size > 0):
                    print(f'{i}/{numValidCats} - {setor}, {subsetor}')
                    # print(stocks)

                    # plt.subplot(10, 1, i)
                    if stocks.size > 10:
                        validSegmentos = self.__getNumberOfValidSegmentos(setor, subsetor)

                        j = 1
                        for segmento in self.getSegmentos():
                            stocks2 = None
                            stocks2 = self.getStocksByCategory(setor, subsetor, segmento).copy()
                            if stocks2.size > 0:
                                stockData = self.getStockDataByCategory(setor=setor, subsetor=subsetor, \
                                                                        segmento=segmento, \
                                                                        startDate=startDate, \
                                                                        initialAmount=initialAmount,\
                                                                        endDate=endDate)
                                if stocks2.size < 10:


                                    print(
                                        f'{i}/{numValidCats} , {j}/{validSegmentos} - {setor}, {subsetor}, {segmento}' )
                                    # plt.figure()
                                    stockData.plot(title=f'{setor}-{subsetor}-{segmento}')
                                    plt.savefig(f'{folder}\\{setor}-{subsetor}-{segmento}.png')
                                    # plt.show()
                                else:
                                    j2 = 1
                                    k_ant = 0
                                    for k in range(10, stocks2.size, 10):
                                        # plt.subplot(stocks2.size // 10 + 1, 1, j)
                                        print(
                                            f'{i}/{numValidCats} , {j}/{validSegmentos}, {j2} / '
                                            f'{stocks2.size // 10} - {setor}, {subsetor}, {segmento}')
                                        if k + 10 > stocks2.size:

                                            stockData.iloc[:, k_ant:stocks2.size].plot(
                                                title=f'{setor}-{subsetor}-{segmento} {j2} de '
                                                      f'{stocks2.size // 10}').plot(figSize=(30, 10))
                                            plt.savefig(
                                                f'{folder}\\{setor}-{subsetor}-{segmento} - '
                                                f'{j2} de {stocks2.size // 10}.png')

                                            #print(
                                             #   f'k_ant: {k_ant}, k: {stocks2.size}, subplot: {j}/{stocks2.size // 10}')
                                            # plt.show()
                                        else:
                                            #print(f'k_ant: {k_ant}, i: {k}, subplot: {j}/{stocks2.size // 10}')
                                            stockData.iloc[:, k_ant:k].plot(
                                                title=f'{setor}-{subsetor}-{segmento} {j2} de '
                                                      f'{stocks2.size // 10}').plot(figSize=(30, 10))
                                            plt.savefig(
                                                f'{folder}\\{setor}-{subsetor}-{segmento} - '
                                                f'{j2} de {stocks2.size // 10}.png')
                                            # plt.show()

                                        k_ant = k + 1
                                        j2 = j2 + 1
                                    # plt.savefig(f'{folder}\\{setor}-{subsetor}-{segmento}.png')
                                j += 1


                                # plt.savefig

                    else:
                        # plt.figure()
                        self.getStockDataByCategory(setor=setor, subsetor=subsetor, initialAmount=initialAmount,\
                                                    startDate=startDate, \
                                                    endDate=endDate).plot(title=f'{setor}, {subsetor}').plot(figSize=(30, 10))
                        plt.savefig(f'{folder}\\{setor}-{subsetor}.png')
                        # plt.show()
                    i += 1

    def getMarketAvgDataBySetor(self, setores='all', folder_path='charts', initialAmount=1, startDate='2016-01-01', endDate='today'):
        if endDate == 'today':
            endDate = pd.to_datetime("today")
        numValidCats = self.__getNumberOfValidSetoresAndSubsetores()
        i = 1
        folder = folder_path

        #temp_norm_prices = self.getNormalizedPrices(initialAmount=initialAmount, startDate=startDate,
#                                                    endDate=endDate)
        temp_df = self.getPrice().loc[pd.Timestamp(startDate):pd.Timestamp(endDate)]

        #print(df_means.head())
        if setores == 'all':
            df_means = pd.DataFrame(data=None, index=temp_df.index, columns=self.getSetores())
            for setor in self.getSetores():

                stockData = self.getStockDataByCategory(setor=setor, initialAmount=initialAmount, startDate=startDate,
                                                        endDate=endDate)

                df_means[setor] = stockData.mean(axis=1)
        else:
            if type(setores) == str:
                df_columns = self.__getValidSubsetores(setores)

                df_means = pd.DataFrame(data=None, index=temp_df.index, columns=df_columns)
                for subsetor in df_columns:
                    #print(f'{setores}, {subsetor}')
                    stockData = self.getStockDataByCategory(setor=setores, subsetor=subsetor, initialAmount=initialAmount,
                                                            startDate=startDate, endDate=endDate)
                    print(stockData.head())
                    df_means[subsetor] = stockData.mean(axis=1)

        return df_means.dropna(how='all', axis=0)


    def plotMktAvg(self, setores='all', folder_path='charts', initialAmount=1, startDate='2016-01-01', endDate='today'):
        self.getMarketAvgData(setores=setores, folder_path=folder_path,
                              initialAmount=1, startDate=startDate, endDate=endDate).plot(
                                                title=f'Média de {setores}')#, figSize=(30, 10))

    def plotMktAvgBySetor(self, folder_path='charts', initialAmount=1, startDate='2016-01-01', endDate='today'):
        if endDate == 'today':
            endDate = pd.to_datetime("today")
        numValidCats = self.__getNumberOfValidSetoresAndSubsetores()
        i = 1
        folder = folder_path

        # temp_norm_prices = self.getNormalizedPrices(initialAmount=initialAmount, startDate=startDate,
        #                                                    endDate=endDate)
        temp_df = self.getPrice().loc[pd.Timestamp(startDate):pd.Timestamp(endDate)]

        # print(df_means.head())


        for setor in self.getSetores():
            df_columns = self.__getValidSubsetores(setor)
            df_means = pd.DataFrame(data=None, index=temp_df.index, columns=df_columns)
            for subsetor in df_columns:
                stockData = self.getStockDataByCategory(setor=setor, subsetor=subsetor, initialAmount=initialAmount,
                                                        startDate=startDate, endDate=endDate)
                df_means[subsetor] = stockData.mean(axis=1)
            df_means.dropna(how='all', axis=0).plot(title=f'Médias {setor}').plot(figSize=(30, 10))
            plt.savefig(f'{folder}\\Médias {setor}.png')


    def plotMktAvgBySubsetor(self, subsetor = 'all', folder_path='charts', initialAmount=1, startDate='2016-01-01', endDate='today'):
        print("plotting Mkt Avg By Subsetor")
        if endDate == 'today':
            endDate = pd.to_datetime("today")
        numValidCats = self.__getNumberOfValidSetoresAndSubsetores()
        i = 1
        folder = folder_path

        # temp_norm_prices = self.getNormalizedPrices(initialAmount=initialAmount, startDate=startDate,
        #                                                    endDate=endDate)
        temp_df = self.getPrice().loc[pd.Timestamp(startDate):pd.Timestamp(endDate)]

        # print(df_means.head())
        if type(subsetor) == str:
            if subsetor == 'all':
                pass
            else:
                pass


        df_columns = self.__getValidSegmentos(subsetor)

        df_means = pd.DataFrame(data=None, index=temp_df.index, columns=df_columns)

        if type(subsetor) == list:
            for localSubsetor in subsetor:
                for segmento in df_columns:
                    stockData = self.getStockDataByCategory(segmento=segmento, initialAmount=initialAmount,
                                                            startDate=startDate, endDate=endDate)
                    df_means[segmento] = stockData.mean(axis=1)

        else:
            for segmento in df_columns:
                stockData = self.getStockDataByCategory(segmento=segmento, initialAmount=initialAmount,
                                                        startDate=startDate, endDate=endDate)
                df_means[segmento] = stockData.mean(axis=1)

        text_means = str(f'Média {subsetor}').replace("'","").replace("[","").replace("]","")
        df_means[f'{text_means}'] = df_means.mean(axis=1)
        df_means.dropna(how='all', axis=0).plot(title=f'{text_means}').plot(figSize=(30, 10))
        plt.savefig(f'{folder}\\{text_means}.png')

    def plotTopStocks(self, first_position_number=1, qt_stocks=10, start_date='2020-01-01', end_date='today', initial_amount=1):

        #TODO: criar uma função que compara se o tempNormPrices tem as mesmas datas e as mesmas colunas que os parâmetros dessa função
        #caso tenha, não calcular novamente.
        self.tempNormPrices = self.getStockDataByCategory(initialAmount=initial_amount, startDate=start_date, endDate=end_date)



        #self.tempNormPrices = tempNormPrices[tempNormPrices.iloc[-1,:].sort_values().index]
        self.tempNormPrices.iloc[:,(first_position_number-1):(first_position_number+qt_stocks)].plot(title = f'Top ações: '
                                                    f'{first_position_number} até {first_position_number+qt_stocks-1}').plot(figSize=(30, 10))

    def plotAllTopStocks(self,folder_path='charts', start_date='2020-01-01',end_date='today', initial_amount=1):
        self.tempNormPrices = self.getStockDataByCategory(initialAmount=initial_amount, startDate=start_date,
                                                          endDate=end_date)

        temp_qt_col = self.tempGrouth.columns.size
        temp_df_index = list(range(1,temp_qt_col+1,1))
        temp_df = pd.DataFrame(data=self.tempGrouth.copy().transpose())


        temp_df['Position'] = temp_df_index

        # temp_df.append(start_date, ignore_index = False)
        temp_df.to_excel(f'{folder_path}\\stocks_trends.xlsx')

        total_stocks=self.tempNormPrices.columns.size

        for i in range(1,total_stocks,10):
            print(f'Ploting stocks from {i} to {i+10-1}')
            #print(f'{self.tempNormPrices.iloc[:, (i - 1):(i+ 10)].columns}')
            self.tempNormPrices.iloc[:, (i - 1):(i+ 10)].plot(title=f'Top ações: '
                  f'{i} até {i + 10 - 1}').plot(figSize=(30, 10))
            plt.savefig(f'{folder_path}\\Top {i} até {i+10-1}.png')



    def plotHistogram(self):
        pass



    def __auxPlotMkt(self,stockData,numOfValidSegmentos,numOfValidCats,setor,subsetor,segmento,folder,currentCat,
                     currentSegmento):
        print(
            f'{currentCat}/{numOfValidCats} , {currentSegmento}/{numOfValidSegmentos} - '
            f'{setor}, {subsetor}, {segmento}')
        # plt.figure()
        stockData.plot(title=f'{setor}-{subsetor}-{segmento}')
        plt.savefig(f'{folder}\\{setor}-{subsetor}-{segmento}.png')

    def __getNumberOfValidSetoresAndSubsetores(self):
        numValidCats = 0
        for setor in self.getSetores():
            for subsetor in self.getSubsetores():
                stocks = self.getStocksByCategory(setor, subsetor)
                if (stocks.size > 0):
                    # print(f'{i} - {setor}, {subsetor}')
                    # print(stocks)
                    numValidCats += 1
        return numValidCats

    def __getNumberOfValidSegmentos(self, setor, subsetor):
        validSegmentos = 0
        for segmento in self.getSegmentos():
            stocks2 = None
            stocks2 = self.getStocksByCategory(setor, subsetor, segmento).copy()
            if stocks2.size > 0:
                #print(f'{setor}, {subsetor}, {segmento}')
                validSegmentos += 1
        return validSegmentos

    def __getValidSubsetores(self, setores):
        validSetor= []
        if type(setores) == str:
            if setores != 'any':
                for subsetor in self.getSubsetores():
                    stocks = self.getStocksByCategory(setor = setores, subsetor = subsetor)
                    if (stocks.size > 0):
                        validSetor.append(subsetor)
        else:
            for setor in setores:
                for subsetor in self.getSubsetores():
                    stocks = self.getStocksByCategory(setor = setor, subsetor = subsetor)
                    if (stocks.size > 0):
                        validSetor.append(subsetor)

        return validSetor

    def __getValidSegmentos(self, subsetores):
        validSetor= []
        if type(subsetores) == str:
            if subsetores != 'any':
                for segmento in self.getSegmentos():
                    stocks = self.getStocksByCategory(subsetor=subsetores, segmento=segmento)
                    if (stocks.size > 0):
                        validSetor.append(segmento)
        else:
            for subsetor in subsetores:
                for segmento in self.getSegmentos():
                    stocks = self.getStocksByCategory(subsetor=subsetor, segmento=segmento)
                    if (stocks.size > 0):
                        validSetor.append(segmento)

        return validSetor

    #TODO: criar diretório quando ele não existir
    #TODO: salvar um excel com várias planilhas quando plotar todas as ações por categoria
    #TODO: reutilizar o crescimento quando tiver que calcular usando a mesma data de início, de fim e as mesmas ações.
    #TODO: colocar no excel as imagens
    #plotar histogramas
