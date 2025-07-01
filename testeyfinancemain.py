import yfinanceLib as yfinlib
import pickle
import os
import pandas as pd


def save_yfinance_data(yfinance_obj, path):
    # Caminho do arquivo onde o objeto será salvo
    # Salvando o objeto no disco
    with open(path, 'wb') as arquivo:
        pickle.dump(yfinance_obj, arquivo)

def load_yfinance_data(path):
    # Carregando o objeto do disco
    objeto_carregado = None
    with open(path, 'rb') as arquivo:
        objeto_carregado = pickle.load(arquivo)
    return objeto_carregado



def main():
    tickers_list = yfinlib.loadTickers('./data/Tickers2.xlsx', './data/NotValidTickers.csv')
    tickers_all = yfinlib.loadAllTickers('./data/Tickers2.xlsx')

    #df_prices = yfinlib.fetch_prices(tickers_list)
    #save_yfinance_data(df_prices,"./data/df_prices.obj")
    df_prices = load_yfinance_data("./data/df_prices.obj")
    print(df_prices.head())

    #df_all_stock_param = yfinlib.transpose_stock_data_all(df_prices)
    
    #save_yfinance_data(df_all_stock_param,"./data/df_all_stock_param.obj")
    df_all_stock_param = load_yfinance_data("./data/df_all_stock_param.obj")
    print(df_all_stock_param.head())
    
    df_avg_ret = yfinlib.get_avg_daily_monthly_returns(df_all_stock_param)
    print(df_avg_ret.head())
    df_avg_ret.to_excel(f'./data/df_avg_ret.xlsx')


    
    pltSectors(tickers_all, df_all_stock_param)


def pltSectors(df_tickers, df_stock_data):

    
    sectors = df_tickers['SETOR ECONÔMICO'].unique().tolist()
    subsectors = df_tickers['SUBSETOR'].unique().tolist()

    

    df_columns = ['sector','subsector','segment','avg_daily_ret','avg_monthly_ret']

    df_avg_daily_monthly_returns = pd.DataFrame(columns=df_columns)

    for sector in sectors:
        df_test = yfinlib.get_tickers_by_sector_subsector_segment(df_tickers, sector = sector)
        df_avg_returns_selected = yfinlib.get_avg_returns_by_tickers(df_stock_data, df_test)
        sector_avg_daily_ret = yfinlib.calculate_avg_daily_return(df_avg_returns_selected['avg_return'])
        sector_avg_monthly_ret = yfinlib.calculate_avg_monthly_return(df_avg_returns_selected['avg_return'])
        
        data_line = [sector,'','',sector_avg_daily_ret,sector_avg_monthly_ret] 
        

        # Adiciona a linha no final
        df_avg_daily_monthly_returns.loc[len(df_avg_daily_monthly_returns)] = data_line
      
        
        # Exemplo de uso com o dataframe de retorno médio selecionado anteriormente:
        df_integrated_avg_returns = yfinlib.integrate_returns(df_avg_returns_selected)
        #print(df_avg_returns_selected.head())
        sectordir = f'./data/plots/{sector}'
            # Cria o diretório se ele não existir
        if sectordir and not os.path.exists(sectordir):
            os.makedirs(sectordir)

        yfinlib.plot_returns(df_integrated_avg_returns,title=sector,savepath=f'{sectordir}/{sector}.png',showplt=False)

        for subsector in subsectors:
            df_test = yfinlib.get_tickers_by_sector_subsector_segment(df_tickers, sector = sector, subsector = subsector)
            if df_test is not None:
                try:
                    df_avg_returns_selected = yfinlib.get_avg_returns_by_tickers(df_stock_data, df_test)

                    subsector_avg_daily_ret = yfinlib.calculate_avg_daily_return(df_avg_returns_selected['avg_return'])
                    subsector_avg_monthly_ret = yfinlib.calculate_avg_monthly_return(df_avg_returns_selected['avg_return'])
                    data_line = [sector,subsector,'',subsector_avg_daily_ret,subsector_avg_monthly_ret] 
        
                    # Adiciona a linha no final
                    df_avg_daily_monthly_returns.loc[len(df_avg_daily_monthly_returns)] = data_line

                    # Exemplo de uso com o dataframe de retorno médio selecionado anteriormente:
                    df_integrated_avg_returns = yfinlib.integrate_returns(df_avg_returns_selected)
                    #print(df_avg_returns_selected.head())
                    subsectordir = f'./data/plots/{sector}'
                    # Cria o diretório se ele não existir
                    if subsectordir and not os.path.exists(subsectordir):
                        os.makedirs(subsectordir)

                    yfinlib.plot_returns(df_integrated_avg_returns,title=subsector,savepath=f'{subsectordir}/{subsector}.png',showplt=False)
                except Exception as e:
                    if str(e) != 'Nenhum dos tickers fornecidos está presente no DataFrame de retornos.':
                        print(f'Error in {subsector}, error: {e}')
    print(df_avg_daily_monthly_returns.head())
    df_avg_daily_monthly_returns.to_excel(f'./data/df_avg_daily_monthly_returns.xlsx')
if __name__ == '__main__':
    main()

