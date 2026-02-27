import yfinance as yf
#pip inimport requests
import pandas as pd
import matplotlib.pyplot as plt
from yahooquery import Ticker
import numpy as np
from io import StringIO
from scipy.stats import norm, skew, kurtosis, gaussian_kde
from io import StringIO


def buscar_dados_acao(ticker,period='1y'):
    dados = None
    try:
        acao = yf.Ticker(ticker)
        dados = acao.history(period=period)

        if dados.empty:
            print("Nenhum dado encontrado para o ticker fornecido.")


        # print(f"Dados da ação {ticker}:")
        # print(dados[['Open', 'High', 'Low', 'Close', 'Volume']])
        # plt.plot(dados['Close'])
    except Exception as e:
        print(f"Erro ao buscar dados: {e}")
    return dados


def loadTickers(tickers_xlsx, notValidTickers = None):
  tickersdf = pd.read_excel(tickers_xlsx)

  NotValidTickersdf = pd.read_csv(notValidTickers)
  if notValidTickers is not None:
    tickersdf = tickersdf[~tickersdf["Ticker"].isin(NotValidTickersdf["Not Valid Ticker"])]

  tickersdf['Ticker']=tickersdf['Ticker']+".SA"
  tickersdf = tickersdf.dropna()
  ticker_list = tickersdf['Ticker'].to_list()

  return ticker_list, tickersdf

def loadAllTickers(tickers_list_xls):
  tickersdf = pd.read_excel(tickers_list_xls)
  #
  tickersdf['Ticker']=tickersdf['Ticker']+".SA"
  tickersdf = tickersdf.dropna()
  ticker_list = tickersdf['Ticker'].to_list()

  return tickersdf


  # period option: ['1d', '5d', '7d', '60d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
def fetch_prices(ticker_list, period='1y', interval='1d'):



  all_symbols = " ".join(ticker_list)
  tickers = Ticker(all_symbols, asynchronous=True)
  df = tickers.history(period=period, interval=interval)
  return df

#values: open,high,low,close,volume,adjclose,dividends,splits, market_cap, return
def transpose_stock_data(dataframe, values='adjclose'):


  # Carregar os dados
  df = dataframe

  # Check if the 'date' column exists, if not assume it's 'index'
  if 'date' not in df.columns:
      df = df.reset_index()
      df.rename(columns={'index': 'date'}, inplace=True)

  # print(df['date'])

  # Remove o timezone manualmente (como string)
  df['date'] = df['date'].astype(str).str.replace(r'(\\+|\\-)\d{2}:\d{2}$', '', regex=True)


  # Remover informações extras de horário, mantendo apenas a data
  df['date'] = pd.to_datetime(df['date'] , errors='coerce', dayfirst=False)


  # Calcular a capitalização de mercado

  df['market_cap'] = df['close'] * df['volume']



  # Criar um índice único de datas dos últimos 365 dias
  start_date = df['date'].min()
  end_date = df['date'].max()
  all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

  #remove linhas duplicadas
  df = df[~df.duplicated(subset=['symbol', 'date'], keep=False)]


  stocks_col = df.pivot(index='date', columns='symbol').columns.to_list()
  stocks_param = [ 'open','high','low','close','volume','adjclose','dividends',
                  'splits', 'market_cap', 'return','logreturn']

  # Cria um df multi-index
  colunas = pd.MultiIndex.from_product([stocks_col, stocks_param])
  df_result = pd.DataFrame(index=df.index , columns=colunas)

  # Reorganizar os dados

  if values == 'return':
    pivot_df = df.pivot(index='date', columns='symbol', values='adjclose')
    for col in pivot_df.columns[1:]:  # Ignorando a coluna de data
      pivot_df[col] = pivot_df[col].pct_change()
  else:
    pivot_df = df.pivot(index='date', columns='symbol', values=values)

  # Reindexar para garantir que todas as datas estejam presentes
  pivot_df = pivot_df.reindex(all_dates)

  # Preencher valores ausentes com os do dia anterior
  pivot_df.fillna(method='ffill', inplace=True)

  return pivot_df

#values: open,high,low,close,volume,adjclose,dividends,splits, market_cap, return
def transpose_stock_data_all(dataframe, values='adjclose'):

    df = dataframe.copy()

    # Garantir que a coluna 'date' existe
    if 'date' not in df.columns:
        df = df.reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)

    # Limpar timezone
    df['date'] = df['date'].astype(str).str.replace(r'(\+|\-)\d{2}:\d{2}$', '', regex=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)

    # Calcular market_cap
    df['market_cap'] = df['close'] * df['volume']

    # Criar intervalo de datas
    start_date = df['date'].min()
    end_date = df['date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Remover duplicatas
    df = df.drop_duplicates(subset=['symbol', 'date'])

    # Obter lista única de símbolos
    symbols = df['symbol'].unique()

    # Parâmetros padrão
    std_param = ['close', 'adjclose', 'market_cap']

    all_param = ['close', 'adjclose', 'market_cap', 'return', 'log_return']
    # Criar MultiIndex para colunas
    colunas = pd.MultiIndex.from_product([symbols, all_param])

    # Criar DataFrame de retorno com índice de datas completas
    df_result = pd.DataFrame(index=all_dates, columns=colunas)

    for param in std_param:
        # Pivotar o DataFrame para esse parâmetro
        pivot_df = df.pivot(index='date', columns='symbol', values=param)

        # Reindexar e preencher dados ausentes
        pivot_df = pivot_df.reindex(all_dates)
        pivot_df.ffill(inplace=True)
        pivot_df.bfill(inplace=True)

        # Preencher o df_result

        for symbol in symbols:
            if symbol in pivot_df.columns:
                df_result[(symbol, param)] = pivot_df[symbol]


    # Calcula os retornos
    df_return = df_result.xs('adjclose',axis=1,level=1).pct_change(fill_method=None)
    df_return.ffill(inplace=True)
    df_return.bfill(inplace=True)

    for symbol in symbols:
        if symbol in pivot_df.columns:
           df_result[(symbol, 'return')] = df_return[symbol]
           df_result[(symbol, 'log_return')] = np.log(df_return[symbol] + 1)
          #  ret = df_return[symbol] + 1
          #  if ret > 0:
          #   df_result[(symbol, 'log_return')] = np.log(ret)
          #  else:
          #   df_result[(symbol, 'log_return')] = 0

    return df_result

# Average daily return

def calculate_avg_daily_return(df_ticker_return):
  # local_df = df[ticker,'return']
  local_df = df_ticker_return

  # Calcular o produto acumulado: ∏ (1 + R_i)
  product = np.prod(1 + local_df)

  # Número de períodos (por padrão, usamos 252 dias úteis)
  n = len(local_df)

  # Calcular o retorno médio geométrico conforme a fórmula
  avg_daily_return = product ** (1/n) - 1

  return avg_daily_return

def calculate_avg_monthly_return(df_ticker_return):

  avg_daily_ret = calculate_avg_daily_return(df_ticker_return)
  avg_monthly_ret = (1 + avg_daily_ret) ** (30) - 1
  return avg_monthly_ret

# Average daily log return

def calculate_avg_daily_log_return(df, ticker):
  local_df = df[ticker,'log_return']


  # Calcular o produto acumulado:  sum (r_i)
  sum = np.sum(1 + local_df)

  # Número de períodos (por padrão, usamos 252 dias úteis)
  n = len(local_df)

  # Calcular o retorno médio conforme a fórmula
  avg_daily_return = sum * (1/n)

  return avg_daily_return

def calculate_avg_monthly_log_return(df, ticker):
  avg_daily_log_ret = calculate_avg_daily_log_return(df, ticker)
  # TODO: encontrar fórmula para calcular o retorno log médio mensal
  # avg_monthly_log_ret = (1 + avg_daily_log_ret) ** (1/30) - 1

  return None

def calculate_daily_log_std(df, ticker):
    """
    Calcula o desvio padrão diário dos retornos logarítmicos de um ticker.
    """
    local_df = df[ticker, 'log_return']

    # Desvio padrão dos retornos diários
    daily_std = np.std(local_df, ddof=1)  # ddof=1 para usar amostra

    return daily_std


def calculate_monthly_log_std(df, ticker):
    """
    Calcula o desvio padrão mensal (aprox. 21 dias úteis) a partir do desvio diário.
    """
    daily_std = calculate_daily_log_std(df, ticker)

    # Escalar para período mensal (raiz do tempo)
    monthly_std = daily_std * np.sqrt(21)

    return monthly_std

def calculate_avg_daily_return(local_df):
    """
    Retorno médio diário (retornos simples).
    """
    return np.mean(local_df)


def calculate_avg_monthly_return(local_df, days_in_month=21):
    """
    Retorno médio mensal (aprox. 21 dias úteis).
    """
    avg_daily = calculate_avg_daily_return(local_df)
    return avg_daily * days_in_month


def calculate_daily_std(local_df):
    """
    Desvio padrão diário dos retornos simples.
    """
    return np.std(local_df, ddof=1)


def calculate_monthly_std(local_df, days_in_month=21):
    """
    Desvio padrão mensal dos retornos simples (escala pela raiz do tempo).
    """
    daily_std = calculate_daily_std(local_df)
    return daily_std * np.sqrt(days_in_month)
"""Criar data frames"""

def get_avg_daily_monthly_returns(df):
  # columns = ['Ticker', 'Avg Daily Return', 'Avg Monthly Return']
  columns = [
        'Ticker',
        'Avg Daily Return',
        'Avg Monthly Return',
        'Std Daily Return',
        'Std Monthly Return'
    ]
  tickers = df.columns.get_level_values(0).unique().tolist()
  #df_avg_returns = pd.DataFrame(columns=columns)

  data = []
  for ticker in tickers:
    local_df = df[ticker,'return']
    avg_daily_return = calculate_avg_daily_return(local_df)
    avg_monthly_return = calculate_avg_monthly_return(local_df)

    # Desvios padrão
    std_daily_return = calculate_daily_std(local_df)
    std_monthly_return = calculate_monthly_std(local_df)

    # data.append([ticker, avg_daily_return, avg_monthly_return])
    data.append([
            ticker,
            avg_daily_return,
            avg_monthly_return,
            std_daily_return,
            std_monthly_return
        ])

  df_avg_returns = pd.DataFrame(data, columns=columns)
  return df_avg_returns



def get_tickers_by_sector_subsector_segment(df_tickers, sector='all', subsector='all', segment='all'):
  return_df = None
  df_columns = df_tickers.columns

  return_df = pd.DataFrame(columns=df_columns)
  i = 0
  result_rows = []
  for ticker in df_tickers['Ticker']:
    i +=  1


    ticker_info = df_tickers[df_tickers['Ticker'] == ticker]
    ticker_segment = ticker_info['SEGMENTO'].values[0]
    ticker_subsector = ticker_info['SUBSETOR'].values[0]
    ticker_sector = ticker_info['SETOR ECONÔMICO'].values[0]


    if (segment == 'all') or (ticker_segment == segment):
      # print(ticker_subsector == 'all')
      if (subsector == 'all') or (ticker_subsector == subsector):

        if sector == 'all' or ticker_sector == sector:
          # print(f'{i} - {ticker}, seg = {ticker_segment}, subsec = {ticker_subsector}, sec = {ticker_sector}')
          result_rows.append(ticker_info)
  if len(result_rows) > 0:
    return_df = pd.concat(result_rows)
    # print(f'{ticker}, seg = {ticker_segment}, subsec = {ticker_subsector}, sec = {ticker_sector}')
  return return_df




# prompt: Considere o dataframe gerado na célula anterior. Crie um dataframe que retorne a média dos retornos apenas o "return", não o "log_return" a partir de um outro dataframe que contenha um conjunto específico de tickers. Dessa forma, o dataframe resultante vai ter a coluna date (que será o index, da mesma forma que o dataframe recebido como primeiro parâmetro), uma coluna para cada ticker, e a coluna avg_return

def get_avg_returns_by_tickers(df_returns, df_tickers_list):
  """
  Calcula a média dos retornos diários para um conjunto específico de tickers.

  Args:
    df_returns: DataFrame com MultiIndex nas colunas (Ticker, métrica).
    df_tickers_list: DataFrame contendo a lista de tickers.

  Returns:
    DataFrame com retornos diários dos tickers válidos + média (avg_return).
  """
  tickers_to_analyze = df_tickers_list['Ticker'].tolist()

  # Garantir que só usaremos tickers que estão presentes no df_returns
  available_tickers = df_returns.columns.get_level_values(0).unique()
  valid_tickers = [ticker for ticker in tickers_to_analyze if ticker in available_tickers]

  # Verificar se há pelo menos um ticker válido
  if not valid_tickers:
    raise ValueError("Nenhum dos tickers fornecidos está presente no DataFrame de retornos.")

  # Filtrar os retornos
  df_filtered_returns = df_returns.loc[:, pd.IndexSlice[valid_tickers, 'return']].copy()

  # Remover o MultiIndex nas colunas
  df_filtered_returns.columns = df_filtered_returns.columns.get_level_values(0)

  # Calcular a média
  df_filtered_returns['avg_return'] = df_filtered_returns.mean(axis=1)

  return df_filtered_returns


def integrate_returns(df_returns):
  """
  Integra os retornos diários em um DataFrame de retornos acumulados.

  Args:
    df_returns: DataFrame contendo os retornos diários. Assume-se que
                o índice é a data e a(s) coluna(s) contêm os retornos diários.

  Returns:
    Um DataFrame com os retornos acumulados. O índice é a data.
  """
  # Para calcular retornos acumulados, somamos 1 a cada retorno diário e calculamos o produto acumulado.
  # O produto acumulado reflete o crescimento do capital investido.
  # Para obter apenas o retorno acumulado (excluindo o capital inicial), subtraímos 1.
  df_integrated = (1 + df_returns).cumprod() - 1
  return df_integrated

def plot_returns(df_integrated_all, title='Retorno médio no setor',savepath=None, showplt=True):
  # Cria a figura
  plt.figure(figsize=(12, 6))

  # Plota cada coluna individualmente
  for column in df_integrated_all.columns:
      if column == 'avg_return':
          # Linha da média com traço mais grosso e cor destacada
          plt.plot(df_integrated_all.index, df_integrated_all[column], label='avg_return', linewidth=2.5, color='black')
      else:
          # Outras linhas mais finas e discretas
          plt.plot(df_integrated_all.index, df_integrated_all[column],  label=column, linewidth=0.8, alpha=0.5)

  plt.title(title)
  plt.xlabel("Data")
  plt.ylabel("Retorno Acumulado")
  plt.grid(True)
  plt.legend(loc='best', fontsize='small', ncol=2)
  plt.tight_layout()
  if savepath is not None:
     plt.savefig(savepath, dpi=300)
  if showplt:
    plt.show()


def calculateDaylyMonthlyRetsBySector(df_tickers, df_stock_data):



    sectors = df_tickers['SETOR ECONÔMICO'].unique().tolist()

    df_columns = ['sector','avg_daily_ret','avg_monthly_ret']

    df_avg_daily_monthly_returns = pd.DataFrame(columns=df_columns)

    for sector in sectors:
        df_test = get_tickers_by_sector_subsector_segment(df_tickers, sector = sector)
        df_avg_returns_selected = get_avg_returns_by_tickers(df_stock_data, df_test)
        df_avg_returns_selected = remove_peak_outliers(df_avg_returns_selected, 8)
        sector_avg_daily_ret = calculate_avg_daily_return(df_avg_returns_selected['avg_return'])
        sector_avg_monthly_ret = calculate_avg_monthly_return(df_avg_returns_selected['avg_return'])

        data_line = [sector,sector_avg_daily_ret,sector_avg_monthly_ret]


        # Adiciona a linha no final
        df_avg_daily_monthly_returns.loc[len(df_avg_daily_monthly_returns)] = data_line     


    return df_avg_daily_monthly_returns



def calculateDaylyMonthlyRetsBySubSector(df_tickers, df_stock_data):



    subsectors = df_tickers['SUBSETOR'].unique().tolist()

    df_columns = ['sub sector','avg_daily_ret','avg_monthly_ret']

    df_avg_daily_monthly_returns = pd.DataFrame(columns=df_columns)

    for subsector in subsectors:
        df_test = get_tickers_by_sector_subsector_segment(df_tickers, subsector = subsector)
        df_avg_returns_selected = get_avg_returns_by_tickers(df_stock_data, df_test)
        sector_avg_daily_ret = calculate_avg_daily_return(df_avg_returns_selected['avg_return'])
        sector_avg_monthly_ret = calculate_avg_monthly_return(df_avg_returns_selected['avg_return'])

        data_line = [subsector,sector_avg_daily_ret,sector_avg_monthly_ret]


        # Adiciona a linha no final
        df_avg_daily_monthly_returns.loc[len(df_avg_daily_monthly_returns)] = data_line     


    return df_avg_daily_monthly_returns


def calculateDaylyMonthlyRetsBySegment(df_tickers, df_stock_data):



  segments = df_tickers['SEGMENTO'].unique().tolist()

  df_columns = ['segment','avg_daily_ret','avg_monthly_ret']

  df_avg_daily_monthly_returns = pd.DataFrame(columns=df_columns)

  for segment in segments:
      df_test = get_tickers_by_sector_subsector_segment(df_tickers, segment = segment)
      df_avg_returns_selected = get_avg_returns_by_tickers(df_stock_data, df_test)
      sector_avg_daily_ret = calculate_avg_daily_return(df_avg_returns_selected['avg_return'])
      sector_avg_monthly_ret = calculate_avg_monthly_return(df_avg_returns_selected['avg_return'])

      data_line = [segment,sector_avg_daily_ret,sector_avg_monthly_ret]


      # Adiciona a linha no final
      df_avg_daily_monthly_returns.loc[len(df_avg_daily_monthly_returns)] = data_line     


  return df_avg_daily_monthly_returns

def calculateDaylyMonthlyRetsByCath(sec_subsec_seg, df_tickers, df_stock_data, outlier_return=7.5):
    # DataFrame acumulador de outliers
    df_outliers = pd.DataFrame()
    # print(df_stock_data)
    if sec_subsec_seg is None:
        return None
    elif sec_subsec_seg == 'SETOR ECONOMICO':
      cath_upper = 'SETOR ECONÔMICO'
      cath_low = 'sector'
      pass
    elif sec_subsec_seg == 'SUBSETOR':
      cath_upper = 'SUBSETOR'
      cath_low = 'sub sector'
    elif sec_subsec_seg == 'SEGMENTO':
      cath_upper = 'SEGMENTO'
      cath_low = 'segment'
    else:
      return None



    caths = df_tickers[cath_upper].unique().tolist()

    df_columns = [cath_low,'avg_daily_ret','avg_monthly_ret']

    df_avg_daily_monthly_returns = pd.DataFrame(columns=df_columns)

    for cath in caths:
        try:
          if sec_subsec_seg == 'SETOR ECONOMICO':
            df_tickers_by_cath = get_tickers_by_sector_subsector_segment(df_tickers, sector = cath)
          elif sec_subsec_seg == 'SUBSETOR':
            df_tickers_by_cath = get_tickers_by_sector_subsector_segment(df_tickers, subsector = cath)
          elif sec_subsec_seg == 'SEGMENTO':
            df_tickers_by_cath = get_tickers_by_sector_subsector_segment(df_tickers, segment = cath)

          df_avg_returns_selected = get_avg_returns_by_tickers(df_stock_data, df_tickers_by_cath)

          df_avg_returns_selected, outlier = remove_peak_outliers(df_avg_returns_selected, outlier_return)
          # 🔹 concatena os outliers no DataFrame acumulador

          if outlier is not None and not outlier.empty:
              outlier_df = outlier.reset_index()
              outlier_df.columns = ['ticker', 'retavg_daily_return']

              df_outliers = pd.concat(
                  [df_outliers, outlier_df],
                  ignore_index=True
              )
        except Exception as e:
          print(f'{cath_upper} {cath}: {e}')
          cath_avg_daily_ret = 0.
          cath_avg_monthly_ret = 0.


        # cath_avg_daily_ret = calculate_avg_daily_return(df_stock_data['avg_return'])
        cath_avg_daily_ret = calculate_avg_daily_return(df_avg_returns_selected)


        cath_avg_monthly_ret = calculate_avg_monthly_return(df_avg_returns_selected['avg_return'])

        data_line = [cath,cath_avg_daily_ret,cath_avg_monthly_ret]


        # Adiciona a linha no final
        df_avg_daily_monthly_returns.loc[len(df_avg_daily_monthly_returns)] = data_line


    return df_avg_daily_monthly_returns, df_outliers

def calculateDaylyMonthlyRetsBySector(df_tickers, df_stock_data, outlier_return=7.5):
    
    df_avg_daily_monthly_returns, df_outliers = (
       calculateDaylyMonthlyRetsByCath(
          'SETOR ECONOMICO', 
          df_tickers, 
          df_stock_data, 
          outlier_return=outlier_return
      )
    )


    return df_avg_daily_monthly_returns, df_outliers

def calculateDaylyMonthlyRetsBySubSector(df_tickers, df_stock_data, outlier_return=7.5):
    df_avg_daily_monthly_returns, df_outliers = (
       calculateDaylyMonthlyRetsByCath(
          'SUBSETOR', 
          df_tickers, 
          df_stock_data, 
          outlier_return=outlier_return
      )
    )


    return df_avg_daily_monthly_returns, df_outliers


def calculateDaylyMonthlyRetsBySegment(df_tickers, df_stock_data, outlier_return=7.5):
    df_avg_daily_monthly_returns, df_outliers = (
       calculateDaylyMonthlyRetsByCath(
          'SEGMENTO', 
          df_tickers, 
          df_stock_data, 
          outlier_return=outlier_return
      )
    )


    return df_avg_daily_monthly_returns, df_outliers

def remove_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    upper = Q3 + 4 * IQR
    lower = Q1 - 4 * IQR

    return series[(series > upper) | (series < lower)]

def remove_outlier_stocks(df,lim=1.5):
    cumulative_returns = (1 + df).prod() - 1

    Q1 = cumulative_returns.quantile(0.25)
    Q3 = cumulative_returns.quantile(0.75)
    IQR = Q3 - Q1

    upper = Q3 + lim * IQR
    lower = Q1 - lim * IQR

    outliers = cumulative_returns[
        (cumulative_returns > upper) |
        (cumulative_returns < lower)
    ]

    df_clean = df.drop(columns=outliers.index)

    return df_clean, outliers

def remove_peak_outliers(df,lim=1.5):
    df_cum = (1 + df).cumprod() - 1
    max_peaks = df_cum.max()

    Q1 = max_peaks.quantile(0.25)
    Q3 = max_peaks.quantile(0.75)
    IQR = Q3 - Q1

    upper = Q3 + lim * IQR

    outliers = max_peaks[max_peaks > upper]

    df_clean = df.drop(columns=outliers.index)

    return df_clean, outliers


def get_top_stocks(df_tickers, df_stock_data, df_sector=None, df_subsector=None, df_segment=None, n=30, outlier_return=7.):
  df_avg_ret = get_avg_daily_monthly_returns(df_stock_data)

  top_n_df = (
      df_avg_ret
      .sort_values(by='Avg Monthly Return', ascending=False)
      .head(n+20)
  )

  if df_sector is None:
    df_sector=(
      calculateDaylyMonthlyRetsBySector(
          df_tickers = df_tickers, df_stock_data = df_stock_data
          )
      )
    
  if df_subsector is None:
    df_subsector=(
      calculateDaylyMonthlyRetsBySubSector(
          df_tickers = df_tickers, df_stock_data = df_stock_data
          )
      )
    
  if df_segment is None:
    df_segment=(
      calculateDaylyMonthlyRetsBySegment(
          df_tickers = df_tickers, df_stock_data = df_stock_data
          )
      )



  # top_n_list = top_n_df['Ticker'].tolist()
  # top_n_avg_daily_return_mean = top_n_df['Avg Daily Return'].mean()
  top_n_avg_monthly_return_mean = top_n_df['Avg Monthly Return'].mean()

  df_avg_returns_selected = get_avg_returns_by_tickers(
      df_stock_data, 
      top_n_df
  )



  df_avg_returns_selected, outliers = remove_peak_outliers(
      df=df_avg_returns_selected,
      lim=outlier_return
  )
  df_avg_returns_selected = df_avg_returns_selected.iloc[:,:n]

  

  top_n_df = top_n_df[
      ~top_n_df['Ticker'].isin(outliers.index)
  ]

 
  df_integrated_avg_returns = integrate_returns(
      df_avg_returns_selected
  )


  tickers_filtered = top_n_df['Ticker'].to_list()

  df_avg_ret_sector1 = (
      df_avg_ret
      .loc[df_avg_ret['Ticker'].isin(tickers_filtered)]
      .sort_values(by='Avg Monthly Return', ascending=False)    
  )
  
  # tickers_info = tickers_df.loc[tickers_df['Ticker'].isin(tickers_filtered)]

  # df_final = pd.concat([df_avg_ret_sector1, tickers_info], axis=1)

  df_final = pd.merge(df_avg_ret_sector1,
      df_tickers,
      on="Ticker",   # coluna comum entre os dois
      how="inner"    # ou "left", "right", "outer", dependendo do que você quiser
  )
  # df_final.head(50)

  # ---- Merge Setor ----
  df_final = pd.merge(
      df_final,
      df_sector.rename(columns={
          'avg_daily_ret': 'Sector avg daily ret',
          'avg_monthly_ret': 'Sector avg monthly ret'
      }),
      left_on='SETOR ECONÔMICO',
      right_on='sector',
      how='left'
  ).drop(columns=['sector'])

  # ---- Merge Subsetor ----
  df_final = pd.merge(
      df_final,
      df_subsector.rename(columns={
          'avg_daily_ret': 'Subsector avg daily ret',
          'avg_monthly_ret': 'Subsector avg monthly ret'
      }),
      left_on='SUBSETOR',
      right_on='sub sector',
      how='left'
  ).drop(columns=['sub sector'])

  # ---- Merge Segmento ----
  df_final = pd.merge(
      df_final,
      df_segment.rename(columns={
          'avg_daily_ret': 'Segment avg daily ret',
          'avg_monthly_ret': 'Segment avg monthly ret'
      }),
      left_on='SEGMENTO',
      right_on='segment',
      how='left'
  ).drop(columns=['segment'])

  return df_final, df_avg_returns_selected, df_integrated_avg_returns, top_n_avg_monthly_return_mean, outliers


def plot_stock_acc_ret_hist(stock, df_avg_returns_selected):


  ret_stock = df_avg_returns_selected[stock]
  mu, sigma = np.mean(ret_stock), np.std(ret_stock, ddof=1)
  skewness = skew(ret_stock)
  kurt = kurtosis(ret_stock)
  kde = gaussian_kde(ret_stock)
  # Estatísticas básicas

  
  fig, ax = plt.subplots(1, 2, figsize=(12, 5))
  count, bins, ignored = plt.hist(ret_stock, bins=50, edgecolor="black", alpha=0.7)

  x = np.linspace(min(bins), max(bins), 100)

  ticker_avg_daily_ret = calculate_avg_daily_return(ret_stock)
  ticker_avg_monthly_ret = calculate_avg_monthly_return(ret_stock)


  df_integrated_avg_returns = integrate_returns(ret_stock)


  ax[1].set_title(f'{stock}, retorno médio: {ticker_avg_monthly_ret*100:.3f}% a.m., {ticker_avg_daily_ret*100:.5f}% a.d.\n Skew={skewness:.2f}, Kurt={kurt:.2f}')
  ax[1].set_xlabel('Retorno Diário')
  ax[1].set_ylabel('Frequência')

  # Curva empírica (densidade observada suavizada
  ax[1].plot(x, kde(x), 'b-', linewidth=2, label='Densidade empírica')
  # ax[1].plot(ret_stock)

  # Curva normal teórica
  ax[1].plot(x, norm.pdf(x, mu, sigma), 'r--', linewidth=2, label='Normal teórica')
  ax[0].plot(df_integrated_avg_returns.index, df_integrated_avg_returns, label='Retorno Acumulado')


  plt.show()