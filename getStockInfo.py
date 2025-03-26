import yfinance as yf


def buscar_dados_acao(ticker):
    try:
        acao = yf.Ticker(ticker)
        dados = acao.history(period='1d')

        if dados.empty:
            print("Nenhum dado encontrado para o ticker fornecido.")
            return

        print(f"Dados da ação {ticker}:")
        print(dados[['Open', 'High', 'Low', 'Close', 'Volume']])
    except Exception as e:
        print(f"Erro ao buscar dados: {e}")


if __name__ == "__main__":
    # ticker = input("Digite o ticker da ação (ex: AAPL, TSLA, PETR4.SA): ")
    ticker = "PETR4.SA"
    buscar_dados_acao(ticker)
