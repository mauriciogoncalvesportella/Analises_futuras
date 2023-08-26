import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

# Digite o código da ação
ticker = input("Digite o código da ação: ")

# Obtém os dados do histórico de preços
dados = yf.Ticker(ticker).history("2y")

# Renomeia a coluna de preços de fechamento para 'y' e redefine o índice
dados = dados.reset_index()
dados = dados.rename(columns={"Date": "ds", "Close": "y"})

# Remove o fuso horário da coluna 'ds'
dados['ds'] = dados['ds'].dt.tz_localize(None)

# Cria um modelo Prophet
modelo = Prophet()

# Treina o modelo com os dados
modelo.fit(dados)

# Cria um DataFrame com datas futuras para fazer previsões
futuro = modelo.make_future_dataframe(periods=365)  # 365 dias no futuro

# Faz as previsões
previsao = modelo.predict(futuro)

# Plota o gráfico usando a função plot_plotly
fig = plot_plotly(modelo, previsao)
fig.show()
