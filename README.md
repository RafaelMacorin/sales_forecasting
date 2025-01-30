# Rossmann Sales Forecasting Project

<div align="center">
<img src="img/rossmann_logo.png" />
</div>

# Introdução
Este é um projeto end-to-end de Data Science com modelo de regressão adaptada para séries temporais. Foram criados 4 tipos de modelos para predizer o valor das vendas das lojas Rossmann nas próximas 6 semanas. As previsões podem ser acessadas pelo usuário por meio de um BOT no aplicativo do Telegram.

Este repositório contém a solução para a resolução de um problema do Kaggle: https://www.kaggle.com/c/rossmann-store-sales

### Metodologia: CRISP-DS (Cross-Industry Standard Process for Data Science)
Para a execução deste projeto, segui a metodologia CRISP-DS, um padrão bastante utilizado em projetos de ciência de dados. Essa metodologia oferece um ciclo de vida claro, permitindo transformar dados em conhecimento prático para suportar a tomada de decisões. As etapas principais aplicadas foram:
1. Entendimento do Problema de Negócio
* Identificação das necessidades do cliente: prever as vendas das lojas Rossmann nas próximas 6 semanas.
* Tradução do objetivo de negócio em um problema de ciência de dados.
2. Entendimento dos Dados
* Exploração das variáveis disponíveis, como promoções, feriados, sazonalidade, concorrência e outras características das lojas.
3. Preparação dos Dados
* Limpeza e organização dos dados.
* Criação de novas variáveis (feature engineering) para capturar padrões temporais e comportamentais.
4. Modelagem
* Construção de múltiplos modelos de regressão, incluindo Regressão Linear, Random Forest e XGBoost.
* Ajuste de hiperparâmetros para maximizar o desempenho.
5. Avaliação
* Comparação dos modelos com métricas como RMSE (Root Mean Square Error) e validação cruzada.
* Análise dos resultados com foco em gerar insights que agreguem valor ao negócio.
6. Deploy e Monitoramento
* Implementação do modelo final em produção por meio de uma API, permitindo que as previsões sejam acessadas de forma prática e escalável.
