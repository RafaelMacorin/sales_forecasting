# Projeto de Previsão de Vendas - Rossmann

<div align="center">
<img src="img/rossmann_logo.png" />
</div>

# Introdução
Este é um projeto end-to-end de Data Science com modelo de regressão adaptada para séries temporais. Foram criados 4 tipos de modelos para predizer o valor das vendas das lojas Rossmann nas próximas 6 semanas. As previsões podem ser acessadas pelo usuário por meio de um BOT no aplicativo do Telegram.

Este repositório contém a solução para a resolução de um problema do Kaggle: https://www.kaggle.com/c/rossmann-store-sales

### Plano de desenvolvimento do projeto

Este projeto foi desenvolvido seguindo a metodologia do CRISP-DM (Cross-Industry Standard Process for Data Mining), mas com um enfoque maior para Ciência de Dados. Dessa forma, o processo tradicional do CRISP-DM foi expandido para contemplar melhor o desenvolvimento e a implementação de modelos de Machine Learning, garantindo um fluxo mais robusto e alinhado às demandas atuais da área.

1. **Questão de Negócio** – O primeiro passo é definir claramente o problema que se deseja resolver. Aqui, compreendi os desafios e objetivos que o projeto precisa atender.
2. **Entendimento do Negócio** – Nesta etapa, analisei o contexto do problema, explorei as necessidades do usuário e identifiquei quais insights e previsões são mais valiosas.
3. **Coleta de Dados** – Busquei, extraí e armazenei os dados necessários para a análise, garantindo que sejam relevantes e confiáveis.
4. **Limpeza dos Dados** – Apliquei técnicas de data wrangling, removendo inconsistências, lidando com valores ausentes e padronizando o formato dos dados para garantir qualidade.
5. **Exploração dos Dados** – Realizei uma Análise Exploratória dos Dados (EDA), identificando padrões, tendências e possíveis problemas nos dados.
6. **Modelagem dos Dados** – Selecionei as variáveis mais relevantes e transformei os dados para alimentar os modelos de Machine Learning.
7. **Algoritmos de Machine Learning** – Apliquei diferentes modelos e técnicas, ajustando hiperparâmetros e otimizando a performance.
8. **Avaliação do Algoritmo** – Medi a performance dos modelos com métricas apropriadas, garantindo que atendam aos requisitos do projeto.
9. **Modelo em Produção** – Implementei a solução em um ambiente real, disponibilizando o modelo para uso e monitorando seu desempenho.

<div align="center">
<img src="img/crisp-dm.png" />
</div>

### Sumário

* [1. Descrição e Problema de Negócio](#1-descrição-e-problema-de-negócio)
* [2. Base de Dados](#2-base-de-dados)
* [3. Estratégia de Solução](#3-estratégia-de-solução)
* [4. Featuring Engineering](#4-featuring-engineering)
* [5. Análise Exploratória dos Dados](#5-análise-exploratória-dos-dados)
* [6. Seleção do Modelo de Machine Learning](#6-seleção-do-modelo-de-machine-learning)
* [7. Performance do Modelo](#7-performance-do-modelo)
* [8. Resultados de Negócio](#8-resultados-de-negócio)
* [9. Modelo em Produção](#9-modelo-em-produção)
* [10. Conclusão](#10-conclusão)
* [11. Aprendizados e Trabalhos Futuros](#11-aprendizados-e-trabalhos-futuros)

# 1. Descrição e Problema de Negócio

## 1.1. Descrição

A Rossmann, uma das maiores redes de drogarias da Europa, opera mais de 3.000 lojas em 7 países. Atualmente, os gerentes das lojas são responsáveis por prever as vendas diárias com até seis semanas de antecedência. No entanto, essa tarefa pode ser desafiadora, pois as vendas são impactadas por diversos fatores, como:

* Promoções e descontos;
* Concorrência na região;
* Feriados escolares e estaduais;
* Sazonalidade e tendências de mercado;
* Características específicas da localidade.

Com mais de mil gerentes tomando decisões individualmente com base em suas próprias experiências, a precisão das previsões pode variar significativamente.

## 1.2. Problema de Negócio

A Rossmann deseja um modelo de previsão de vendas confiável, capaz de gerar previsões diárias para 1.115 lojas na Alemanha. Previsões precisas permitirão:

1. Melhor planejamento da equipe, aumentando a produtividade e a satisfação dos funcionários;
2. Redução de custos operacionais, evitando excesso ou falta de funcionários;
3. Aprimoramento da experiência do cliente, garantindo que a equipe esteja sempre preparada para a demanda esperada.

O desafio consiste em desenvolver um modelo robusto de Machine Learning para prever as vendas diárias das lojas ao longo de seis semanas. Além disso, será necessário fornecer ao CEO uma forma de consulta rápida dessas previsões por meio do celular, garantindo acesso fácil e ágil às informações estratégicas.

# 2. Base de dados

O conjunto de dados possui as seguintes variáveis:

| Variável                          | Descrição |
|-----------------------------------|-----------|
| **Id**                            | Um identificador que representa um par (Loja, Data) dentro do conjunto de teste. |
| **Store**                         | Um identificador único para cada loja. |
| **Sales**                         | O faturamento de um determinado dia (essa é a variável que será prevista). |
| **Customers**                     | O número de clientes em um determinado dia. |
| **Open**                          | Indicador de funcionamento da loja: 0 = fechada, 1 = aberta. |
| **StateHoliday**                  | Indica um feriado estadual. Normalmente, todas as lojas, com algumas exceções, estão fechadas nesses dias. Todas as escolas estão fechadas em feriados públicos e finais de semana. <br> **a** = feriado público, **b** = feriado de Páscoa, **c** = Natal, **0** = nenhum feriado. |
| **SchoolHoliday**                 | Indica se a loja foi afetada pelo fechamento das escolas públicas naquela data. |
| **StoreType**                     | Diferencia quatro tipos diferentes de lojas: **a, b, c, d**. |
| **Assortment**                    | Descreve o nível de variedade de produtos da loja: <br> **a** = básico, **b** = extra, **c** = estendido. |
| **CompetitionDistance**           | Distância, em metros, até a loja concorrente mais próxima. |
| **CompetitionOpenSince[Month/Year]** | Indica o mês e o ano aproximado em que a loja concorrente mais próxima foi inaugurada. |
| **Promo**                         | Indica se a loja está realizando uma promoção naquele dia. |
| **Promo2**                        | Promoção contínua e consecutiva para algumas lojas: **0** = a loja não participa, **1** = a loja participa. |
| **Promo2Since[Year/Week]**        | Indica o ano e a semana do calendário em que a loja começou a participar da Promo2. |
| **PromoInterval**                 | Descreve os intervalos consecutivos em que a Promo2 é iniciada, indicando os meses em que a promoção se renova. <br> Exemplo: **"Fev, Mai, Ago, Nov"** significa que a promoção se inicia novamente em fevereiro, maio, agosto e novembro de cada ano para aquela loja. |

# 3. Estratégia de Solução

A estratégia adotada foi dividida nas seguintes etapas:

### Etapa 01: Análise do Conjunto de Dados
Inicialmente, foi realizada uma análise do conjunto de dados, verificando aspectos como os nomes das colunas, as dimensões do conjunto, os tipos de dados, além da identificação e preenchimento de valores ausentes (NA). Também foi realizada uma análise descritiva das variáveis, identificando quais eram categóricas.
### Etapa 02: Featuring Engineering
Nesta etapa, foram criados novos atributos (colunas) a partir das variáveis originais, com o objetivo de melhorar a compreensão dos fenômenos representados por cada variável.
### Etapa 03: Filtragem de Dados
O conjunto de dados foi filtrado de duas formas: primeiro, removendo as linhas que não correspondiam às lojas ativas e que não realizaram vendas (open != 0 e sales > 0), e segundo, eliminando colunas que não agregavam valor à análise ou cujos dados já haviam sido derivados para outras variáveis.
### Etapa 04: Análise Exploratória dos Dados (EDA)
Foi realizada uma exploração detalhada dos dados com o intuito de identificar insights valiosos para o entendimento do negócio. Foram realizadas análises univariadas, bivariadas e multivariadas, buscando descobrir padrões e correlações importantes entre as variáveis.
### Etapa 05: Preparação dos Dados
Nesta fase, os dados foram preparados para serem utilizados nos algoritmos de Machine Learning. Isso envolveu a aplicação de técnicas de normalização e codificação, convertendo variáveis categóricas em valores numéricos.
### Etapa 06: Seleção de Atributos para o Modelo
A seleção das variáveis mais relevantes foi realizada utilizando o método Boruta, que ajudou a identificar os atributos com maior impacto na performance do modelo.
### Etapa 07: Treinamento do Modelo de Machine Learning
Os modelos de Machine Learning foram treinados utilizando o conjunto de dados, e aquele que obteve a melhor performance, considerando a validação cruzada, seguiu para a próxima fase, que consistiu na otimização dos parâmetros do modelo para melhorar sua generalização.
### Etapa 08: Ajuste de Hiperparâmetro
Nesta etapa, os parâmetros do modelo foram ajustados para maximizar o aprendizado, utilizando o método RandomSearch para encontrar a combinação ideal de valores.
### Etapa 09: Conversão dos Resultados em Valor de Negócio
O desempenho do modelo foi analisado do ponto de vista do impacto no negócio, traduzindo os resultados para métricas que pudessem agregar valor real à empresa.
### Etapa 10: Implantação do Modelo em Produção
O modelo foi colocado em produção em um ambiente de nuvem (Render), permitindo que ele fosse acessado por diferentes usuários ou serviços, facilitando o processo de tomada de decisão no negócio.
### Etapa 11: Criação de um Bot no Telegram
Foi desenvolvido um bot no Telegram, permitindo que as previsões pudessem ser acessadas a qualquer hora e de qualquer lugar, bastando para isso uma conexão à internet e o aplicativo instalado no celular.

# 4. Featuring Engineering

Nesta etapa, novas variáveis foram criadas para capturar padrões temporais e melhorar a qualidade dos dados utilizados no modelo. Essas variáveis ajudam a destacar tendências sazonais e o impacto de diferentes períodos no volume de vendas.

Foram extraídas informações como:
* Dia, mês e ano das vendas para identificar variações sazonais.
* Semana do ano para capturar eventos recorrentes e feriados.
* Transformações temporais, como a criação de variáveis cíclicas (seno e cosseno) para representar padrões semanais e mensais de forma contínua.

### 4.1. Mapeamento de Hipóteses

Antes da modelagem, foi feita uma análise detalhada dos fatores que influenciam as vendas. Para isso, um mapa mental foi estruturado, organizando hipóteses que relacionam variáveis como promoções, sazonalidade, concorrência e comportamento dos consumidores.

<div align="center">
<img src="img/MindMapHypothesis.png" />
</div>

# 5. Análise Exploratória dos Dados

### 5.1. Análise Univariada

Para compreender melhor o comportamento das variáveis do conjunto de dados, foi realizada uma Análise Univariada, que permite observar a distribuição individual de cada variável e identificar padrões, outliers e tendências relevantes.

A análise das variáveis numéricas foi feita por meio de histogramas, que mostram a distribuição dos valores no dataset. Esses gráficos ajudam a identificar:

* Padrões de distribuição, como dados concentrados em certos intervalos.
* Presença de outliers, que podem impactar a modelagem.
* Assimetria dos dados, indicando possíveis transformações necessárias.

A visualização dessas distribuições auxilia na preparação dos dados e na definição de estratégias para tratamento de valores extremos e normalização das variáveis.

### 5.2. Análise Bivariada
