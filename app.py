# app.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# Importação dos Modelos (necessário para carregar as classes)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Carregamento de Dados e Objetos Pré-treinados ---
# Carregar os transformadores e a lista de colunas da pasta 'data'
try:
    std = joblib.load('data/scaler.pkl')
    le = joblib.load('data/label_encoder.pkl')
    cols = joblib.load('data/features.pkl')
except FileNotFoundError:
    print("Erro: Arquivos de pré-processamento não encontrados. Certifique-se de ter executado 'train_models.py' primeiro.")
    exit()

# Carregar os dados brutos e aplicar o pré-processamento salvo
telcom_raw = pd.read_csv('dataset/churn-bigml-80.csv')
telcom_test_raw = pd.read_csv('dataset/churn-bigml-20.csv')

def preprocess_data_for_app(df):
    """Aplica as transformações salvas nos dados brutos."""
    col_to_drop = [
        'State', 'Area code', 'Total day charge', 'Total eve charge',
        'Total night charge', 'Total intl charge'
    ]
    df = df.drop(columns=col_to_drop, axis=1)

    bin_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Churn']
    for col in bin_cols:
        df[col] = le.transform(df[col])

    num_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col not in bin_cols + ['Churn']]
    df[num_cols] = std.transform(df[num_cols])
    return df

telcom = preprocess_data_for_app(telcom_raw)
telcom_test = preprocess_data_for_app(telcom_test_raw)

# Separar os dados para avaliação
y_test = telcom[['Churn']]
X_test = telcom[cols]

# --- Carregar os Modelos Treinados ---
print("Carregando modelos treinados...")
model_results = {}
try:
    for filename in os.listdir('models'):
        if filename.endswith('.pkl'):
            # Transforma o nome do arquivo para o formato de exibição
            model_name = filename.replace('.pkl', '').replace('_', ' ')
            model_results[model_name] = joblib.load(f'models/{filename}')
    print("Modelos carregados com sucesso!")
except FileNotFoundError:
    print("Erro: Pasta 'models' não encontrada. Certifique-se de ter executado 'train_models.py' primeiro.")
    exit()

# --- Definição das Métricas ---
def get_metrics_df(X, y):
    df_rows = []
    for name, model in model_results.items():
        predictions = model.predict(X)
        try:
            probabilities = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, probabilities)
        except (AttributeError, IndexError):
            roc_auc = "N/A"
        
        df_rows.append({
            'Modelo': name,
            'Acurácia': accuracy_score(y, predictions),
            'Revocação': recall_score(y, predictions, zero_division=0),
            'Precisão': precision_score(y, predictions, zero_division=0),
            'F1-Score': f1_score(y, predictions, zero_division=0),
            'ROC-AUC': roc_auc,
            'Kappa': cohen_kappa_score(y, predictions),
        })
    return pd.DataFrame(df_rows).round(4)

metrics_train = get_metrics_df(X_test, y_test)
metrics_test = get_metrics_df(telcom_test[cols], telcom_test[['Churn']])


# --- Funções para gerar textos estáticos e dinâmicos ---
def get_best_and_worst_models(df, metric='F1-Score'):
    """Encontra os 2 melhores e 2 piores modelos para uma métrica específica."""
    df_sorted = df.sort_values(by=metric, ascending=False)
    best_2 = df_sorted.head(2).reset_index(drop=True)
    worst_2 = df_sorted.tail(2).reset_index(drop=True)
    return best_2, worst_2

def generate_static_metrics_summary(df, data_type):
    best_models_f1, worst_models_f1 = get_best_and_worst_models(df, 'F1-Score')
    best_model_name_f1 = best_models_f1.loc[0, 'Modelo']
    best_model_f1_score = best_models_f1.loc[0, 'F1-Score']
    second_best_model_name_f1 = best_models_f1.loc[1, 'Modelo']
    second_best_f1_score = best_models_f1.loc[1, 'F1-Score']

    return html.P([
        f"Avaliamos o desempenho dos modelos no conjunto de {data_type} usando métricas-chave. ",
        "Os modelos de melhor desempenho foram ", html.B(f"{best_model_name_f1}"),
        " com um ", html.B(f"F1-Score de {best_model_f1_score}"), ", seguido por ",
        html.B(f"{second_best_model_name_f1}"), " com um ", html.B(f"F1-Score de {second_best_f1_score}"),
        ". Em geral, esses modelos avançados de árvore mostraram resultados excepcionais."
    ])

def generate_static_confusion_summary(df, data_type):
    best_models_acc, _ = get_best_and_worst_models(df, 'Acurácia')
    best_model = model_results[best_models_acc.loc[0, 'Modelo']]
    y_pred = best_model.predict(telcom_test[cols]) if data_type == 'teste' else best_model.predict(X_test)
    y_actual = telcom_test['Churn'] if data_type == 'teste' else y_test
    cm = confusion_matrix(y_actual, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return html.P([
        "Uma análise detalhada da matriz de confusão para o modelo de melhor desempenho, ",
        html.B(best_models_acc.loc[0, 'Modelo']), f" (acurácia de {best_models_acc.loc[0, 'Acurácia']}), revela sua capacidade de prever com precisão. ",
        f"O modelo identificou corretamente ", html.B(f"{tp} Verdadeiros Positivos"),
        " e ", html.B(f"{tn} Verdadeiros Negativos"), ", enquanto os erros foram minimizados, com apenas ",
        html.B(f"{fp} Falsos Positivos"), " e ", html.B(f"{fn} Falsos Negativos"),
        ". Isso demonstra um equilíbrio ideal entre capturar o churn e evitar falsos alarmes."
    ])

def generate_static_roc_summary(df, data_type):
    best_models_roc, _ = get_best_and_worst_models(df, 'ROC-AUC')
    best_model_name = best_models_roc.loc[0, 'Modelo']
    best_model_auc = best_models_roc.loc[0, 'ROC-AUC']
    second_best_model_name = best_models_roc.loc[1, 'Modelo']
    second_best_auc = best_models_roc.loc[1, 'ROC-AUC']

    return html.P([
        "A curva ROC avalia a capacidade de diferenciação do modelo. Os modelos com os maiores valores de Área Sob a Curva (AUC) são os melhores. ",
        "Os melhores modelos para essa métrica foram ", html.B(f"{best_model_name}"), " com um ",
        html.B(f"AUC de {best_model_auc}"), ", e ", html.B(f"{second_best_model_name}"),
        f" com um ", html.B(f"AUC de {second_best_auc}"),
        ". Ambas as pontuações, próximas de 1.00, indicam que esses modelos são excelentes em distinguir clientes que cancelam de clientes que não cancelam."
    ])

# --- Layout do Painel ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Previsão de Cancelamento (Churn) de Clientes de Empresa de Telecomunicações"
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("☎️", className="me-2"),
                    dbc.NavbarBrand("Previsão de Cancelamento (Churn) de Clientes de Empresa de Telecomunicações", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. Aba ASK
ask_tab = dcc.Markdown(
    """
    ### ❓ **ASK** — A Visão Geral
    Esta seção define o propósito do projeto e seu valor para o negócio.

    **Tarefa de Negócio**: O objetivo é prever quais clientes são propensos a parar de usar nosso serviço, um processo conhecido como **churn de clientes**. Para as empresas de telecomunicações, manter os clientes existentes é muito mais barato do que encontrar novos. Ao prever o churn, podemos entrar em contato proativamente com clientes em risco e tentar reconquistá-los.

    **Partes Interessadas**: Os principais tomadores de decisão que usarão este painel são **Marketing** e **Serviço ao Cliente**. Eles precisam dessas informações para planejar campanhas direcionadas e estratégias de retenção. A liderança executiva também se beneficia de uma visão de alto nível de nossos esforços de retenção de clientes.

    **Entregáveis**: O produto final é este dashboard, que oferece um passo a passo claro de nossa análise e apresenta as principais descobertas e recomendações.
    """, className="p-4"
)

# 2. Aba PREPARE
# Para a tabela de dados, vamos definir explicitamente os tipos de colunas para permitir a filtragem numérica
columns_with_types = []
for col in telcom_raw.columns:
    col_type = telcom_raw[col].dtype
    if pd.api.types.is_numeric_dtype(col_type):
        columns_with_types.append({"name": col, "id": col, "type": "numeric"})
    elif pd.api.types.is_bool_dtype(col_type):
        columns_with_types.append({"name": col, "id": col, "type": "text"})
    else:
        columns_with_types.append({"name": col, "id": col})

# Converte a coluna 'Churn' para string para exibição e filtragem correta
telcom_raw_display = telcom_raw.head(10).copy()
telcom_raw_display['Churn'] = telcom_raw_display['Churn'].astype(str)

prepare_tab = html.Div(
    children=[
        html.H4(
            ["📝 ", html.B("PREPARE"), " — Preparando os Dados"],
            className="mt-4"
        ),
        html.P("Antes de podermos construir um modelo preditivo, precisamos entender e limpar nossos dados."),
        html.H5("Fonte de Dados"),
        html.P(
            ["Usamos um dataset padrão de churn de telecom, dividido em um ", html.B("conjunto de treinamento"), " (80% dos dados) para construir nossos modelos e um ", html.B("conjunto de teste"), " separado (20%) para verificar se nossos modelos funcionam em novos dados, nunca vistos."]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Dataset de Treinamento"),
                            dbc.CardBody(
                                [
                                    html.P(f"Linhas: {telcom.shape[0]}"),
                                    html.P(f"Características: {telcom.shape[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Dataset de Teste"),
                            dbc.CardBody(
                                [
                                    html.P(f"Linhas: {telcom_test.shape[0]}"),
                                    html.P(f"Características: {telcom_test.shape[1]}"),
                                ]
                            ),
                        ], className="mb-3"
                    )
                ),
            ]
        ),
        html.H4("Resumo Estatístico", className="mt-4"),
        html.P("Esta tabela fornece uma visão geral estatística rápida das características. Note a relação linear perfeita entre minutos e cobranças para diferentes tipos de chamada. Para evitar multicolinearidade em nossos modelos, removemos as colunas relacionadas a cobranças."),
        dbc.Table.from_dataframe(telcom.describe().T.reset_index().rename(columns={'index': 'característica'}).round(2),
                                 striped=True, bordered=True, hover=True),
        # Adiciona a tabela de amostra do dataset
        html.H5("Amostra do Dataset Original (Primeiras 10 Linhas)", className="mt-4"),
        dash_table.DataTable(
            id='sample-table',
            columns=columns_with_types, # Usa a nova lista de colunas com tipos definidos
            data=telcom_raw_display.to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_action="none",
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'font-size': '12px',
                'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
    ], className="p-4"
)

# 3. Aba ANALYZE com sub-abas
analyze_tab = html.Div(
    children=[
        html.H4(
            ["📈 ", html.B("ANALYZE"), " — Encontrando Padrões e Construindo Modelos"],
            className="mt-4"
        ),
        html.P("Aqui é onde exploramos os dados e construímos o cérebro preditivo do nosso painel."),
        dbc.Tabs([
            dbc.Tab(label="Análise Exploratória de Dados", children=[
                html.Div(
                    children=[
                        html.H5("Distribuição e Correlações do Churn", className="mt-4"),
                        html.P([
                            "O gráfico de pizza abaixo mostra que nossos dados ",
                            html.B("não"), " estão ",
                            html.B("balanceados"),
                            " — uma pequena porcentagem de clientes ",
                            html.B("cancelou,"), " apenas ",
                            html.B("14.6%"),
                            ". Isso é importante porque significa que um ",
                            html.B("modelo simples"),
                            " poderia obter uma alta pontuação de ",
                            html.B("acurácia"),
                            " apenas prevendo que ninguém nunca cancelará. É por isso que precisamos de ",
                            html.B("métricas de avaliação mais avançadas"),
                            "."
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="churn-pie-chart",
                                             figure=go.Figure(
                                                 data=[go.Pie(labels=telcom["Churn"].value_counts().keys().tolist(),
                                                              values=telcom["Churn"].value_counts().values.tolist(),
                                                              marker=dict(colors=['#1f77b4', '#ff7f0e'], line=dict(color="white", width=1.3)),
                                                              hoverinfo="label+percent", hole=0.5)],
                                                 layout=go.Layout(title="Distribuição de Churn de Clientes", height=400, margin=dict(t=50, b=50))
                                             )), md=6),
                            dbc.Col(dcc.Graph(id="correlation-matrix"), md=6),
                        ]),
                        html.P([
                            "A ",
                            html.B("Matriz de Correlação"),
                            " acima mostra o quão fortemente cada ",
                            html.B("característica"),
                            " se relaciona com as outras. Quanto mais ",
                            html.B("escura"),
                            " for a cor da célula de uma característica na interseção com o ",
                            html.B("churn"),
                            ", maior a ",
                            html.B("relação"),
                            " entre essa característica e o churn. Quanto mais ",
                            html.B("clara"),
                            " for a cor da célula, menor a ",
                            html.B("relação"),
                            ". A principal conclusão é que características como ",
                            html.B("minutos de chamada, plano internacional,"),
                            " e ",
                            html.B("chamadas para o serviço de atendimento ao cliente"),
                            " estão correlacionadas com o ",
                            html.B("churn"),
                            "."
                        ]),
                        html.H5("Visualização das Características", className="mt-4"),
                        html.P([
                            "Este gráfico visualiza os dados usando duas ",
                            html.B("características chave"),
                            ": ",
                            html.B("total de minutos diurnos"),
                            " e ",
                            html.B("total de minutos noturnos"),
                            ". Separamos esses períodos porque o ",
                            html.B("comportamento do cliente"),
                            " e os ",
                            html.B("motivos para o churn"),
                            " podem ser diferentes ao longo do dia. Um cliente que faz muitas ",
                            html.B("chamadas longas durante o dia"),
                            " pode ser um ",
                            html.B("usuário de negócios"),
                            ", enquanto um cliente com ",
                            html.B("chamadas noturnas longas"),
                            " pode ser um ",
                            html.B("usuário familiar"),
                            ". Esses diferentes comportamentos podem ter diferentes ",
                            html.B("razões para o churn"),
                            ". Um ",
                            html.B("modelo"),
                            " que olha apenas o ",
                            html.B("total de minutos"),
                            " não capturaria essas ",
                            html.B("nuances"),
                            "."
                        ]),
                        dbc.Row([
                            dbc.Col(dcc.Graph(
                                id="day-eve-minutes-plot",
                                figure=go.Figure(
                                    data=go.Scatter(x=telcom['Total day minutes'], y=telcom['Total eve minutes'],
                                                     mode='markers', marker_color=telcom['Churn'], showlegend=False),
                                    layout=go.Layout(title="Total de Minutos Diurnos vs. Total de Minutos Noturnos",
                                                     xaxis_title="Total de Minutos Diurnos (Escalado)",
                                                     yaxis_title="Total de Minutos Noturnos (Escalado)",
                                                     height=400, margin=dict(t=50, b=50))
                                ))),
                        ]),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Desempenho do Modelo (Treinamento)", children=[
                html.Div(
                    children=[
                        html.H5("Desempenho do Modelo nos Dados de Treinamento", className="mt-4"),
                        html.P("Treinamos uma variedade de modelos de aprendizado de máquina para ver qual deles tem o melhor desempenho."),
                        html.P(
                            [html.B("O Problema com a Acurácia"), ": Para nossos dados não balanceados, a ", html.B("Acurácia"), " (a porcentagem de previsões corretas) não é a melhor métrica. Um modelo que sempre prevê 'sem churn' poderia ter 85% de acurácia, mas seria inútil para identificar clientes em risco."]
                        ),
                        html.P(
                            [html.B("Métricas Chave"), ": Focamos em um conjunto mais completo de métricas:",
                            html.Ul([
                                html.Li([html.B("Revocação"), " – quantos clientes que cancelaram nós conseguimos identificar?"]),
                                html.Li([html.B("Precisão"), " – daqueles que marcamos como churners, quantos estavam corretos?"]),
                                html.Li([html.B("F1-Score"), " – um equilíbrio entre Precisão e Revocação."]),
                                html.Li([html.B("ROC-AUC"), " – o quão bem o modelo separa os clientes que cancelam dos que não cancelam."])
                            ])
                            ]
                        ),
                        generate_static_metrics_summary(metrics_train, 'treinamento'),
                        dbc.Row([dbc.Col(dcc.Graph(id="train-metrics-bar"), md=12)]),
                        html.Hr(),
                        html.H5("Matriz de Confusão e Curva ROC", className="mt-4"),
                        html.H6("Matriz de Confusão", className="mt-4"),
                        html.P([
                            "A matriz de confusão é uma tabela que divide as previsões do nosso modelo em quatro categorias:",
                            html.Ul([
                                html.Li([html.B("Verdadeiros Positivos (VP):"), " Clientes que o modelo previu corretamente que iriam cancelar."]),
                                html.Li([html.B("Verdadeiros Negativos (VN):"), " Clientes que o modelo previu corretamente que não iriam cancelar."]),
                                html.Li([html.B("Falsos Positivos (FP):"), " Clientes que o modelo previu incorretamente que iriam cancelar (Erro Tipo I). Isso pode levar a esforços de retenção desnecessários."]),
                                html.Li([html.B("Falsos Negativos (FN):"), " Clientes que iriam cancelar, mas que o modelo não identificou (Erro Tipo II). Esta é a categoria mais custosa, pois representa a perda de um cliente."])
                            ])
                        ]),
                        generate_static_confusion_summary(metrics_train, 'treinamento'),
                        html.H6("Curva ROC (Receiver Operating Characteristic)", className="mt-4"),
                        html.P([
                            "A curva ROC plota a ",
                            html.B("Taxa de Verdadeiros Positivos"),
                            " em relação à ",
                            html.B("Taxa de Falsos Positivos"),
                            ". Quanto mais próxima a curva estiver do canto superior esquerdo, melhor o modelo diferencia entre as duas classes (churn e não-churn). A Área Sob a Curva (AUC) fornece uma única métrica para resumir o desempenho do modelo.",
                        ]),
                        generate_static_roc_summary(metrics_train, 'treinamento'),
                        html.P("Selecione um modelo para visualizar as métricas, Matriz de Confusão e a Curva ROC (Treinamento):"),
                        dcc.Dropdown(
                            id='model-selector-train',
                            options=[{'label': name, 'value': name} for name in model_results.keys()],
                            value='Classificador LGBM',
                            clearable=False,
                        ),
                        html.Div(id='selected-train-metrics-summary'),
                        html.Div(id='selected-train-confusion-summary'),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="train-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="train-roc-curve"), md=6),
                        ]),
                        html.Div(id='selected-train-roc-summary'),
                        html.Hr(),
                        html.H5("Importância das Características (para modelos baseados em árvores)", className="mt-4"),
                        html.P("Este gráfico classifica as características com base em quanto elas contribuíram para a previsão do modelo. As duas características mais importantes foram `Total day minutes` e `International plan`. Isso nos dá um ponto de partida claro para nossas recomendações."),
                        dcc.Dropdown(
                            id="feature-importance-model",
                            options=[{'label': name, 'value': name} for name in ['Árvore de Decisão', 'Random Forest', 'Classificador LGBM', 'Classificador XGBoost', 'Gradient Boosting']],
                            value='Classificador LGBM'
                        ),
                        dcc.Graph(id="feature-importance-plot"),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Desempenho do Modelo (Teste)", children=[
                html.Div(
                    children=[
                        html.H5("Desempenho do Modelo nos Dados de Teste", className="mt-4"),
                        html.P(
                            ["Testamos nossos principais modelos nos dados não vistos para garantir que não estão com ", html.B("overfitting"), " (memorizando os dados de treinamento em vez de aprender padrões gerais). O desempenho permaneceu alto, confirmando que os modelos são confiáveis e funcionarão bem em um cenário real."]
                        ),
                        # Inserir o texto estático do melhor e pior modelo aqui
                        generate_static_metrics_summary(metrics_test, 'teste'),
                        dbc.Row([dbc.Col(dcc.Graph(id="test-metrics-bar"), md=12)]),
                        html.Hr(),
                        html.H5("Matriz de Confusão e Curva ROC", className="mt-4"),
                        html.H6("Matriz de Confusão", className="mt-4"),
                        html.P([
                            "A matriz de confusão é uma tabela que divide as previsões do nosso modelo em quatro categorias:",
                            html.Ul([
                                html.Li([html.B("Verdadeiros Positivos (VP):"), " Clientes que o modelo previu corretamente que iriam cancelar."]),
                                html.Li([html.B("Verdadeiros Negativos (VN):"), " Clientes que o modelo previu corretamente que não iriam cancelar."]),
                                html.Li([html.B("Falsos Positivos (FP):"), " Clientes que o modelo previu incorretamente que iriam cancelar (Erro Tipo I). Isso pode levar a esforços de retenção desnecessários."]),
                                html.Li([html.B("Falsos Negativos (FN):"), " Clientes que iriam cancelar, mas que o modelo não identificou (Erro Tipo II). Esta é a categoria mais custosa, pois representa a perda de um cliente."])
                            ])
                        ]),
                        generate_static_confusion_summary(metrics_test, 'teste'),
                        html.H6("Curva ROC (Receiver Operating Characteristic)", className="mt-4"),
                        html.P([
                            "A curva ROC plota a ",
                            html.B("Taxa de Verdadeiros Positivos"),
                            " em relação à ",
                            html.B("Taxa de Falsos Positivos"),
                            ". Quanto mais próxima a curva estiver do canto superior esquerdo, melhor o modelo diferencia entre as duas classes (churn e não-churn). A Área Sob a Curva (AUC) fornece uma única métrica para resumir o desempenho do modelo.",
                        ]),
                        generate_static_roc_summary(metrics_test, 'teste'),
                        html.P("Selecione um modelo para visualizar as métricas, Matriz de Confusão e a Curva ROC (Teste):"),
                        dcc.Dropdown(
                            id='model-selector-test',
                            options=[{'label': name, 'value': name} for name in model_results.keys()],
                            value='Classificador LGBM',
                            clearable=False,
                        ),
                        html.Div(id='selected-test-metrics-summary'),
                        html.Div(id='selected-test-confusion-summary'),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="test-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="test-roc-curve"), md=6),
                        ]),
                        html.Div(id='selected-test-roc-summary'), # Este é o Div que faltava
                    ], className="p-4"
                )
            ]),
        ])
    ]
)

# 4. Aba ACT
act_tab = dcc.Markdown(
    """
    ### 🚀 **ACT** — O que Fazer em Seguida
    Esta é a seção mais importante, pois traduz os insights dos dados em uma estratégia de negócio.

    -   **Direcionar Clientes de Alto Risco**: Os modelos identificaram que clientes com um **plano internacional** e alto **total de minutos diurnos** são os mais propensos a cancelar. Este é o grupo perfeito para uma campanha de marketing direcionada.
    -   **Retenção Proativa**: Em vez de esperar que os clientes cancelem, a empresa deve usar o modelo implantado para obter uma lista diária de clientes com alto risco de churn. Um representante de serviço ao cliente pode então ligar para eles proativamente para oferecer um desconto, uma atualização de serviço, ou simplesmente para verificar sua satisfação.
    -   **Implantar o Melhor Modelo**: O **Classificador LightGBM** é nosso modelo recomendado para implantação devido ao seu desempenho superior nos dados de treinamento e teste. Este modelo será o cérebro por trás de nossa nova estratégia proativa de redução de churn.
    """, className="p-4"
)

app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(ask_tab, label="Perguntar"),
                dbc.Tab(prepare_tab, label="Preparar"),
                dbc.Tab(analyze_tab, label="Analisar"),
                dbc.Tab(act_tab, label="Agir"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks ---
@app.callback(
    Output("correlation-matrix", "figure"),
    Input("churn-pie-chart", "id") # Entrada fictícia para acionar no carregamento
)
def update_corr_matrix(dummy):
    correlation = telcom.corr()
    fig = ff.create_annotated_heatmap(
        z=correlation.values.round(2),
        x=list(correlation.columns),
        y=list(correlation.index),
        colorscale="Viridis",
        showscale=True,
        reversescale=True
    )
    fig.update_layout(title="Matriz de Correlação", height=500, margin=dict(t=50, b=50))
    return fig

# Callback para os gráficos e texto de desempenho (treinamento)
@app.callback(
    Output("train-metrics-bar", "figure"),
    Output("train-confusion-matrix", "figure"),
    Output("train-roc-curve", "figure"),
    Output("selected-train-metrics-summary", "children"),
    Output("selected-train-confusion-summary", "children"),
    Output("selected-train-roc-summary", "children"),
    Input('model-selector-train', 'value')
)
def update_train_performance(selected_model):
    def get_bar_chart(df, title):
        fig = go.Figure()
        for metric in ['Acurácia', 'Revocação', 'Precisão', 'F1-Score', 'ROC-AUC', 'Kappa']:
            fig.add_trace(go.Bar(
                y=df["Modelo"],
                x=df[metric],
                orientation='h',
                name=metric
            ))
        fig.update_layout(
            barmode='group',
            title=title,
            height=450,
            margin=dict(l=150, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    train_bar_chart = get_bar_chart(metrics_train, "Métricas do Modelo (Dados de Treinamento)")
    
    model = model_results.get(selected_model, model_results['Classificador LGBM'])
    y_pred_train = model.predict(X_test)
    cm_train = confusion_matrix(y_test, y_pred_train)
    
    tn, fp, fn, tp = cm_train.ravel()
    z_data_train = np.array([[tp, fn], [fp, tn]])
    cm_text_train = np.array([[f'TP: {tp}', f'FN: {fn}'], [f'FP: {fp}', f'TN: {tn}']])
    
    fig_cm_train = ff.create_annotated_heatmap(
        z=z_data_train,
        x=["Previu Churn (1)", "Previu Não-Churn (0)"],
        y=["Churn Real (1)", "Não-Churn Real (0)"],
        annotation_text=cm_text_train,
        colorscale='blues',
        showscale=False
    )
    fig_cm_train.update_yaxes(autorange='reversed')
    fig_cm_train.update_layout(title=f"Matriz de Confusão ({selected_model} no Treinamento)", height=450, margin=dict(t=50, b=50))
    fig_cm_train.update_annotations(font_size=16)

    def get_roc_curve(model, X, y, title):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probabilities)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure(data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Curva ROC (AUC={roc_auc:.2f})'),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chute Aleatório')
            ])
            fig.update_layout(
                title=title,
                xaxis_title="Taxa de Falso Positivo",
                yaxis_title="Taxa de Verdadeiro Positivo",
                height=450,
                margin=dict(t=50, b=50)
            )
            roc_summary = html.P([
                "O ", html.B(selected_model), " obteve um ", html.B(f"AUC de {roc_auc:.2f}"), 
                ". Uma pontuação de AUC próxima de 1.00 indica uma excelente capacidade de diferenciar entre as classes de churn e não-churn."
            ])
        else:
            fig = go.Figure(go.Scatter())
            fig.update_layout(title=f"Curva ROC Não Disponível para {selected_model}", height=450, margin=dict(t=50, b=50))
            roc_summary = html.P(f"A Curva ROC e a métrica AUC não estão disponíveis para o modelo {selected_model}.")
        return fig, roc_summary

    roc_train_fig, roc_train_summary = get_roc_curve(model, X_test, y_test, f"Curva ROC ({selected_model} no Treinamento)")
    
    # Textos de resumo dinâmicos para o modelo selecionado
    selected_metrics_train = metrics_train[metrics_train['Modelo'] == selected_model].iloc[0]
    metrics_summary_train = html.P([
        "O modelo selecionado, ", html.B(selected_model), ", obteve os seguintes resultados: ",
        html.B(f"Acurácia de {selected_metrics_train['Acurácia']}"), ", ",
        html.B(f"Precisão de {selected_metrics_train['Precisão']}"), ", ",
        html.B(f"Revocação de {selected_metrics_train['Revocação']}"), ", ",
        html.B(f"F1-Score de {selected_metrics_train['F1-Score']}"), "."
    ])

    confusion_summary_train = html.P([
        "Para o modelo selecionado, ", html.B(selected_model), ", a matriz de confusão mostrou: ",
        html.B(f"{tp} Verdadeiros Positivos (VP)"), ", ",
        html.B(f"{tn} Verdadeiros Negativos (VN)"), ", ",
        html.B(f"{fp} Falsos Positivos (FP)"), " e ",
        html.B(f"{fn} Falsos Negativos (FN)"), "."
    ])

    return (
        train_bar_chart,
        fig_cm_train,
        roc_train_fig,
        metrics_summary_train,
        confusion_summary_train,
        roc_train_summary
    )

# Novo callback para os gráficos e texto de desempenho (teste)
@app.callback(
    Output("test-metrics-bar", "figure"),
    Output("test-confusion-matrix", "figure"),
    Output("test-roc-curve", "figure"),
    Output("selected-test-metrics-summary", "children"),
    Output("selected-test-confusion-summary", "children"),
    Output("selected-test-roc-summary", "children"),
    Input('model-selector-test', 'value')
)
def update_test_performance(selected_model):
    def get_bar_chart(df, title):
        fig = go.Figure()
        for metric in ['Acurácia', 'Revocação', 'Precisão', 'F1-Score', 'ROC-AUC', 'Kappa']:
            fig.add_trace(go.Bar(
                y=df["Modelo"],
                x=df[metric],
                orientation='h',
                name=metric
            ))
        fig.update_layout(
            barmode='group',
            title=title,
            height=450,
            margin=dict(l=150, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    test_bar_chart = get_bar_chart(metrics_test, "Métricas do Modelo (Dados de Teste)")
    
    model = model_results.get(selected_model, model_results['Classificador LGBM'])
    y_pred_test = model.predict(telcom_test[cols])
    cm_test = confusion_matrix(telcom_test['Churn'], y_pred_test)

    tn, fp, fn, tp = cm_test.ravel()
    z_data_test = np.array([[tp, fn], [fp, tn]])
    cm_text_test = np.array([[f'TP: {tp}', f'FN: {fn}'], [f'FP: {fp}', f'TN: {tn}']])
    
    fig_cm_test = ff.create_annotated_heatmap(
        z=z_data_test,
        x=["Previu Churn (1)", "Previu Não-Churn (0)"],
        y=["Churn Real (1)", "Não-Churn Real (0)"],
        annotation_text=cm_text_test,
        colorscale='blues',
        showscale=False
    )
    fig_cm_test.update_yaxes(autorange='reversed')
    fig_cm_test.update_layout(title=f"Matriz de Confusão ({selected_model} no Teste)", height=450, margin=dict(t=50, b=50))
    fig_cm_test.update_annotations(font_size=16)

    def get_roc_curve(model, X, y, title):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probabilities)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure(data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Curva ROC (AUC={roc_auc:.2f})'),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chute Aleatório')
            ])
            fig.update_layout(
                title=title,
                xaxis_title="Taxa de Falso Positivo",
                yaxis_title="Taxa de Verdadeiro Positivo",
                height=450,
                margin=dict(t=50, b=50)
            )
            roc_summary = html.P([
                "O ", html.B(selected_model), " obteve um ", html.B(f"AUC de {roc_auc:.2f}"), 
                ". Uma pontuação de AUC próxima de 1.00 indica uma excelente capacidade de diferenciar entre as classes de churn e não-churn."
            ])
        else:
            fig = go.Figure(go.Scatter())
            fig.update_layout(title=f"Curva ROC Não Disponível para {selected_model}", height=450, margin=dict(t=50, b=50))
            roc_summary = html.P(f"A Curva ROC e a métrica AUC não estão disponíveis para o modelo {selected_model}.")
        return fig, roc_summary
    
    roc_test_fig, roc_test_summary = get_roc_curve(model, telcom_test[cols], telcom_test['Churn'], f"Curva ROC ({selected_model} no Teste)")
    
    # Textos de resumo dinâmicos para o modelo selecionado
    selected_metrics_test = metrics_test[metrics_test['Modelo'] == selected_model].iloc[0]
    metrics_summary_test = html.P([
        "O modelo selecionado, ", html.B(selected_model), ", obteve os seguintes resultados: ",
        html.B(f"Acurácia de {selected_metrics_test['Acurácia']}"), ", ",
        html.B(f"Precisão de {selected_metrics_test['Precisão']}"), ", ",
        html.B(f"Revocação de {selected_metrics_test['Revocação']}"), ", ",
        html.B(f"F1-Score de {selected_metrics_test['F1-Score']}"), "."
    ])
    
    confusion_summary_test = html.P([
        "Para o modelo selecionado, ", html.B(selected_model), ", a matriz de confusão mostrou: ",
        html.B(f"{tp} Verdadeiros Positivos (VP)"), ", ",
        html.B(f"{tn} Verdadeiros Negativos (VN)"), ", ",
        html.B(f"{fp} Falsos Positivos (FP)"), " e ",
        html.B(f"{fn} Falsos Negativos (FN)"), "."
    ])

    return (
        test_bar_chart,
        fig_cm_test,
        roc_test_fig,
        metrics_summary_test,
        confusion_summary_test,
        roc_test_summary
    )

@app.callback(
    Output("feature-importance-plot", "figure"),
    Input("feature-importance-model", "value")
)
def update_feature_importance(selected_model):
    model = model_results.get(selected_model)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = cols
        df_importance = pd.DataFrame({
            'característica': feature_names,
            'importância': importances
        }).sort_values(by='importância', ascending=False)
        
        fig = go.Figure(go.Bar(
            x=df_importance['importância'],
            y=df_importance['característica'],
            orientation='h'
        ))
        fig.update_layout(
            title=f"Importância das Características para {selected_model}",
            xaxis_title="Importância",
            yaxis_title="Característica",
            height=500,
            margin=dict(l=150, t=50, b=50)
        )
        return fig
    else:
        fig = go.Figure(go.Scatter())
        fig.update_layout(title=f"Importância das Características Não Disponível para {selected_model}", height=450, margin=dict(t=50, b=50))
        return fig

if __name__ == "__main__":
    app.run(debug=True)