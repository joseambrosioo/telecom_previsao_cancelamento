# app.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
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
telcom = pd.read_csv('dataset/churn-bigml-80.csv')
telcom_test = pd.read_csv('dataset/churn-bigml-20.csv')
    
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

telcom = preprocess_data_for_app(telcom)
telcom_test = preprocess_data_for_app(telcom_test)

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

# --- Layout do Painel ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("☎️", className="me-2"),
                    dbc.NavbarBrand("Previsão de Cancelamento (Churn) de Clientes de Telecom", class_name="fw-bold text-wrap", style={"color": "black"}),
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
prepare_tab = html.Div(
    children=[
        html.H4(
            ["📝 ", html.B("PREPARE"), " — Preparando os Dados"],
            className="mt-4"
        ),
        html.P("Antes de podermos construir um modelo preditivo, precisamos entender e limpar nossos dados."),
        html.H5("Fonte de Dados"),
        html.P(
            ["Usamos um conjunto de dados padrão de churn de telecom, dividido em um ", html.B("conjunto de treinamento"), " (80% dos dados) para construir nossos modelos e um ", html.B("conjunto de teste"), " separado (20%) para verificar se nossos modelos funcionam em novos dados, nunca vistos."]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Conjunto de Dados de Treinamento"),
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
                            dbc.CardHeader("Conjunto de Dados de Teste"),
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
                        html.P(
                            ["O gráfico de pizza abaixo mostra que nossos dados estão ", html.B("desbalanceados"), "—uma pequena porcentagem de clientes cancelou. Isso é importante porque significa que um modelo simples poderia obter uma alta pontuação de acurácia apenas prevendo que ninguém nunca cancelará. É por isso que precisamos de métricas de avaliação mais avançadas."]
                        ),
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
                        html.P(
                            ["A ", html.B("Matriz de Correlação"), " à direita mostra o quão fortemente cada característica se relaciona com as outras. Uma cor escura indica uma relação forte. A principal conclusão é que características como minutos de chamada e chamadas para o serviço de atendimento ao cliente estão correlacionadas com o churn, o que confirma nossa intuição."]
                        ),
                        html.H5("Visualização das Características", className="mt-4"),
                        html.P("Este gráfico visualiza os dados usando duas características chave: total de minutos diurnos e total de minutos noturnos. Separamos esses períodos porque o comportamento do cliente e os motivos para o churn podem ser diferentes ao longo do dia. Um cliente que faz muitas chamadas longas durante o dia pode ser um usuário de negócios, enquanto um cliente com chamadas noturnas longas pode ser um usuário familiar. Esses diferentes comportamentos podem ter diferentes razões para o churn. Um modelo que olha apenas o total de minutos não capturaria essas nuances."),
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
                            ["• ", html.B("O Problema com a Acurácia"), ": Para nossos dados desbalanceados, a ", html.B("Acurácia"), " (a porcentagem de previsões corretas) não é a melhor métrica. Um modelo que sempre prevê 'sem churn' poderia ter 85% de acurácia, mas seria inútil para identificar clientes em risco."]
                        ),
                        html.P(
                            ["• ", html.B("Métricas Chave"), ": Focamos em um conjunto mais completo de métricas: ", html.B("Revocação"), " (quantos clientes que cancelaram nós pegamos?), ", html.B("Precisão"), " (daqueles que marcamos como churners, quantos estavam corretos?), ", html.B("F1-Score"), " (um equilíbrio de ambos), e ", html.B("ROC-AUC"), " (o quão bem o modelo separa os clientes que cancelam dos que não cancelam)."]
                        ),
                        dbc.Row([dbc.Col(dcc.Graph(id="train-metrics-bar"), md=12)]),
                        html.P(
                            ["Nossa análise mostra que ", html.B("LightGBM"), " e ", html.B("XGBoost"), ", ambos tipos de modelos avançados baseados em árvores, superaram significativamente os outros. Eles alcançaram altas pontuações em todas as métricas, provando que são excelentes na identificação do churn."]
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="train-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="train-roc-curve"), md=6),
                        ]),
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
                        dbc.Row([dbc.Col(dcc.Graph(id="test-metrics-bar"), md=12)]),
                        html.P(
                            ["O desempenho nos dados de teste é semelhante ao dos dados de treinamento, confirmando que o ", html.B("Classificador LGBM"), " e o ", html.B("Classificador XGBoost"), " são excelentes escolhas para este problema. Suas altas pontuações de ", html.B("F1-Score"), " e ", html.B("ROC-AUC"), " em ambos os conjuntos de dados indicam uma capacidade preditiva forte e confiável."]
                        ),
                        dbc.Row([
                            dbc.Col(dcc.Graph(id="test-confusion-matrix"), md=6),
                            dbc.Col(dcc.Graph(id="test-roc-curve"), md=6),
                        ]),
                        html.P("O desempenho consistente desses modelos no conjunto de teste é uma descoberta chave. Sugere que um sistema preditivo pode ser construído para identificar clientes em risco de churn com alta confiança."),
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

@app.callback(
    Output("train-metrics-bar", "figure"),
    Output("test-metrics-bar", "figure"),
    Output("train-confusion-matrix", "figure"),
    Output("test-confusion-matrix", "figure"),
    Output("train-roc-curve", "figure"),
    Output("test-roc-curve", "figure"),
    Input('feature-importance-model', 'value')
)
def update_model_performance(selected_model):
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
    test_bar_chart = get_bar_chart(metrics_test, "Métricas do Modelo (Dados de Teste)")

    model = model_results.get(selected_model, model_results['Classificador LGBM'])

    y_pred_train = model.predict(X_test)
    cm_train = confusion_matrix(y_test, y_pred_train)
    fig_cm_train = ff.create_annotated_heatmap(
        z=cm_train, x=["Não Churn", "Churn"], y=["Não Churn", "Churn"],
        colorscale='blues'
    )
    fig_cm_train.update_layout(title=f"Matriz de Confusão ({selected_model} no Treinamento)", height=450, margin=dict(t=50, b=50))

    y_pred_test = model.predict(telcom_test[cols])
    cm_test = confusion_matrix(telcom_test['Churn'], y_pred_test)
    fig_cm_test = ff.create_annotated_heatmap(
        z=cm_test, x=["Não Churn", "Churn"], y=["Não Churn", "Churn"],
        colorscale='blues'
    )
    fig_cm_test.update_layout(title=f"Matriz de Confusão ({selected_model} no Teste)", height=450, margin=dict(t=50, b=50))

    def get_roc_curve(model, X, y, title):
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, probabilities)
            fig = go.Figure(data=[
                go.Scatter(x=fpr, y=tpr, mode='lines', name='Curva ROC'),
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Chute Aleatório')
            ])
            fig.update_layout(
                title=title,
                xaxis_title="Taxa de Falso Positivo",
                yaxis_title="Taxa de Verdadeiro Positivo",
                height=450,
                margin=dict(t=50, b=50)
            )
        else:
            fig = go.Figure(go.Scatter())
            fig.update_layout(title=f"Curva ROC Não Disponível para {selected_model}", height=450, margin=dict(t=50, b=50))
        return fig

    roc_train = get_roc_curve(model, X_test, y_test, f"Curva ROC ({selected_model} no Treinamento)")
    roc_test = get_roc_curve(model, telcom_test[cols], telcom_test['Churn'], f"Curva ROC ({selected_model} no Teste)")

    return (
        train_bar_chart,
        test_bar_chart,
        fig_cm_train,
        fig_cm_test,
        roc_train,
        roc_test,
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