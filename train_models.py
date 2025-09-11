# train_models.py

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

# --- Pré-processamento dos Dados ---
def preprocess_data(train_path, test_path):
    telcom = pd.read_csv(train_path)
    telcom_test = pd.read_csv(test_path)
    
    col_to_drop = [
        'State', 'Area code', 'Total day charge', 'Total eve charge',
        'Total night charge', 'Total intl charge'
    ]
    telcom = telcom.drop(columns=col_to_drop, axis=1)
    telcom_test = telcom_test.drop(columns=col_to_drop, axis=1)

    bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
    bin_cols = [col for col in bin_cols if col != 'Churn']
    
    le = LabelEncoder()
    for col in bin_cols:
        telcom[col] = le.fit_transform(telcom[col])
        telcom_test[col] = le.transform(telcom_test[col])

    num_cols = [col for col in telcom.columns if telcom[col].dtype in ['float64', 'int64'] and col not in bin_cols + ['Churn']]
    std = StandardScaler()
    telcom[num_cols] = std.fit_transform(telcom[num_cols])
    telcom_test[num_cols] = std.transform(telcom_test[num_cols])
    
    target_col = ['Churn']
    cols = [col for col in telcom.columns if col not in target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        telcom[cols], telcom[target_col], test_size=0.25, random_state=111
    )
    
    # Salvar o scaler e o label encoder na pasta 'data'
    joblib.dump(std, 'data/scaler.pkl')
    joblib.dump(le, 'data/label_encoder.pkl')

    return telcom, telcom_test, X_train, X_test, y_train, y_test, cols

if __name__ == '__main__':
    # Criar as pastas 'models' e 'data' se elas não existirem
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('data'):
        os.makedirs('data')

    telcom, telcom_test, X_train, X_test, y_train, y_test, cols = preprocess_data(
        'dataset/churn-bigml-80.csv', 'dataset/churn-bigml-20.csv'
    )

    models = {
        'Regressao Logistica': LogisticRegression(solver='liblinear', random_state=42),
        'Arvore de Decisao': DecisionTreeClassifier(max_depth=9, random_state=42),
        'Classificador KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=9, random_state=42),
        'Naive Bayes Gaussiano': GaussianNB(),
        'SVM (RBF)': SVC(C=10.0, gamma=0.1, probability=True, random_state=42),
        'Classificador LGBM': LGBMClassifier(learning_rate=0.5, max_depth=7, n_estimators=100, random_state=42),
        'Classificador XGBoost': XGBClassifier(learning_rate=0.9, max_depth=7, n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
        'Classificador MLP': MLPClassifier(max_iter=1000, random_state=42),
        'Classificador Bagging': BaggingClassifier(random_state=42),
    }

    # Treinar e salvar todos os modelos
    model_results = {}
    print("Iniciando o treinamento e salvamento dos modelos...")
    for name, model in models.items():
        print(f"Treinando {name}...")
        model.fit(X_train, y_train.values.ravel())
        joblib.dump(model, f'models/{name.replace(" ", "_")}.pkl')
        model_results[name] = model
    print("Treinamento e salvamento concluídos.")
    
    # Salvar a lista de colunas para uso posterior
    joblib.dump(cols, 'data/features.pkl')