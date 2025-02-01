import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

def preprocess_data(df, target_column=None):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    st.write(categorical_cols)
    
    if target_column in categorical_cols and target_column is not None:
        categorical_cols.remove(target_column)

    # df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
        # Inicializa o LabelEncoder
    label_encoder = LabelEncoder()

    # Aplica o LabelEncoder a cada coluna categórica
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    st.write(df)
    
    df.columns = [
        col.replace("[", "")
        .replace("]", "")
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "_")
        for col in df.columns
    ]

    return df, categorical_cols

uploaded_file = "../upload_bases/dados_seguro.csv"
target_column = "preco_ideal"

df = pd.read_csv(uploaded_file, delimiter=",")

# st.write(df)

df, categorical_cols = preprocess_data(df, target_column)

X = df.drop(columns=[target_column])
y = df[target_column]

# st.write(X)

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo de exemplo
# modelo = RandomForestRegressor(n_estimators=50, random_state=42)
modelo = XGBRegressor(random_state=42)
modelo.fit(X_train, y_train)

# Criar um subconjunto dos dados para SHAP
amostra_X = X_train.sample(n=50, random_state=42)  # Reduzir tamanho para evitar problemas de memória

# Definir o explicador SHAP para modelos baseados em árvores
explainer = shap.TreeExplainer(modelo)

# Calcular valores SHAP
shap_values = explainer.shap_values(amostra_X)

# Plotagem no Streamlit
st.title("📊 Explicabilidade do Modelo com SHAP")
st.subheader("🌟 Importância das Variáveis")

fig, ax = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values, amostra_X, show=False)
st.pyplot(fig)
