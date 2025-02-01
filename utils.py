import json
import os
import pickle
import pandas as pd
import streamlit as st

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from config import get_settings
from model import HistoricoModel
from repository_historico_modelo import insert_historico_modelo


settings = get_settings()


def preprocess_data(df, target_column):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    df.columns = [
        col.replace("[", "")
        .replace("]", "")
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "_")
        for col in df.columns
    ]

    return df, categorical_cols


def validar_coluna_alvo(df):
    if df is not None:
        target_column = st.sidebar.selectbox("Escolha a variável alvo", df.columns)
    else:
        st.warning(
            "⚠️ Nenhum arquivo foi carregado. Por favor, faça o upload de um arquivo CSV."
        )
        st.stop()
    return target_column


@st.cache_data
def load_data(uploaded_file=None):
    # sourcery skip: remove-unnecessary-else, swap-if-else-branches
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, delimiter=";")
        df.columns = df.columns.str.strip()
        return df
    else:
        return None


def gerar_modelo_pickle(
    uploaded_file, X_train, modelo, modelo_selecionado, mae, mse, r2
):
    # modelo_salvo = {"modelo": modelo, "features": X_train.columns.tolist()}

    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"modelo_{modelo_selecionado}_{os.path.splitext(uploaded_file.name)[0]}_{data_hora}.pkl"

    with open(model_filename, "wb") as f:
        pickle.dump(modelo, f)
    
    historico_modelo = HistoricoModel(
        data_treinamento=data_hora,
        modelo=modelo_selecionado,
        mae=mae,
        mse=mse,
        r2=r2,
        arquivo=model_filename,
    )
    
    id_insert = insert_historico_modelo(historico_modelo)

    st.sidebar.success(f"✅ Modelo salvo como {model_filename} e histórico atualizado! - {id_insert}")
    return model_filename



def dividir_conjunto_dados(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_modelo_forest_regressor():
    n_estimators = st.sidebar.slider(
        "Número de Árvores", 10, 500, 100, 10, key="rf_n_estimators"
    )
    max_depth = st.sidebar.slider(
        "Profundidade Máxima", 1, 50, 10, 1, key="rf_max_depth"
    )
    min_samples_split = st.sidebar.slider(
        "Mínimo de Amostras para Divisão", 2, 20, 2, 1, key="rf_min_samples_split"
    )
    min_samples_leaf = st.sidebar.slider(
        "Mínimo de Amostras por Folha", 1, 20, 1, 1, key="rf_min_samples_leaf"
    )

    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    return modelo


def get_modelo_xbg_regressor():
    learning_rate = st.sidebar.slider(
        "Taxa de Aprendizado", 0.01, 1.0, 0.1, 0.01, key="xgb_learning_rate"
    )
    n_estimators = st.sidebar.slider(
        "Número de Árvores", 10, 500, 100, 10, key="xgb_n_estimators"
    )
    max_depth = st.sidebar.slider(
        "Profundidade Máxima", 1, 50, 6, 1, key="xgb_max_depth"
    )
    subsample = st.sidebar.slider(
        "Subamostragem", 0.1, 1.0, 1.0, 0.1, key="xgb_subsample"
    )
    colsample_bytree = st.sidebar.slider(
        "Amostragem de Colunas", 0.1, 1.0, 1.0, 0.1, key="xgb_colsample_bytree"
    )

    modelo = XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
    )
    return modelo


def salvar_json_entrada_exemplo(uploaded_file, sample_input):
    json_filename = f"entrada_{os.path.splitext(uploaded_file.name)[0]}.json"

    with open(json_filename, "w") as f:
        json.dump(sample_input, f, indent=4)
