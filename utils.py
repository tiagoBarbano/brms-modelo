import json
import os
import pickle
import pandas as pd
from sklearn.calibration import LabelEncoder
import streamlit as st

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from config import get_settings, UPLOAD_BASES_PATH
from model import HistoricoModel
from repository_historico_modelo import insert_historico_modelo


settings = get_settings()


def preprocess_data(df, target_column=None):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_column in categorical_cols and target_column is not None:
        categorical_cols.remove(target_column)

    # df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    # Inicializa o LabelEncoder
    label_encoder = LabelEncoder()

    # Aplica o LabelEncoder a cada coluna categórica
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])    
    
    df.columns = [
        col.replace("[", "")
        .replace("]", "")
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "_")
        for col in df.columns
    ]

    return df


def validar_coluna_alvo(df):
    if df is not None:
        target_column = st.sidebar.selectbox("Escolha a variável alvo", df.columns)
    else:
        st.warning(
            "⚠️ Nenhum arquivo foi carregado. Por favor, faça o upload de um arquivo CSV."
        )
        st.stop()
        target_column=''
    return target_column


def load_upload_file():
    st.sidebar.write("O arquivo dever ser um CSV delimitador por ';'!!!")
    uploaded_file = st.sidebar.file_uploader("Faça upload de um arquivo CSV", type=["csv"])
    arquivos_disponiveis = ["Selecione um arquivo..."] + [
        f for f in os.listdir(UPLOAD_BASES_PATH) if f.endswith(".csv")
    ]

    # Exibir opção de seleção apenas se houver arquivos disponíveis
    arquivo_selecionado = None
    if arquivos_disponiveis:
        arquivo_selecionado = st.sidebar.selectbox(
            "📂 Ou selecione um arquivo já publicado", arquivos_disponiveis
        )

    # Carregar os dados com base na escolha do usuário
    df = None
    nome_arquivo = None
    if uploaded_file:
        nome_arquivo = uploaded_file.name
        try:
            df = pd.read_csv(uploaded_file, delimiter=";")
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: O delimitardor deve ser ';' - {str(e)}")
            st.stop()

        caminho_salvar = os.path.join(UPLOAD_BASES_PATH, uploaded_file.name)
        with open(caminho_salvar, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.info(f"✅ Arquivo carregado com sucesso e 📁 salvo em: {caminho_salvar}")
    elif arquivo_selecionado and arquivo_selecionado != "Selecione um arquivo...":
        nome_arquivo = arquivo_selecionado
        df = pd.read_csv(os.path.join(UPLOAD_BASES_PATH, arquivo_selecionado), delimiter=";")

        st.sidebar.success(f"📂 Arquivo {arquivo_selecionado} carregado!")
    
    return uploaded_file, df, nome_arquivo


def gerar_modelo_pickle(
    uploaded_file, X_train, modelo, modelo_selecionado, mae, mse, r2
):
    # modelo_salvo = {"modelo": modelo, "features": X_train.columns.tolist()}

    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join(
        "modelos_historicos",
        f"modelo_{modelo_selecionado}_{os.path.splitext(uploaded_file)[0]}_{data_hora}.pkl",
    )

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

    st.sidebar.success(
        f"✅ Modelo salvo como {model_filename} e histórico atualizado! - {id_insert}"
    )
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
    json_filename = f"entrada_{os.path.splitext(uploaded_file)[0]}.json"

    with open(json_filename, "w") as f:
        json.dump(sample_input, f, indent=4)
