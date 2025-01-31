import shap
import requests
import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from database import get_database
from config import get_settings

settings = get_settings()

st.set_page_config(
    page_title="MODELAGEM - BRMS",
    page_icon="🧊",
    layout="wide",
    menu_items={
        "About": "© 2024 - Desenvolvido com 💙 por PSG - SISTEMAS ANALITICOS AUTO NOVOS SEG | Versão: 1.0.0!"
    },
)


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
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, delimiter=";")
        df.columns = df.columns.str.strip()
        return df
    else:
        return None


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


menu_option = st.sidebar.radio(
    "Selecione uma opção",
    ["Treinamento", "Comparação", "Previsão", "Deploy", "Download"],
)

df = None
uploaded_file = st.sidebar.file_uploader("Faça upload de um arquivo CSV", type=["csv"])
df = load_data(uploaded_file)


def gerar_modelo_pickle(
    uploaded_file, X_train, modelo, modelo_selecionado, mae, mse, r2
):
    modelo_salvo = {"modelo": modelo, "features": X_train.columns.tolist()}

    data_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"modelo_{modelo_selecionado}_{os.path.splitext(uploaded_file.name)[0]}_{data_hora}.pkl"

    with open(model_filename, "wb") as f:
        pickle.dump(modelo_salvo, f)

    with get_database() as db:
        collection_model = db[settings.collection_model]

        collection_model.insert_one(
            {
                "data_treinamento": data_hora,
                "modelo": modelo_selecionado,
                "mae": mae,
                "mse": mse,
                "r2": r2,
                "arquivo": model_filename,
            }
        )

    st.sidebar.success(f"✅ Modelo salvo como {model_filename} e histórico atualizado!")
    return model_filename


@st.cache_data
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
        

mae = ""
mse = ""
r2 = ""

if menu_option == "Treinamento":
    st.title("📊 Treinamento do Modelo")
    aba_dataset, aba_modelo = st.tabs(["📊 Análise do Dataset", "📈 Avaliação do Modelo"])

    target_column = validar_coluna_alvo(df)
    df, categorical_cols = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = dividir_conjunto_dados(df, target_column)
    
    sample_input = X_train.iloc[0].to_dict()
    salvar_json_entrada_exemplo(uploaded_file, sample_input)
    
    st.sidebar.header("Configurações do Modelo")
    modelo_selecionado = st.sidebar.selectbox("Escolha o Modelo", ["Random Forest", "XGBoost"])

    if modelo_selecionado == "Random Forest":
        modelo = get_modelo_forest_regressor()

    elif modelo_selecionado == "XGBoost":
        modelo = get_modelo_xbg_regressor()

    if "modelo" not in st.session_state or st.session_state["modelo"] is None:
        st.session_state["modelo"] = {}

    with aba_dataset:
        st.subheader("📊 Visão Geral do Dataset")
        st.write(df.head())

        st.subheader("📊 Estatísticas Gerais do Dataset")
        st.write(df.describe())

        st.subheader("📊 Estatísticas por Variável")
        coluna_selecionada = st.selectbox(
            "Escolha uma variável para análise", df.columns
        )
        st.write(df[coluna_selecionada].describe())

    with aba_modelo:
        modelo.fit(X_train, y_train)
        st.session_state["modelo"][modelo_selecionado] = modelo

        y_pred = modelo.predict(X_test)
        y_pred_class = np.round(y_pred)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        modelo_salvo_info = {}
        if st.sidebar.button("Treinar Modelo"):
            model_filename = gerar_modelo_pickle(
                uploaded_file=uploaded_file,
                X_train=X_train,
                modelo=modelo,
                modelo_selecionado=modelo_selecionado,
                mae=mae,
                mse=mse,
                r2=r2,
            )
        
        st.write(f"### Desempenho do Modelo: {modelo_selecionado} - {uploaded_file.name}")
        st.write(f"📉 **MAE:** {mae:.2f}")
        st.write(f"📉 **MSE:** {mse:.2f}")
        st.write(f"📈 **R²:** {r2:.2f}")

        st.subheader("📊 Comparação de Preços Reais vs Previstos")
        df_resultados = pd.DataFrame({"Real": y_test.values, "Previsto": y_pred})
        st.line_chart(df_resultados)

        df_erros = pd.DataFrame({"Erro": y_test.values - y_pred})
        st.subheader("🔍 Erros da Previsão")
        st.bar_chart(df_erros)

        if hasattr(modelo, "feature_importances_"):
            importances = modelo.feature_importances_
            feature_names = X_train.columns
            df_importance = pd.DataFrame(
                {"Feature": feature_names, "Importância": importances}
            )
            df_importance = df_importance.sort_values(by="Importância", ascending=False)
            st.subheader("📊 Importância das Variáveis")
            st.bar_chart(df_importance.set_index("Feature"))

        # Visualização dos preços reais vs previstos
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(y_test, label="Real", color="blue", kde=True, alpha=0.5, ax=ax)
        sns.histplot(y_pred, label="Previsto", color="red", kde=True, alpha=0.5, ax=ax)
        ax.legend()
        ax.set_title("Distribuição dos preços reais vs previstos")
        st.pyplot(fig)

if menu_option == "Comparação":
    st.title("📊 Treinamento e Comparação de Modelos")
    target_column = validar_coluna_alvo(df)

    df, categorical_cols = preprocess_data(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.sidebar.header("Configurações do Treinamento")
    modelos_selecionados = st.sidebar.multiselect(
        "Escolha os Modelos",
        ["Random Forest", "XGBoost"],
        default=["Random Forest", "XGBoost"],
    )

    resultados = {}

    if "Random Forest" in modelos_selecionados:
        st.sidebar.subheader("🔹 Hiperparâmetros - Random Forest")
        modelo_forest = get_modelo_forest_regressor()

    if "XGBoost" in modelos_selecionados:
        st.sidebar.subheader("🔹 Hiperparâmetros - XGBoost")
        modelo_xgb = get_modelo_xbg_regressor()

    if st.sidebar.button("Treinar Modelos"):
        for modelo_nome in modelos_selecionados:
            if modelo_nome == "Random Forest":
                modelo = modelo_forest
            elif modelo_nome == "XGBoost":
                modelo = modelo_xgb

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            resultados[modelo_nome] = {
                "MAE": mae,
                "MSE": mse,
                "R²": r2,
                "Modelo": modelo,
            }

            model_filename = gerar_modelo_pickle(
                uploaded_file=uploaded_file,
                X_train=X_train,
                modelo=modelo,
                modelo_selecionado=modelo_nome,
                mae=mae,
                mse=mse,
                r2=r2,
            )

        st.sidebar.success("Modelos treinados com sucesso! ✅")

        # Salvar no session_state para previsões futuras
        st.session_state["modelos"] = resultados

        # Mostrar Resultados
        st.subheader("📊 Comparação de Modelos")
        df_resultados = pd.DataFrame(resultados).T.drop(columns=["Modelo"])
        st.write(df_resultados)

        # Gráfico de Comparação
        st.bar_chart(df_resultados)

        # Importância das Features
        for modelo_nome, info in resultados.items():
            if hasattr(info["Modelo"], "feature_importances_"):
                st.subheader(f"📊 Importância das Variáveis - {modelo_nome}")
                importances = info["Modelo"].feature_importances_
                df_importance = pd.DataFrame(
                    {"Feature": X_train.columns, "Importância": importances}
                )
                df_importance = df_importance.sort_values(
                    by="Importância", ascending=False
                )
                st.bar_chart(df_importance.set_index("Feature"))

if menu_option == "Previsão":
    st.title("📈 Previsão de Preço")

    st.sidebar.header("Fazer Previsão")
    entrada_modelo_disponivel = [
        file for file in os.listdir() if file.endswith(".json")
    ]
    entrada_modelo_escolhido = st.sidebar.selectbox(
        "Escolha o modelo para previsão", entrada_modelo_disponivel
    )

    if entrada_modelo_escolhido:
        try:
            with open(entrada_modelo_escolhido, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                df = pd.json_normalize(json_data)
        except Exception as e:
            st.error(f"❌ Erro ao ler JSON: {e}")
            st.stop()
    else:
        st.warning(
            "⚠️ Nenhum arquivo foi carregado. Por favor, faça o upload de um arquivo JSON."
        )
        st.stop()

    entrada_usuario = {}

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            entrada_usuario[col] = st.sidebar.number_input(
                f"{col}",
                min_value=float(df[col].min()),
                max_value=float(df[col].max()),
                value=float(df[col].median()),
            )

    modelo_disponivel = [file for file in os.listdir() if file.endswith(".pkl")]
    modelo_escolhido = st.sidebar.selectbox(
        "Escolha o modelo para previsão", modelo_disponivel
    )

    if st.sidebar.button("Prever Preço"):
        try:
            with open(modelo_escolhido, "rb") as f:
                modelo_data = pickle.load(f)

            if not isinstance(modelo_data, dict):
                raise ValueError("O arquivo do modelo não contém os dados esperados.")

            modelo_previsao = modelo_data["modelo"]
            feature_names = modelo_data["features"]  # Pegando os nomes das features

            entrada_df = pd.DataFrame([entrada_usuario])

            # Garantir que as colunas estão na mesma ordem e faltantes são preenchidas com 0
            entrada_df = entrada_df.reindex(columns=feature_names, fill_value=0)

            previsao = modelo_previsao.predict(entrada_df)[0]
            st.success(f"💰 Preço Sugerido: R$ {previsao:.2f}")

        except FileNotFoundError:
            st.error("O modelo selecionado ainda não foi treinado! Treine primeiro.")

elif menu_option == "Deploy":
    API_URL = "https://sua-api.com/upload"

    modelo_disponivel = [file for file in os.listdir() if file.endswith(".pkl")]
    model_filename = st.sidebar.selectbox(
        "Escolha o modelo para previsão", modelo_disponivel
    )

    if st.sidebar.button("📤 Enviar Modelo e JSON para API"):
        try:
            files = {"modelo": open(model_filename, "rb")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                st.sidebar.success("✅ Arquivos enviados com sucesso!")
            else:
                st.sidebar.error(
                    f"❌ Erro ao enviar: {response.status_code} - {response.text}"
                )

        except Exception as e:
            st.sidebar.error(f"Erro ao enviar arquivos: {str(e)}")

elif menu_option == "Download":
    modelo_disponivel = [file for file in os.listdir() if file.endswith(".pkl")]

    # Se houver modelos, permitir seleção e download
    if modelo_disponivel:
        model_filename = st.sidebar.selectbox(
            "Escolha o modelo para previsão", modelo_disponivel
        )

        with open(model_filename, "rb") as f:
            st.sidebar.download_button(
                label="Baixar Modelo",
                data=f,
                file_name=model_filename,
                mime="application/octet-stream",
            )
    else:
        st.sidebar.warning("Nenhum modelo disponível para download.")
