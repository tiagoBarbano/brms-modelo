import os
import streamlit as st
import pandas as pd
import time
from app.core.config import CHUNK_SIZE, UPLOAD_BASES_PATH
from app.core.utils import gerar_modelo_pickle
from app.service.preprocess import preprocess_data
from app.service.training import (
    train_model,
    evaluate_model,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def treinamento():
    st.title("📊 Treinamento do Modelo")

    df = None
    nome_arquivo = None
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type=["csv"])
    arquivos_disponiveis = ["Selecione um arquivo..."] + [
        f for f in os.listdir(UPLOAD_BASES_PATH) if f.endswith(".csv")
    ]
    arquivo_selecionado = st.selectbox(
        "📑 Escolha um arquivo para treinamento:", arquivos_disponiveis
    )

    st.write(f"{UPLOAD_BASES_PATH}/{arquivo_selecionado}")

    if uploaded_file:
        nome_arquivo = uploaded_file.name
        processed_chunks = pd.read_csv(
            uploaded_file, sep=";", engine="python", chunksize=CHUNK_SIZE
        )
        df = pd.concat(processed_chunks, ignore_index=True)

    if arquivo_selecionado and arquivo_selecionado != "Selecione um arquivo...":
        nome_arquivo = arquivo_selecionado
        processed_chunks = pd.read_csv(
            os.path.join(UPLOAD_BASES_PATH, arquivo_selecionado),
            delimiter=";",
            engine="python",
            chunksize=CHUNK_SIZE,
        )
        df = pd.concat(processed_chunks, ignore_index=True)

    if df is not None:
        df, label_encoders = preprocess_data(df)
        caminho_salvar = os.path.join(UPLOAD_BASES_PATH, nome_arquivo)
        df.to_csv(caminho_salvar, index=False, sep=";")
        target_column = st.selectbox("Selecione a coluna alvo", df.columns)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Divisão embaralhada dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        # Visualização da distribuição da variável alvo
        st.subheader("Distribuição da Variável Alvo")
        fig, ax = plt.subplots()
        sns.histplot(y, kde=True, ax=ax)
        st.pyplot(fig)

        model_type = st.sidebar.selectbox(
            "Escolha o Modelo",
            ["Random Forest", "XGBoost", "Gradient Boosting", "Linear Regression"],
        )
        optimize_params = st.sidebar.checkbox("Otimizar Hiperparâmetros", value=False)

        if st.sidebar.button("Treinar Modelo"):
            start_treinamento = time.perf_counter()
            if optimize_params:
                model, best_params = train_model(
                    X_train, y_train, model_type=model_type
                )
                st.write(f"Melhores hiperparâmetros: {best_params}")
            else:
                model, _ = train_model(
                    X_train, y_train, model_type=model_type, param_grid={}
                )

            mae, mse, r2, mean_cv_mse = evaluate_model(model, X_test, y_test)

            finish_treinamento = time.perf_counter()
            tempo_treinamento = finish_treinamento - start_treinamento
            st.write(f"### Tempo de Treinamento: {tempo_treinamento:.2f} segundos")

            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**MSE:** {mse:.2f}")
            st.write(f"**R²:** {r2:.2f}")
            st.write(f"**Validação Cruzada MSE Médio:** {mean_cv_mse:.2f}")

            gerar_modelo_pickle(
                uploaded_file=nome_arquivo,
                modelo=model,
                modelo_selecionado=model_type,
                mae=mae,
                mse=mse,
                r2=r2,
                tempo_treinamento=tempo_treinamento,
            )
