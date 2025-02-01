from datetime import datetime
import requests
import json
import os
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    get_settings,
    UPLOAD_BASES_PATH,
    DIRETORIO_MODELOS_PATH,
    MODELOS_CUSTOM_PATH,
)
from model import Model
from repository_historico_modelo import find_all_historico_modelo
from repository_modelos import listar_modelos, salvar_modelo
from utils import (
    get_modelo_forest_regressor,
    get_modelo_xbg_regressor,
    preprocess_data,
    salvar_json_entrada_exemplo,
    validar_coluna_alvo,
    gerar_modelo_pickle,
    dividir_conjunto_dados,
    load_upload_file,
)

settings = get_settings()

st.set_page_config(
    page_title="MODELAGEM - BRMS",
    page_icon="🧊",
    layout="wide",
    menu_items={
        "About": "© 2024 - Desenvolvido com 💙 por PSG - SISTEMAS ANALITICOS AUTO NOVOS SEG | Versão: 1.0.0!"
    },
)

# Criar colunas para centralizar o conteúdo
col1, col2, col3 = st.columns([1, 3, 1])  # A coluna do meio é maior

with col2:  # Centraliza os elementos
    c1, c2 = st.columns([1, 4])  # Criar colunas internas para alinhar lado a lado
    with c1:
        st.image("images/brms-logo.png", width=150)
    with c2:
        st.title("BRMS - MODELAGEM")

uploaded_file, df, nome_arquivo = load_upload_file()

menu_option = st.sidebar.radio(
    "Selecione uma opção",
    ["Treinamento", "Comparação", "Modelo Custom", "Previsão", "Deploy", "Download"],
)

if menu_option == "Treinamento":
    mae = ""
    mse = ""
    r2 = ""

    st.title("📊 Treinamento do Modelo")
    aba_dataset, aba_modelo, aba_alterar_base = st.tabs(
        ["📊 Análise do Dataset", "📈 Avaliação do Modelo", "📈 Alterar Base"]
    )

    target_column = validar_coluna_alvo(df)
    df, categorical_cols = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = dividir_conjunto_dados(df, target_column)

    sample_input = X_train.iloc[0].to_dict()
    salvar_json_entrada_exemplo(nome_arquivo, sample_input)

    st.sidebar.header("Configurações do Modelo")
    modelo_selecionado = st.sidebar.selectbox(
        "Escolha o Modelo", ["Random Forest", "XGBoost", "Custom"]
    )

    if modelo_selecionado == "Random Forest":
        modelo = get_modelo_forest_regressor()
    elif modelo_selecionado == "XGBoost":
        modelo = get_modelo_xbg_regressor()
    elif modelo_selecionado == "Custom":
        modelos_custom = listar_modelos()

        # Se houver modelos, permitir seleção e download
        if modelos_custom:
            df_modelos = pd.DataFrame(modelos_custom)
            modelos = [f"{m['arquivo_pickle']}" for m in modelos_custom]

            model_filename = st.sidebar.selectbox(
                "Escolha o modelo para treinamento", modelos
            )

            with open(model_filename, "rb") as f:
                modelo = pickle.load(f)

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
                uploaded_file=nome_arquivo,
                X_train=X_train,
                modelo=modelo,
                modelo_selecionado=modelo_selecionado,
                mae=mae,
                mse=mse,
                r2=r2,
            )

        st.write(f"### Desempenho do Modelo: {modelo_selecionado} - {nome_arquivo}")
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

    with aba_alterar_base:
        st.subheader("📊 Alterar Base")

        # Listar arquivos disponíveis para edição
        arquivos_disponiveis = [
            f for f in os.listdir(UPLOAD_BASES_PATH) if f.endswith(".csv")
        ]

        if arquivos_disponiveis:
            arquivo_selecionado = st.selectbox(
                "📑 Escolha um arquivo para editar:", arquivos_disponiveis
            )

            if arquivo_selecionado:
                file_path = os.path.join(UPLOAD_BASES_PATH, arquivo_selecionado)
                # Detectar delimitador automaticamente ("," ou ";")
                try:
                    df = pd.read_csv(file_path, delimiter=",")
                    if df.shape[1] == 1:  # Se tiver apenas 1 coluna, tenta ";"
                        df = pd.read_csv(file_path, delimiter=";")
                except Exception as e:
                    st.error(f"Erro ao ler o arquivo: {e}")
                    df = None

                if df is not None:
                    st.subheader("📝 Edite os dados antes do treinamento")
                    edited_df = st.data_editor(
                        df, num_rows="dynamic"
                    )  # Permite edição na interface

                    # Botão para salvar alterações
                    if st.button("💾 Salvar Alterações"):
                        new_file_path = os.path.join(
                            UPLOAD_BASES_PATH, f"v2_{arquivo_selecionado}"
                        )
                        edited_df.to_csv(new_file_path, index=False)
                        st.success(
                            "✅ Alterações salvas com sucesso! Pronto para treinamento."
                        )

if menu_option == "Comparação":
    mae = ""
    mse = ""
    r2 = ""

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
                uploaded_file=nome_arquivo,
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
    aba_previsao, aba_consulta_historico_modelos = st.tabs(
        ["📊 Previsao", "📈 Consultar Resultados Histórico"]
    )

    with aba_previsao:
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
                entrada_usuario[col] = st.number_input(f"{col}", value=float(df[col].iloc[0]))

        modelo_disponivel = [
            file for file in os.listdir(DIRETORIO_MODELOS_PATH) if file.endswith(".pkl")
        ]
        modelo_escolhido = st.sidebar.selectbox(
            "Escolha o modelo para previsão", modelo_disponivel
        )

        if st.sidebar.button("Prever Preço"):
            try:
                modelo_escolhido_path = os.path.join(
                    DIRETORIO_MODELOS_PATH, f"{modelo_escolhido}"
                )
                with open(modelo_escolhido_path, "rb") as f:
                    modelo_data = pickle.load(f)

                modelo_previsao = modelo_data
                # feature_names = modelo_data["features"]  # Pegando os nomes das features

                entrada_df = pd.DataFrame([entrada_usuario])

                # Garantir que as colunas estão na mesma ordem e faltantes são preenchidas com 0
                entrada_df = entrada_df.reindex(columns=df.columns, fill_value=0)

                previsao = modelo_previsao.predict(entrada_df)[0]
                st.sidebar.success(f"💰 Preço Sugerido: R$ {previsao:.2f}")

            except FileNotFoundError:
                st.sidebar.error(
                    "O modelo selecionado ainda não foi treinado! Treine primeiro."
                )

    with aba_consulta_historico_modelos:
        st.title("📈 Consultar Modelos")
        st.write("Aqui voce pode consultar os modelos cadastrados.")

        modelos_disponiveis = find_all_historico_modelo()

        df = pd.DataFrame(modelos_disponiveis)
        st.dataframe(df)  # Exibir tabela no Streamlit

if menu_option == "Deploy":
    API_URL = "https://sua-api.com/upload"

    st.title("📊 Deploy de Modelos Treinados")
    
    modelo_disponivel = [
        file for file in os.listdir(DIRETORIO_MODELOS_PATH) if file.endswith(".pkl")
    ]

    if modelo_disponivel:
        model_filename = st.selectbox("Escolha o modelo para deploy", modelo_disponivel)
        modelo_escolhido_path = os.path.join(
            DIRETORIO_MODELOS_PATH, f"{model_filename}"
        )

    if st.button("📤 Enviar Modelo para API"):
        try:
            files = {"modelo": open(model_filename, "rb")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                st.success("✅ Arquivos enviados com sucesso!")
            else:
                st.error(
                    f"❌ Erro ao enviar: {response.status_code} - {response.text}"
                )

        except Exception as e:
            st.error(f"Erro ao enviar arquivos: {str(e)}")

if menu_option == "Download":
    
    st.title("📊 Download de Modelos Treinados")
    
    modelo_disponivel = [
        file for file in os.listdir(DIRETORIO_MODELOS_PATH) if file.endswith(".pkl")
    ]

    # Se houver modelos, permitir seleção e download
    if modelo_disponivel:
        model_filename = st.selectbox(
            "Escolha o modelo para previsão", modelo_disponivel
        )
        modelo_escolhido_path = os.path.join(
            DIRETORIO_MODELOS_PATH, f"{model_filename}"
        )

        with open(modelo_escolhido_path, "rb") as f:
            st.download_button(
                label="Baixar Modelo",
                data=f,
                file_name=model_filename,
                mime="application/octet-stream",
            )
    else:
        st.warning("Nenhum modelo disponível para download.")

if menu_option == "Modelo Custom":
    aba_cadastro_modelo, aba_consulta_modelo = st.tabs(
        ["📊 Cadastrar Modelo", "📈 Consultar Modelos"]
    )

    with aba_cadastro_modelo:
        st.title("📊 Cadastrar Modelo")
        st.write("Aqui voce pode cadastrar um novo modelo de previsão.")

        uploaded_model = st.file_uploader("Upload de Modelo Pickle", type=["pkl"])
        nome_modelo = st.text_input("Nome do Modelo")
        tipo_modelo = st.selectbox(
            "Tipo do Modelo",
            [
                "Custom",
            ],
        )

        if uploaded_model and nome_modelo:
            file_path_modelo_custom = os.path.join(
                MODELOS_CUSTOM_PATH, f"{uploaded_model.name}"
            )
            with open(file_path_modelo_custom, "wb") as f:
                f.write(uploaded_model.read())

            dados_insert = Model(
                nome=nome_modelo,
                tipo=tipo_modelo,
                arquivo_pickle=file_path_modelo_custom,
                metricas=json.dumps({"MAE": 0, "MSE": 0, "R²": 0}),
                data_treinamento=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            salvar_modelo(dados_insert)
            st.sidebar.success(f"Modelo '{nome_modelo}' salvo com sucesso! ✅")

    with aba_consulta_modelo:
        st.title("📈 Consultar Modelos")
        st.write("Aqui voce pode consultar os modelos cadastrados.")

        modelos_disponiveis = listar_modelos()

        df = pd.DataFrame(modelos_disponiveis)
        st.dataframe(df)  # Exibir tabela no Streamlit
