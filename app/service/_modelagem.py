import gc
import numpy as np
import requests
import json
import os
import time
import xgboost as xgb

from datetime import datetime
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
)

from config import (
    get_settings,
    UPLOAD_BASES_PATH,
    DIRETORIO_MODELOS_PATH,
    MODELOS_CUSTOM_PATH,
    CHUNK_SIZE,
)
from app.models.model import Model
from repository_historico_modelo import find_all_historico_modelo
from repository_modelos import listar_modelos, salvar_modelo
from app.core.utils import (
    conn,
    get_modelo_ramdom_forest_regressor,
    get_modelo_xbg_regressor,
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

menu_option = st.sidebar.radio(
    "Selecione uma opção",
    [
        "Treinamento",
        #"Comparação",
        "Modelo Custom",
        "Alterar Dados",
        "Previsão",
        "Deploy",
        "Download",
    ],
)

if menu_option == "Treinamento":
    mae = ""
    mse = ""
    r2 = ""

    st.title("📊 Treinamento do Modelo")

    uploaded_file, nome_arquivo, column_names = load_upload_file()
    target_column = st.sidebar.selectbox("Escolha a variável alvo", column_names)

    st.sidebar.header("Configurações do Modelo")
    modelo_selecionado = st.sidebar.selectbox("Escolha o Modelo", ["Random Forest", "XGBoost",  "XGBoost-Lote", "Custom"])

    if modelo_selecionado == "Random Forest":
        modelo = get_modelo_ramdom_forest_regressor()
    elif modelo_selecionado == "XGBoost":
        modelo = get_modelo_xbg_regressor()
    elif modelo_selecionado == "XGBoost-Lote":
        modelo = get_modelo_xbg_regressor()        
    elif modelo_selecionado == "Custom":
        modelos_custom = listar_modelos()

        if modelos_custom:
            df_modelos = pd.DataFrame(modelos_custom)
            modelos = [f"{m['arquivo_pickle']}" for m in modelos_custom]

            model_filename = st.sidebar.selectbox("Escolha o modelo para treinamento", modelos)

            with open(model_filename, "rb") as f:
                modelo = pickle.load(f)

    if nome_arquivo is not None:
        scaler = StandardScaler()
        test_size = 0.2

        accumulated_predictions = []
        accumulated_actuals = []
        accumulated_X_test = []

        if st.sidebar.button("Treinar Modelo"):
            start_treinamento = time.perf_counter()

            is_incremental = hasattr(modelo, "partial_fit")
            is_xgboost = True if modelo_selecionado == "XGBoost-Lote" else False
            
            if not is_incremental and not is_xgboost:
                st.write("Treinamento NÃO incremental. Ajustando o modelo para o conjunto de dados inteiro.")

                query = f"SELECT * FROM '{nome_arquivo}'"
                df_full = conn.execute(query).fetchdf()

                X_full, y_full = df_full.drop(columns=[target_column]), df_full[target_column]

                X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=42)

                sample_input = X_train.iloc[0].to_dict()
                salvar_json_entrada_exemplo(nome_arquivo, sample_input)

                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                accumulated_predictions.extend(y_pred)
                accumulated_actuals.extend(y_test)
                accumulated_X_test.extend(X_test)
                
                del df_full, X_train, X_test, y_train, y_test 
                
            elif is_xgboost:
                st.write("Treinamento incremental em lote para XGBoost.")
                offset = 0
                first_chunk = True
                save_example = True
                booster = None
                params = modelo.get_params()

                while True:
                    query = f"SELECT * FROM '{nome_arquivo}' LIMIT {CHUNK_SIZE} OFFSET {offset}"
                    df_chunk = conn.execute(query).fetchdf()

                    if df_chunk.empty:
                        break

                    X_chunk, y_chunk = df_chunk.drop(columns=[target_column]), df_chunk[target_column]

                    if save_example:
                        sample_input = X_chunk.iloc[0].to_dict()
                        salvar_json_entrada_exemplo(nome_arquivo, sample_input)
                        save_example = False
                    
                    if first_chunk:
                        scaler.fit(X_chunk)
                        first_chunk = False

                    X_chunk = scaler.transform(X_chunk)
                    X_chunk = np.nan_to_num(X_chunk, nan=0.0, posinf=1e10, neginf=-1e10)
                    y_chunk = np.nan_to_num(y_chunk, nan=0.0, posinf=1e10, neginf=-1e10)                
                    limite_superior = np.percentile(y_chunk, 99.9)
                    y_chunk = np.clip(y_chunk, a_min=None, a_max=limite_superior)

                    X_train_chunk, X_test_chunk, y_train_chunk, y_test_chunk = train_test_split(
                        X_chunk, y_chunk, test_size=test_size, random_state=42
                    )

                    dtrain = xgb.DMatrix(X_train_chunk, label=y_train_chunk)

                    if booster is None:
                        booster = xgb.train(params, dtrain, num_boost_round=10)
                    else:
                        booster = xgb.train(params, dtrain, num_boost_round=10, xgb_model=booster)

                    dtest = xgb.DMatrix(X_test_chunk)
                    y_pred_chunk = booster.predict(dtest)
                    accumulated_predictions.extend(y_pred_chunk)
                    accumulated_actuals.extend(y_test_chunk)
                    accumulated_X_test.extend(X_test_chunk)

                    offset += CHUNK_SIZE
                
                modelo = booster
                
            else:
                st.write("Treinamento incremental em lote.")
                offset = 0
                first_chunk = True
                save_example = True

                while True:
                    query = f"SELECT * FROM '{nome_arquivo}' LIMIT {CHUNK_SIZE} OFFSET {offset}"
                    df_chunk = conn.execute(query).fetchdf()

                    if df_chunk.empty:
                        break
                    
                    X_chunk, y_chunk = df_chunk.drop(columns=[target_column]), df_chunk[target_column]

                    if save_example:
                        sample_input = X_chunk.iloc[0].to_dict()
                        salvar_json_entrada_exemplo(nome_arquivo, sample_input)
                        save_example = False

                    if first_chunk:
                        scaler.fit(X_chunk)  # Ajusta o scaler na primeira iteração
                        first_chunk = False

                    X_chunk = scaler.transform(X_chunk)

                    X_train_chunk, X_test_chunk, y_train_chunk, y_test_chunk = train_test_split(
                        X_chunk, y_chunk, test_size=test_size, random_state=42
                    )

                    if hasattr(modelo, "partial_fit"):
                        modelo.partial_fit(X_train_chunk, y_train_chunk)  # Treina de forma incremental
                    else:
                        modelo.fit(X_train_chunk, y_train_chunk)  # Treina normalmente

                    y_pred_chunk = modelo.predict(X_test_chunk)
                    accumulated_predictions.extend(y_pred_chunk)
                    accumulated_actuals.extend(y_test_chunk)
                    accumulated_X_test.extend(X_test_chunk)

                    offset += CHUNK_SIZE

            gc.collect()
            
            finish_treinamento = time.perf_counter()
            tempo_treinamento = finish_treinamento - start_treinamento
            st.write(f"### Tempo de Treinamento: {tempo_treinamento:.2f} segundos")

            mae = mean_absolute_error(accumulated_actuals, accumulated_predictions)
            mse = mean_squared_error(accumulated_actuals, accumulated_predictions)
            r2 = r2_score(accumulated_actuals, accumulated_predictions)
            cv_scores = cross_val_score(modelo, accumulated_X_test, accumulated_predictions, cv=5, scoring='neg_mean_squared_error')           
            mean_cv_mse = -cv_scores.mean()
            
            # Precisão: capacidade de prever corretamente uma classe positiva            
            accumulated_predictions_int = np.round(accumulated_predictions).astype(int)
            accumulated_actuals_int = np.round(accumulated_actuals).astype(int)
            precision = precision_score(accumulated_actuals_int, accumulated_predictions_int, average="weighted")
            recall = recall_score(accumulated_actuals_int, accumulated_predictions_int, average="weighted")

            model_filename = gerar_modelo_pickle(
                uploaded_file=nome_arquivo,
                modelo=modelo,
                modelo_selecionado=modelo_selecionado,
                mae=mae,
                mse=mse,
                r2=r2,
                tempo_treinamento=tempo_treinamento,
            )

            st.write(f"### Desempenho do Modelo: {modelo_selecionado} - {nome_arquivo}")

            st.write(f"📉 **Acurácia final:** {accuracy_score(accumulated_actuals_int, accumulated_predictions_int):.5f}")
            st.write(f"📉 **Precisão:** {precision:.4f}")
            st.write(f"📉 **Recall:** {recall:.4f}")
            st.write(f"📉 **MAE:** {mae:.2f}")
            st.write(f"📉 **MSE:** {mse:.2f}")
            st.write(f"📈 **R²:** {r2:.2f}")
            st.write(f"📈 **Validação Cruzada MSE Médio:** {mean_cv_mse:.2f}")

            accumulated_results = pd.DataFrame({"Actual": accumulated_actuals, "Predicted": accumulated_predictions})
            accumulated_results.to_csv(f"accumulated_results_{nome_arquivo}.csv", index=False)

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x="Actual", y="Predicted", data=accumulated_results)

            plt.plot(
                [
                    accumulated_results["Actual"].min(),
                    accumulated_results["Actual"].max(),
                ],
                [
                    accumulated_results["Actual"].min(),
                    accumulated_results["Actual"].max(),
                ],
                color="red",
                linestyle="--",
            )

            # Adicionando rótulos e título
            plt.xlabel("Valores Reais")
            plt.ylabel("Valores Preditos")
            plt.title("Valores Reais vs. Preditos")

            st.pyplot(plt)  # Exibe o gráfico com matplotlib no Streamlit
            
            del accumulated_predictions, accumulated_actuals
            gc.collect()

if menu_option == "Alterar Dados":
    st.subheader("📊 Alterar Base")

    # Listar arquivos disponíveis para edição
    arquivos_disponiveis = ["Selecione um arquivo..."] + [
        f for f in os.listdir(UPLOAD_BASES_PATH) if f.endswith(".csv")
    ]

    if arquivos_disponiveis:
        arquivo_selecionado = st.selectbox(
            "📑 Escolha um arquivo para editar:", arquivos_disponiveis
        )

        if arquivo_selecionado and arquivo_selecionado != "Selecione um arquivo...":
            file_path = os.path.join(UPLOAD_BASES_PATH, arquivo_selecionado)
            try:
                # Carregar o arquivo em pedaços
                chunksize = (
                    1000  # Define o tamanho do pedaço (exemplo: 1000 linhas por vez)
                )
                chunk_list = []
                for chunk in pd.read_csv(file_path, delimiter=";", chunksize=chunksize):
                    chunk_list.append(chunk)

                # Exibir apenas um pedaço por vez
                num_chunks = len(chunk_list)
                if num_chunks > 1:
                    chunk_index = st.slider(
                        "Escolha o pedaço a ser editado", 0, num_chunks - 1, 0
                    )
                else:
                    chunk_index = 0

                df = chunk_list[chunk_index]

                st.subheader(
                    f"📝 Edite os dados - Pedaço {chunk_index + 1} de {num_chunks}"
                )

                # Permite edição na interface
                edited_df = st.data_editor(df, num_rows="dynamic")

                id = datetime.now().strftime("%H%M%S")

                # Botão para salvar alterações
                if st.button("💾 Salvar Alterações"):
                    # Carregar o arquivo completo de volta ou apenas o pedaço alterado, se necessário
                    df_start = chunk_list[:chunk_index]
                    df_end = chunk_list[chunk_index + 1 :]
                    full_df = pd.concat(
                        df_start + [edited_df] + df_end, ignore_index=True
                    )

                    # Salvar no arquivo original
                    full_df.to_csv(file_path, index=False, sep=";")
                    st.success(
                        "✅ Alterações salvas com sucesso! Pronto para treinamento."
                    )

            except Exception as e:
                st.error(
                    f"Erro ao ler o arquivo: O delimitador deve ser ';' - {str(e)}"
                )
            finally:
                # Limpa a memória
                del df
                gc.collect()

if menu_option == "Comparação":
    uploaded_file, df, nome_arquivo = load_upload_file()
    mae = ""
    mse = ""
    r2 = ""

    st.title("📊 Treinamento e Comparação de Modelos")
    target_column = validar_coluna_alvo(df)
    X_train, X_test, y_train, y_test = dividir_conjunto_dados(df, target_column)

    st.sidebar.header("Configurações do Treinamento")
    modelos_selecionados = st.sidebar.multiselect(
        "Escolha os Modelos",
        ["Random Forest", "XGBoost"],
        default=["Random Forest", "XGBoost"],
    )

    resultados = {}

    if "Random Forest" in modelos_selecionados:
        st.sidebar.subheader("🔹 Hiperparâmetros - Random Forest")
        modelo_forest = get_modelo_ramdom_forest_regressor()
    if "XGBoost" in modelos_selecionados:
        st.sidebar.subheader("🔹 Hiperparâmetros - XGBoost")
        modelo_xgb = get_modelo_xbg_regressor()

    if st.sidebar.button("Comparar Modelos"):
        for modelo_nome in modelos_selecionados:
            if modelo_nome == "Random Forest":
                modelo = modelo_forest
            elif modelo_nome == "XGBoost":
                modelo = modelo_xgb

            start_treinamento = time.perf_counter()

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            finish_treinamento = time.perf_counter()
            tempo_treinamento = finish_treinamento - start_treinamento
            st.write(
                f"### Tempo de Treinamento: {tempo_treinamento:.2f} segundos - {modelo_nome}"
            )

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            resultados[modelo_nome] = {
                "MAE": mae,
                "MSE": mse,
                "R²": r2,
                "Modelo": modelo,
                "Tempo Treinamento": tempo_treinamento,
            }

            model_filename = gerar_modelo_pickle(
                uploaded_file=nome_arquivo,
                X_train=X_train,
                modelo=modelo,
                modelo_selecionado=modelo_nome,
                mae=mae,
                mse=mse,
                r2=r2,
                tempo_treinamento=tempo_treinamento,
            )

        st.sidebar.success("Modelos treinados com sucesso! ✅")

        # Salvar no session_state para previsões futuras
        # st.session_state["modelos"] = resultados

        # Mostrar Resultados
        st.subheader("📊 Comparação de Modelos")
        df_resultados = pd.DataFrame(resultados).T.drop(columns=["Modelo"])
        st.write(df_resultados)

        # Gráfico de Comparação
        st.bar_chart(df_resultados)

if menu_option == "Previsão":
    aba_previsao, aba_consulta_historico_modelos = st.tabs(
        ["📊 Previsao", "📈 Consultar Resultados Histórico"]
    )

    with aba_previsao:
        st.title("📈 Previsão de Preço")
        st.sidebar.header("Fazer Previsão")
        entrada_modelo_disponivel = [ file for file in os.listdir() if file.endswith(".json") ]
        entrada_modelo_escolhido = st.sidebar.selectbox("Escolha o modelo para previsão", entrada_modelo_disponivel)

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

        modelo_disponivel = [ file for file in os.listdir(DIRETORIO_MODELOS_PATH) if file.endswith(".pkl")]
        modelo_escolhido = st.sidebar.selectbox("Escolha o modelo para previsão", modelo_disponivel)

        if st.sidebar.button("Prever Preço"):
            try:
                modelo_escolhido_path = os.path.join(DIRETORIO_MODELOS_PATH, f"{modelo_escolhido}")
                with open(modelo_escolhido_path, "rb") as f:
                    modelo_previsao = pickle.load(f)

                entrada_df = pd.DataFrame([entrada_usuario])

                # Garantir que as colunas estão na mesma ordem e faltantes são preenchidas com 0
                entrada_df = entrada_df.reindex(columns=df.columns, fill_value=0)

                # if "XGBoos" in modelo_escolhido:
                #     entrada_df = xgb.DMatrix(entrada_df)  # Convertendo X_test para DMatrix

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
                st.error(f"❌ Erro ao enviar: {response.status_code} - {response.text}")

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
