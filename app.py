import pickle
import streamlit as st
import pandas as pd
from preprocess import preprocess_data
from training import train_model, evaluate_model, save_model, analyze_feature_importance
from config import get_settings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

settings = get_settings()

st.set_page_config(
    page_title="MODELAGEM - BRMS",
    page_icon="🎈",
    layout="wide",
)

st.sidebar.title("Menu")
menu_option = st.sidebar.radio(
    "Selecione uma opção",
    ["Treinamento", "Comparação", "Previsão", "Deploy", "Download"]
)

if menu_option == "Treinamento":
    st.title("📊 Treinamento do Modelo")
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';', engine='python')
        
       
        target_column = st.selectbox("Selecione a coluna alvo", df.columns)

        df, label_encoders, scaler = preprocess_data(df, target_column)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Divisão embaralhada dos dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

        # Visualização da distribuição da variável alvo
        st.subheader("Distribuição da Variável Alvo")
        fig, ax = plt.subplots()
        sns.histplot(y, kde=True, ax=ax)
        st.pyplot(fig)

        # Análise de Importância das Variáveis
        st.subheader("📈 Importância das Variáveis")
        importance_df = analyze_feature_importance(X_train, y_train)
        if not importance_df.empty:
                st.dataframe(importance_df)

                fig, ax = plt.subplots()
                sns.barplot(x='Importancia', y='Variavel', data=importance_df.head(10), palette='viridis', ax=ax)
                ax.set_title(f"Importância das Variáveis")
                st.pyplot(fig)
        else:
                st.warning("O modelo selecionado não suporta análise de importância de variáveis.")
       
        model_type = st.sidebar.selectbox("Escolha o Modelo", ["Random Forest", "XGBoost", "Gradient Boosting", "Linear Regression"])
        optimize_params = st.sidebar.checkbox("Otimizar Hiperparâmetros", value=False)

        if st.sidebar.button("Treinar Modelo"):
            if optimize_params:
                model, best_params = train_model(X_train, y_train, model_type=model_type)
                st.write(f"Melhores hiperparâmetros: {best_params}")
            else:
                model, _ = train_model(X_train, y_train, model_type=model_type, param_grid={})

            mae, mse, r2, mean_cv_mse = evaluate_model(model, X_test, y_test)

            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**MSE:** {mse:.2f}")
            st.write(f"**R²:** {r2:.2f}")
            st.write(f"**Validação Cruzada MSE Médio:** {mean_cv_mse:.2f}")


if menu_option == "Comparação":
    st.title("📊 Comparação de Modelos")
    uploaded_file = st.file_uploader("Faça o upload do arquivo CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=';', engine='python')
        target_column = st.selectbox("Selecione a coluna alvo", df.columns)
        df, label_encoders, scaler = preprocess_data(df, target_column)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        modelos = ["Random Forest", "XGBoost", "Gradient Boosting", "Linear Regression"]
        modelos_selecionados = st.multiselect("Escolha os modelos para comparar", modelos, default=modelos[:2])

        resultados = {}

        if st.button("Comparar Modelos"):
            for modelo in modelos_selecionados:
                model, _ = train_model(X, y, model_type=modelo, param_grid={})
                mae, mse, r2, _ = evaluate_model(model, X, y)
                resultados[modelo] = {"MAE": mae, "MSE": mse, "R²": r2}

            resultados_df = pd.DataFrame(resultados).T
            st.dataframe(resultados_df)

            st.subheader("📊 Desempenho dos Modelos")
            fig, ax = plt.subplots()
            resultados_df.plot(kind='bar', ax=ax)
            st.pyplot(fig)

if menu_option == "Previsão":
    st.title("📊 Previsão com Modelo Treinado")
    modelo_file = st.file_uploader("Carregue o modelo treinado", type=["pkl"])
    input_file = st.file_uploader("Carregue os dados para previsão", type=["csv"])

    if modelo_file and input_file:
        model = pickle.load(modelo_file)
        df = pd.read_csv(input_file, sep=';', engine='python')
        df, label_encoders, scaler = preprocess_data(df)
        previsoes = model.predict(df)
        previsoes = scaler.inverse_transform(previsoes)

        st.write("### Previsões")
        st.dataframe(pd.DataFrame(previsoes, columns=["Previsão"]))

if menu_option == "Deploy":
    st.title("📊 Deploy de Modelos Treinados")
    modelo_file = st.file_uploader("Selecione o modelo para deploy", type=["pkl"])

    if modelo_file and st.button("Enviar Modelo"):
        import requests
        files = {"file": modelo_file}
        response = requests.post("https://sua-api.com/upload", files=files)
        if response.status_code == 200:
            st.success("Modelo enviado com sucesso!")
        else:
            st.error(f"Erro ao enviar modelo: {response.status_code}")

if menu_option == "Download":
    st.title("📊 Download de Modelos")
    import os
    modelos_disponiveis = [f for f in os.listdir(settings["DIRETORIO_MODELOS_PATH"]) if f.endswith(".pkl")]

    if modelos_disponiveis:
        model_to_download = st.selectbox("Escolha o modelo para download", modelos_disponiveis)
        with open(os.path.join(settings["DIRETORIO_MODELOS_PATH"], model_to_download), "rb") as f:
            st.download_button("Baixar Modelo", data=f, file_name=model_to_download, mime="application/octet-stream")
    else:
        st.warning("Nenhum modelo disponível para download.")