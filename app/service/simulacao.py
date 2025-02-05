import streamlit as st
import pandas as pd
import requests
import json

st.title("Processador de JSON e API")

# Upload do arquivo JSON
uploaded_file = st.file_uploader("Envie um arquivo JSON", type="json")

# Input para URL da API
api_url = st.text_input("Informe a URL da API")

# Botão para processar
if uploaded_file and api_url:
    st.success("Arquivo e URL carregados!")
    
    # Ler o JSON
    file_content = json.load(uploaded_file)

    # Inicializar DataFrame para armazenar resultados
    results = []

    # Enviar para API
    with st.spinner("Enviando requisição..."):
        response = requests.post(api_url, json=file_content)

        if response.status_code == 200:
            data = response.json()
            results.append(data)
        else:
            st.error(f"Erro na API: {response.status_code}")
    
    # Criar DataFrame acumulado
    if results:
        df = pd.DataFrame(results)
        st.write("🔹 **Dados Acumulados:**", df)

        # Gerar um gráfico se houver valores numéricos
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if not numeric_cols.empty:
            st.line_chart(df[numeric_cols])
