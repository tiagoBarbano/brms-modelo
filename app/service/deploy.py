import os
import requests
import streamlit as st
from app.core.config import DIRETORIO_MODELOS_PATH

def deploy():
    API_URL = "https://sua-api.com/upload"

    st.title("📊 Deploy de Modelos Treinados")

    modelo_disponivel = [ file for file in os.listdir(DIRETORIO_MODELOS_PATH) if file.endswith(".pkl") ]

    if modelo_disponivel:
        model_filename = st.selectbox("Escolha o modelo para deploy", modelo_disponivel)
        modelo_escolhido_path = os.path.join(DIRETORIO_MODELOS_PATH, f"{model_filename}")

        if st.button("📤 Enviar Modelo para API"):
            try:
                files = {"modelo": open(modelo_escolhido_path, "rb")}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    st.success("✅ Arquivos enviados com sucesso!")
                else:
                    st.error(f"❌ Erro ao enviar: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Erro ao enviar arquivos: {str(e)}")
    else:
        st.warning("Nenhum modelo disponível para deploy.")