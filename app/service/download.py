import os
import streamlit as st
from app.core.config import DIRETORIO_MODELOS_PATH

def downloads():
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
