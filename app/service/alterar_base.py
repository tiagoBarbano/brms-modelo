from datetime import datetime
import gc
import os
import streamlit as st
import pandas as pd
from app.core.config import UPLOAD_BASES_PATH
from app.core.utils import load_upload_file
from app.service.preprocess import preprocess_data
from app.service.training import (
    train_model,
    evaluate_model,
    analyze_feature_importance,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def alterar_base():
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
                chunksize = 1000
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

                st.subheader(f"📝 Edite os dados - Pedaço {chunk_index + 1} de {num_chunks}")

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
                    file_path_save = os.path.join(UPLOAD_BASES_PATH, f"{id}_{arquivo_selecionado}")
                    full_df.to_csv(file_path_save, index=False, sep=";")
                    st.success("✅ Alterações salvas com sucesso! Pronto para treinamento.")

            except Exception as e:
                st.error(
                    f"Erro ao ler o arquivo: O delimitador deve ser ';' - {str(e)}"
                )
            finally:
                # Limpa a memória
                del df
                gc.collect()