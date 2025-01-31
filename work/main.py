import streamlit as st

st.set_page_config(
    page_title="MODELAGEM - BRMS",
    page_icon="🧊",
    layout="wide",
    menu_items={
        "About": "© 2024 - Desenvolvido com 💙 por PSG - SISTEMAS ANALITICOS AUTO NOVOS SEG | Versão: 1.0.0!"
    },
)

logo_path = "images/bannerlogo"  # Altere para o caminho correto do seu logo
st.sidebar.image(logo_path, use_column_width=True)
para_campos_page = st.Page("app_03.py", title="Campos", icon=":material/delete:")

api = st.Page("app.py", title="Início", icon=":material/logout:")
# consulta_page = st.Page("app_01.py", title="Simulação", icon=":material/add_circle:")
# compara_page = st.Page("app_02.py", title="Arquivos", icon=":material/delete:")
# compara_campos_page = st.Page("app_03.py", title="Campos", icon=":material/delete:")

pg = st.navigation(
    {
        "Início": [api],
    }
)
pg.run()  # Executa a navegação entre as páginas
