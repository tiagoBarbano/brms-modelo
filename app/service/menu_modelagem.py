import streamlit as st
from app.service.alterar_base import alterar_base
from app.service.deploy import deploy
from app.service.download import downloads
from app.service.treinamento import treinamento

st.title("MODELAGEM")

menu_items = ["Treinamento", "Alterar Base", "Deploy", "Downloads", "Configurações"]
selected = st.columns(len(menu_items))

for i, item in enumerate(menu_items):
    if selected[i].button(item):
        st.session_state["menu"] = item

# Exibir conteúdo com base na seleção
pagina = st.session_state.get("menu", "Home")

if pagina == "Treinamento":
    treinamento()
elif pagina == "Alterar Base":
    alterar_base()
elif pagina == "Deploy":
    deploy()
elif pagina == "Downloads":
    downloads()    
elif pagina == "Configurações":
    st.write("Configurações do sistema.")
    