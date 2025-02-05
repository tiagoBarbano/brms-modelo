import streamlit as st

create_page = st.Page("app/service/menu_modelagem.py", title="MODELAGEM", icon=":material/add_circle:")
delete_page = st.Page("app/service/simulacao.py", title="SIMULACAO", icon=":material/delete:")

pg = st.navigation([create_page, delete_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:", layout="wide")
pg.run()