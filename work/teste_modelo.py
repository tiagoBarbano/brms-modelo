import pickle
import pandas as pd

x = {
    "idade": 44,
    "sexo": 1,
    "estado_civil": 1,
    "renda_anual": 110319,
    "valor_atual_apolice": 2973,
    "numero_sinistros": 2,
    "tipo_seguro": 1,
    "tempo_como_cliente": 10,
    "valor_bem": 102208,
    "ano_fabricacao": 2006,
    "seguradora_anterior": 1,
    "area_risco": 1,
    "uso_frequente": 1,
    "historico_credito": 576,
}

with open("modelo_treinado.pkl", "rb") as f:
    modelo = pickle.load(f)

entrada_df = pd.DataFrame([x])

previsao = modelo.predict(entrada_df)[0]

print(previsao)