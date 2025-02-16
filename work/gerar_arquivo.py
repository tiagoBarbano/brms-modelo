import pandas as pd
import numpy as np

# Gerando dados fictícios
np.random.seed(42)

n = 1000  # Número de registros

df = pd.DataFrame({
    "idade": np.random.randint(18, 70, n),
    "sexo": np.random.choice(["Masculino", "Feminino"], n),
    "estado_civil": np.random.choice(["Solteiro", "Casado", "Viúvo"], n),
    "renda_anual": np.random.randint(20000, 200000, n),
    "valor_atual_apolice": np.random.randint(500, 5000, n),
    "numero_sinistros": np.random.randint(0, 5, n),
    "tipo_seguro": np.random.choice(["Auto", "Residencial", "Saúde", "Vida"], n),
    "tempo_como_cliente": np.random.randint(1, 20, n),
    "valor_bem": np.random.randint(5000, 200000, n),
    "ano_fabricacao": np.random.randint(1990, 2023, n),
    "seguradora_anterior": np.random.choice(["Sim", "Não"], n),
    "area_risco": np.random.choice(["Baixo", "Médio", "Alto"], n),
    "uso_frequente": np.random.choice(["Diário", "Ocasional", "Raro"], n),
    "historico_credito": np.random.randint(300, 850, n)
})

# Criando um preço-alvo baseado nos dados (simulado)
df["preco_ideal"] = (
    df["valor_atual_apolice"] * 0.8 +
    df["numero_sinistros"] * 200 -
    df["tempo_como_cliente"] * 10 +
    df["renda_anual"] * 0.001 +
    np.random.randint(-500, 500, n)
)

# Salvar o dataset para usar depois
df.to_csv("dados_seguro.csv", index=False, sep=";")
print(df.head())
