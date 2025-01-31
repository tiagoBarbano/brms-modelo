import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Carregar os dados
df = pd.read_csv("dados_seguro.csv")

# Convertendo variáveis categóricas em numéricas
label_encoders = {}
for col in ["sexo", "estado_civil", "tipo_seguro", "seguradora_anterior", "area_risco", "uso_frequente"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separando features e target
X = df.drop(columns=["preco_ideal"])
y = df["preco_ideal"]

# Dividindo entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando um modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliando o modelo
y_pred = modelo.predict(X_test)
erro = mean_absolute_error(y_test, y_pred)
print(f"Erro médio absoluto: {erro:.2f}")
