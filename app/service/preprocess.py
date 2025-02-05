from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, target_column=None):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Codificando variáveis categóricas
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Garantir que valores sejam strings
        label_encoders[col] = le

    return df, label_encoders

def melhores_atributos(df, target_column):
    # Engenharia de Atributos: Criar novas variáveis baseadas em interações
    df['interaction_feature'] = df.select_dtypes(include=["int64", "float64"]).sum(axis=1)

    # Normalização dos dados numéricos
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    # Verificar se há colunas numéricas para normalizar
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))  # Preenchendo NaN com 0
    else:
        scaler = None  # Nenhuma coluna para normalizar