import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Configuração da Página
st.set_page_config(
    page_title="Preditor do Titanic", 
    page_icon="🚢", 
    layout="centered"
)

# 2. Treinamento do Modelo (Cacheado para performance local)
@st.cache_resource
def treinar_modelo():
    """
    Carrega os dados, trata valores nulos, padroniza as features 
    e treina o modelo de Regressão Logística.
    """
    df = sns.load_dataset('titanic')
    
    # Tratamento de nulos
    df['age'] = df['age'].fillna(df['age'].median())
    
    # Seleção de features e criação de dummies
    df = df[['survived', 'pclass', 'sex', 'age', 'fare']]
    df = pd.get_dummies(df, columns=['sex'], drop_first=True)

    X = df.drop('survived', axis=1)
    y = df['survived']

    # Padronização (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treinamento
    modelo = LogisticRegression()
    modelo.fit(X_scaled, y)

    return modelo, scaler, X.columns

modelo, scaler, colunas = treinar_modelo()

# 3. Construção da Interface Web (Streamlit)
st.title("🚢 Preditor de Sobrevivência do Titanic")
st.markdown("Ajuste os parâmetros abaixo para simular a probabilidade de sobrevivência de um passageiro com base em um modelo de **Machine Learning (Regressão Logística)** treinado com dados históricos.")

st.divider()

# Layout em colunas
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Classe do Passageiro", [1, 2, 3], format_func=lambda x: f"{x}ª Classe")
    sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])

with col2:
    idade = st.slider("Idade", min_value=0, max_value=100, value=30)
    tarifa = st.number_input("Preço da Passagem (Fare)", min_value=0.0, value=30.0, step=5.0)

# 4. Processamento dos Novos Dados Inseridos
is_male = 1 if sexo == "Masculino" else 0

# Estruturando a entrada conforme o modelo espera
novo_passageiro = pd.DataFrame([[pclass, idade, tarifa, is_male]], columns=colunas)
novo_passageiro_scaled = scaler.transform(novo_passageiro)

# 5. Predição
# predict_proba retorna [probabilidade_classe_0, probabilidade_classe_1]
probabilidade_sobreviver = modelo.predict_proba(novo_passageiro_scaled)[0][1] * 100

st.divider()
st.subheader("Resultado da Predição")

# Barra de progresso visual
st.progress(int(probabilidade_sobreviver))
st.markdown(f"### Probabilidade de Sobrevivência: **{probabilidade_sobreviver:.1f}%**")

# Feedback condicional
if probabilidade_sobreviver >= 50:
    st.success("Sobreviveria! O modelo classificou este passageiro como sobrevivente.")
else:
    st.error("Não Sobreviveria. O modelo classificou este passageiro como vítima.")
