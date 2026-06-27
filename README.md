# titanic-analise
##Estudo de analise de dados e aplicação de modelo de predição.

# 🚢 Preditor de Sobrevivência do Titanic

Este é um projeto prático de Machine Learning e visualização de dados desenvolvido em Python. Ele utiliza um modelo de **Regressão Logística** para prever a probabilidade de sobrevivência de um passageiro do Titanic com base em características como idade, sexo, classe e valor da tarifa. 

A interface gráfica interativa foi construída utilizando **Streamlit**, permitindo inferências em tempo real diretamente do navegador.

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Python 3**
* **Pandas:** Limpeza, estruturação e manipulação da base de dados.
* **Seaborn:** Extração do dataset original.
* **Scikit-Learn:** Padronização matemática (`StandardScaler`) e treinamento do algoritmo de Machine Learning (`LogisticRegression`).
* **Streamlit:** Criação do Dashboard interativo e servidor web local.

## ⚙️ Como executar o projeto localmente (VS Code)

Siga os passos abaixo para rodar a aplicação no seu ambiente de desenvolvimento.

### 1. Clone ou baixe o projeto
Abra o projeto no **VS Code**. Certifique-se de que o arquivo `app.py` está na raiz da pasta aberta.

### 2. Crie um Ambiente Virtual (Recomendado)
Para evitar conflitos com outras bibliotecas instaladas na sua máquina, abra o terminal do VS Code (`Ctrl + '` ou `Terminal > New Terminal`) e crie um ambiente virtual isolado:

#1. Crie o ambiente virtual do projeto
No Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
No Linux/Mac:

Bash
python3 -m venv venv
source venv/bin/activate

#2. Instale as Dependências
Com o ambiente virtual ativado, instale as bibliotecas necessárias executando:

Bash
pip install streamlit pandas seaborn scikit-learn
(Opcional: Você pode gerar um arquivo requirements.txt futuramente usando pip freeze > requirements.txt)

#3. Execute a Aplicação
No terminal, inicie o servidor local do Streamlit:

Bash
streamlit run app.py
#4. Acesse o Dashboard
O comando acima iniciará a aplicação e abrirá automaticamente o seu navegador padrão. Caso não abra sozinho, acesse o endereço fornecido no terminal, que geralmente é:
http://localhost:8501

🧠 Como o Modelo Funciona
O script carrega os dados históricos em cache, trata valores nulos (utilizando a mediana para idades ausentes, evitando distorções de outliers), padroniza as escalas matemáticas numéricas e treina o algoritmo nos bastidores. Ao interagir com a interface web, os novos dados inseridos pelo usuário são normalizados e injetados no modelo, que devolve o cálculo probabilístico instantaneamente.
