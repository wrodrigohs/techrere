# TechReré


## Objetivo do projeto
Este é o projeto de análise de sentimento, é um MVP, que tem o objetivo de auxiliar usuário na sua tomada de decisão ao analisar notícias sobre reputação da empresa com base em informaçoes do sites financeiros.


## Configuração
Certifique-se de ter todas as dependências instaladas antes de executar o projeto. Você pode instalá-las usando o gerenciador de pacotes Python, pip. Utilize o seguinte comando:


```
comando pip install -r requirements.txt
```


Além disso, você precisa ter acesso a um banco de dados PostgreSQL. Certifique-se de ter as credenciais corretas e altere as informações de conexão no código, na função get_noticia().




## Funcionamento
O projeto possui as seguintes funcionalidades:


Ao acessar a página inicial, o usuário é apresentado a um formulário onde pode inserir uma notícia.
Ao enviar o formulário, a notícia é enviada para o servidor para análise de sentimento.
O servidor utiliza o modelo LIABertClassifier para classificar a notícia como "negativa", "positiva" ou "neutra".
Além da classificação, o servidor gera dois gráficos usando a biblioteca Plotly e os exibe na página.
O resultado da classificação e os gráficos são renderizados na página inicial.




# Deploy local


## Como executar usando o docker?


1. Monte a imagem Docker:


```console
docker build . -t techrere
```


2. Execute o container  Docker:


```console
docker run -p 5000:5000 techrere
```


Acesse no navedor o endereço: [localhost:5000](http:localhost:5000)







