[![Gustavo-H-Martins](https://github-readme-stats.vercel.app/api?username=Gustavo-H-Martins&show_icons=true&theme=radical)](https://github.com/Gustavo-H-Martins)


[Página online do modelo em operação](https://sistemaderecomendacao.gustavo-h-marti.repl.co/)
# CRIANDO UM SISTEMA DE RECOMENDAÇÃO DE FILMES
Este notebook apresenta um passo a passo para criar um sistema de recomendação de filmes. A fonte de dados utilizada está disponível no [Kaggle](https://www.kaggle.com/code/alyssonbispopereira/recomenda-o-de-filmes-ptbr/data).

## Importando os pacotes a serem utilizados
Para começar, são importados os pacotes pandas e numpy para manipulação dos dados.
```  PYTHON      
import pandas as pd
import numpy as np
```
## Importando os dados
São importados os arquivos `movies_metadata.csv` e `ratings.csv` disponíveis na fonte de dados.
```  PYTHON      
filmes = pd.read_csv('movies_metadata.csv', low_memory = False)
avaliacoes = pd.read_csv('ratings.csv')
```

## Pré Processamento dos Dados
### Filmes
Primeiro, são selecionadas somente as variáveis que serão utilizadas: 

ID_FILME, TITULO, LINGUAGEM e QT_AVALIACOES.

``` PYTHON
filmes = filmes[[
    'id',
    'original_title',
    'original_language',
    'vote_count']]

```
Em seguida, as variáveis são renomeadas.

``` PYTHON
filmes.rename(columns = {
    'id':'ID_FILME',
    'original_title':'TITULO',
    'original_language':'LINGUAGEM',
    'vote_count':'QT_AVALIACOES'
    }, inplace = True)

```
### Avaliações
Primeiramente, é selecionado somente as variáveis que serão utilizadas: 

ID_USUARIO, ID_FILME e AVALIACAO.

``` PYTHON
avaliacoes = avaliacoes[['userId','movieId','rating']]

```

Em seguida, as variáveis são renomeadas.
``` PYTHON
avaliacoes.rename(columns = {
    'userId':'ID_USUARIO',
    'movieId':'ID_FILME',
    'rating':'AVALIACAO'
    }, inplace = True)
```

Tratamento de valores nulos
Em seguida, é verificado se há valores nulos nos arquivos `filmes` e `avaliacoes`.

``` PYTHON
filmes.isna().sum()
avaliacoes.isna().sum()

```

Como são poucos os valores nulos em `filmes`, eles são removidos porque não terão impacto nenhum.
``` PYTHON
filmes.dropna(inplace = True)

```

### Selecionando usuários com mais avaliações
Para evitar problemas com falta de avaliações, serão selecionados somente os usuários que avaliaram mais de 999 vezes. Primeiro, é verificado a quantidade de avaliações por usuário.
``` PYTHON
avaliacoes['ID_USUARIO'].value_counts()

```
Em seguida, são selecionados os usuários que avaliaram mais de 999 vezes.
``` PYTHON
qt_avaliacoes = avaliacoes['ID_USUARIO'].value_counts() > 999
y = qt_avaliacoes[qt_avaliacoes].index

```
Selecionando filmes
Serão usados somente os filmes que possuem mais de 999 avaliações.
``` PYTHON
filmes = filmes[filmes['QT_AVALIACOES'] > 999]
```

### Converter a variavel ID_FILME em inteiro
``` PYTHON
filmes['ID_FILME'] = filmes['ID_FILME'].astype(int)
```

###  Concatenando os dataframes
``` PYTHON
avaliacoes_e_filmes = avaliacoes.merge(filmes, on = 'ID_FILME')
avaliacoes_e_filmes.head()
```

### descartar os valores duplicados, para que não tenha problemas de termos o mesmo usuário avaliando o mesmo filme diversas vezes
``` PYTHON
avaliacoes_e_filmes.drop_duplicates(['ID_USUARIO','ID_FILME'], inplace = True)
```

### Fazer PIVOT de `avaliacoes_e_filmes`.
``` PYTHON
filmes_pivot = avaliacoes_e_filmes.pivot_table(columns = 'ID_USUARIO', index = 'TITULO', values = 'AVALIACAO')
```

## importar o csr_matrix do pacote SciPy
Esse método possibilita criar uma matriz sparsa
``` PYTHON
from scipy.sparse import csr_matrix
# transformar o dataset em uma matriz sparsa
filmes_sparse = csr_matrix(filmes_pivot)
```

## importar o algoritmo KNN do SciKit Learn
``` PYTHON
from sklearn.neighbors import NearestNeighbors 
```

# Criando e treinando o modelo preditivo
``` PYTHON
modelo = NearestNeighbors(algorithm = 'brute')
modelo.fit(filmes_sparse)
```

## Criar uma função para fazer as previsões e recomendações.
``` PYTHON
def movie_recomendation(movies:list =[""]):
    """
        itera sobre uma lista de títulos e retorna as recomendações com base nestes.    
    """
    for movie in movies:
        distances, sugestions = modelo.kneighbors(filmes_pivot.filter(items = [movie], axis=0).values.reshape(1, -1))

        for i in range(len(sugestions)):
            print(f"Com base no título informado \033[1m{filmes_pivot.index[sugestions[i][0]]}\033[0;0m estas são as recomendações do sistema:")
            titles = list(filmes_pivot.index[sugestions[i]])
            for title in titles:
                print(f"Opção {titles.index(title)} - \033[1m{title}\033[0;0m")
            print("\033[1m ~ \033[0;0m" * 15)
```

### Texte aplicado
``` PYTHON
movie_recomendation(['127 Hours','Toy Story'])
```

