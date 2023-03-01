# Importando os pacotes a serem utilizados
import pandas as pd
import numpy as np

# Importar o arquivo com os filmes e visualizar as primeiras linhas
filmes = pd.read_csv('movies_metadata.csv', low_memory = False)

# Importando o arquivo de avaliações e avaliando as primeiras linhas
avaliacoes = pd.read_csv('ratings.csv')

# Filtrando somente as colunas necessários e renomeando nome das variaveis

# Seleciona somente as variaveis que iremos utilizar
filmes = filmes [['id','original_title','original_language','vote_count']]

# Renomeia as variaveis
filmes.rename(columns = {'id':'ID_FILME','original_title':'TITULO','original_language':'LINGUAGEM','vote_count':'QT_AVALIACOES'}, inplace = True)

# Filtrando somente as colunas necessários e renomeando nome das variaveis

# Seleciona somente as variaveis que iremos utilizar
avaliacoes = avaliacoes [['userId','movieId','rating']]

# Renomeia as variaveis
avaliacoes.rename(columns = {'userId':'ID_USUARIO','movieId':'ID_FILME','rating':'AVALIACAO'}, inplace = True)

# Como são poucos os valores nulos iremos remover porque não terá impacto nenhum
filmes.dropna(inplace = True)

# Vamos pegar o ID_USUARIO somente de usuários que fizeram mais de 999 avaliações
qt_avaliacoes = avaliacoes['ID_USUARIO'].value_counts() > 999
y = qt_avaliacoes[qt_avaliacoes].index

# Pegando somente avaliacoes dos usuarios que avaliaram mais de 999 vezes
avaliacoes = avaliacoes[avaliacoes['ID_USUARIO'].isin(y)]

# Vamos usar os filmes que possuem somente uma quantidade de avaliações superior a 999 avaliações
filmes = filmes[filmes['QT_AVALIACOES'] > 999]

# Vamos agrupar e visualizar a quantidade de filmes pela linguagem
filmes_linguagem = filmes['LINGUAGEM'].value_counts()

# Selecionar somente os filmes da linguagem EN (English)
filmes = filmes[filmes['LINGUAGEM'] == 'en']

# Precisamos converter a variavel ID_FILME em inteiro
filmes['ID_FILME'] = filmes['ID_FILME'].astype(int)

# Concatenando os dataframes
avaliacoes_e_filmes = avaliacoes.merge(filmes, on = 'ID_FILME')

# Vamos descartar os valores duplicados, para que não tenha problemas de termos o mesmo usuário avaliando o mesmo filme
# diversas vezes
avaliacoes_e_filmes.drop_duplicates(['ID_USUARIO','ID_FILME'], inplace = True)

# Vamos excluir a variavel ID_FILME porque não iremos utiliza-la
del avaliacoes_e_filmes['ID_FILME']

# Agora precisamos fazer um PIVOT. O que queremos é que cada ID_USUARIO seja uma variavel com o respectivo valor de nota
# para cada filme avaliado
filmes_pivot = avaliacoes_e_filmes.pivot_table(columns = 'ID_USUARIO', index = 'TITULO', values = 'AVALIACAO')

# Os valores que são nulos iremos preencher com ZERO
filmes_pivot.fillna(0, inplace = True)

# Vamos importar o csr_matrix do pacote SciPy
# Esse método possibilita criarmos uma matriz sparsa
from scipy.sparse import csr_matrix


# Vamos transformar o nosso dataset em uma matriz sparsa
filmes_sparse = csr_matrix(filmes_pivot)

# Vamos importar o algoritmo KNN do SciKit Learn
from sklearn.neighbors import NearestNeighbors 

# Criando e treinando o modelo preditivo
modelo = NearestNeighbors(algorithm = 'brute')
modelo.fit(filmes_sparse)

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
    return titles