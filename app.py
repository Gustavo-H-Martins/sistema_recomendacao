from flask import Flask, render_template, request, jsonify, url_for
from sistemaRecomendacao import movie_recomendation
import pandas as pd
app = Flask(__name__)


# Importar o arquivo com os filmes e visualizar as primeiras linhas
filmes = pd.read_csv('movies_metadata.csv', usecols=['original_title','original_language'],low_memory = False)
filmes.rename(columns = {'original_title':'TITULO','original_language':'LINGUAGEM'}, inplace = True)
filmes = filmes[filmes['LINGUAGEM'] == 'en']

@app.get('/titulos')
def get_coluna():
    valores = filmes['TITULO'].tolist()
    return jsonify(valores)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index_post():
    title = request.form['title']
    results = movie_recomendation([title])
    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run()
