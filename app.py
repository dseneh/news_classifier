from flask import Flask, render_template, request, jsonify, url_for
import pickle
import datetime
from datetime import date, timedelta
import pandas as pd
import news_object

app = Flask( __name__ )


@app.route( "/" )
def get_index():
    return render_template( "index.html" )


@app.route( '/', methods=['post'] )
def result():
    news = request.form.get( 'news-text' )
    news_classification = news_object.getNewsClassification( news )
    return render_template( 'result.html', news=news, news_classification=news_classification )


if __name__ == '__main__':
    import os

    HOST = os.environ.get( 'SERVER_HOST', 'localhost' )
    try:
        PORT = int( os.environ.get( 'SERVER_PORT', '5555' ) )
    except ValueError:
        PORT = 5555
    app.run( HOST, PORT )
