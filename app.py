from flask import Flask, render_template, request, jsonify, url_for
import pickle
import datetime
from datetime import date, timedelta
import pandas as pd
import os

import news_object
model = news_object.classify_news # pickle.load(open('model_predict.plk', 'rb'))


app = Flask( __name__ )

    
@app.route("/" )
def get_index():
    return render_template( "index.html" )
# Dictionary mapping content to respective category
dic = {
        "alt.atheism": "Miscellaneous",
        "comp.graphics": "Technology",
        "comp.os.ms-windows.misc": "Technology",
        "comp.sys.ibm.pc.hardware": "Technology",
        "comp.sys.mac.hardware": "Technology",
        "comp.windows.x": "Technology",
        "misc.forsale": "Miscellaneous",
        "rec.autos": "Sports",
        "rec.motorcycles":"Sports",
        "rec.sport.baseball": "Sports",
        "rec.sport.hockey": "Sports",
        "sci.crypt": "Science | Research",
        "sci.electronics": "Technology",
        "sci.med": "Medical",
        "sci.space": "Science | Research",
        "soc.religion.christian": "Religious",
        "talk.politics.guns": "World News",
        "talk.politics.mideast": "World News",
        "talk.politics.misc": "Politics",
        "talk.religion.misc": "Religious"}

@app.route( '/', methods=['post'] )
def result():
    news = request.form.get('news-text')
    news_classification = model(news)

    try:
        for key in news_classification:
            val = dic[key]
            value = ""
            if (val != ""):
                value = val
            else:
                value = "Miscellaneous"
    except:
        value = "Miscellaneous"

    return render_template( 'result.html', news=news, news_classification=news_classification, value=value )

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
