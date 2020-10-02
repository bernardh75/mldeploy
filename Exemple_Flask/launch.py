from flask import Flask, render_template, request

import json
import funct

app = Flask(__name__)

from funct import Sentiment

@app.route("/")
def home():
    return render_template("index.html", title='Home')

#2
@app.route("/prediction", methods=['POST'])
def retour():
    sentiment = Sentiment()

    user_text = request.form.get('input_text')
    eval = sentiment.evalSentiment(user_text)
    return eval

#3
@app.route("/entrainement", methods=['POST'])
def entr(usertexte=None):
    sentiment = Sentiment()
    ret = sentiment.entrainement()
    return json.dumps({'text_user':ret})
    
#4
@app.route("/customtext", methods=['POST'])
def disp(usertxt=None):
    user_text = request.form.get('input_txt')
    print(user_text)
    return render_template('textUserRender.html', usertxt=user_text)  

if __name__ == "__main__":
    app.run(debug=True)


