import fasttext
from flask import Flask, render_template

app = Flask(__name__)
model = fasttext.load_model('lid.176.ftz')

@app.route('/')
def gpt():
  return render_template('index.html')

# URL 에 매개변수를 받아 진행하는 방식입니다.
@app.route('/api/<text>')
def api(text):
    try:
        if model.predict(text, k=1)[0][0] == '__label__ko':
            try:
                return ko.generate_reply(text).replace(text, "")
            except:
                import ko
                return ko.generate_reply(text).replace(text, "")
        
        elif model.predict(text, k=1)[0][0] == '__label__ja':
            try:
                return ja.generate_reply(text).replace(text, "")
            except:
                import ja
                return ja.generate_reply(text).replace(text, "")
        
        elif model.predict(text, k=1)[0][0] == '__label__en':
            try:
                return en.generate_reply(text).replace(text, "")
            except:
                import en
                return en.generate_reply(text).replace(text, "")
        
        else:
            try:
                return en.generate_reply(text).replace(text, "")
            except:
                import en
                return en.generate_reply(text).replace(text, "")
    except:
        return "ERRORGENERATE"


app.run(host='0.0.0.0', port=5000)