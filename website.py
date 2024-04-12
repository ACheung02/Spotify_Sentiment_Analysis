from flask import Flask, redirect, url_for, render_template

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/machine_learning_models')
def machine_learning():
    return render_template('machinelearning.html')

@app.route('/sentiment_analysis')
def sentiment_analysis():
    return render_template('sentimentanalysis.html')

if __name__ == '__main__':
    app.run(debug=True)