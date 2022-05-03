
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort
from json import loads
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

with open('model.json', 'r') as f:
    content = f.read()
    model = loads(content)

predictor = LinearRegression(n_jobs=-1)
predictor.coef_ = np.array(model)
predictor.intercept_ = np.array([0])

@app.route("/")
def hello_word():
    params = request.args.get('input')
    parameters = params.split(",")
    X = [[
        int(parameters[0]),
        int(parameters[1]),
        int(parameters[2])
    ]]
    outcome = predictor.predict(X)
    return str(outcome)

if  __name__ == "__main__":
    app.run()