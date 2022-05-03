from json import loads
from sklearn.linear_model import LinearRegression
import numpy as np

X_TEST = [[10, 20, 30]]

with open('model.json', 'r') as f:
    content = f.read()
    model = loads(content)


predictor = LinearRegression(n_jobs=-1)
predictor.coef_ = np.array(model)
predictor.intercept_ = np.array([0])

outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_

print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))