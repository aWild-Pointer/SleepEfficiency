from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

from static.machine_learning.model_manage import predict_sleep_efficiency
from static.machine_learning.linear_torch import linear_regression
from static.machine_learning.main import load_data
from static.machine_learning.model_manage import model_save, model_load
from static.machine_learning.neural_network import neural_network_regression
from static.machine_learning.random_forest import random_forest_regression
from static.machine_learning.xgb_re import xgboost_regression

app = Flask(__name__)


file_path = 'static/machine_learning/Sleep_Efficiency.csv'
X_train, X_test, y_train, y_test, scaler, default_values = load_data(file_path)

# print("多元线性回归")

# print("神经网络回归")

# print("XGBoost回归")

# print("随机森林回归")

# 加载训练好的模型和标准化器
model, scaler, default_values = model_load('static/machine_learning/model/mf_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        gender = int(request.form['gender'])

        # 预测
        prediction = predict_sleep_efficiency(age, gender, model, scaler, default_values)

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/linear-regression')
def linearre():
    linear_mse, linear_regression_r2 = linear_regression(X_train, X_test, y_train, y_test)
    return render_template('linear-regression.html', mse=linear_mse, r2=linear_regression_r2)


@app.route('/neural-network')
def neuraln():
    neural_mse, neural_r2 = neural_network_regression(X_train, X_test, y_train, y_test)
    return render_template('neural-network.html', mse=neural_mse, r2=neural_r2)


@app.route('/xgboost')
def xgbo():
    xgboost_mse, xgboost_r2 = model_xgb = xgboost_regression(X_train, X_test, y_train, y_test)
    return render_template('xgboost.html', mse=xgboost_mse, r2=xgboost_r2)


@app.route('/random-forest')
def rf():
    model_rf, rf_mse, rf_r2 = random_forest_regression(X_train, X_test, y_train, y_test)
    model_save(model_rf, scaler, default_values)
    return render_template('random-forest.html', mse=rf_mse, r2=rf_r2)


@app.route('/others')
def other():
    return render_template('others.html')


if __name__ == '__main__':
    app.run(debug=True)

