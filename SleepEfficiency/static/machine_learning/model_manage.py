import joblib
import numpy as np
from matplotlib import pyplot as plt


def model_save(model, scaler, default_values, filename='static/machine_learning/model/mf_model.pkl'):
    """
    保存模型、标准化器和默认值到文件。

    参数:
    - model: 训练好的模型
    - scaler: 用于数据标准化的标准化器
    - default_values: 特征的默认值（均值）
    - filename: 保存文件的名称，默认 'mf_model.pkl'
    """
    data_to_save = {
        'model': model,
        'scaler': scaler,
        'default_values': default_values
    }
    joblib.dump(data_to_save, filename)
    print(f"模型和标准化器已保存到 {filename}")


def model_load(filename='static/machine_learning/model/mf_model.pkl'):
    """
    从文件加载模型、标准化器和默认值。

    参数:
    - filename: 保存文件的名称，默认 'mf_model.pkl'

    返回:
    - model: 加载的模型
    - scaler: 加载的标准化器
    - default_values: 加载的特征默认值
    """
    data = joblib.load(filename)
    print(f"从 {filename} 加载模型和标准化器")
    return data['model'], data['scaler'], data['default_values']


#
#
#
def predict_sleep_efficiency(age, gender, model, scaler, default_values):
    """
    使用模型预测特定年龄和性别的睡眠效率。

    参数:
    - age: 年龄
    - gender: 性别 (0: 男性, 1: 女性)
    - model: 训练好的xgb模型
    - scaler: 标准化器
    - default_values: 特征默认值

    返回:
    - 预测的睡眠效率
    """
    new_data = {
        'Age': age,
        'Gender': gender,
        'Sleep duration': default_values['Sleep duration'],
        'REM sleep percentage': default_values['REM sleep percentage'],
        'Deep sleep percentage': default_values['Deep sleep percentage'],
        'Light sleep percentage': default_values['Light sleep percentage'],
        'Awakenings': default_values['Awakenings'],
        'Caffeine consumption': default_values['Caffeine consumption'],
        'Alcohol consumption': default_values['Alcohol consumption'],
        'Smoking status': default_values['Smoking status'],
        'Exercise frequency': default_values['Exercise frequency']
    }

    new_data_values = np.array(list(new_data.values())).reshape(1, -1)
    new_data_scaled = scaler.transform(new_data_values)
    predicted_efficiency = model.predict(new_data_scaled)
    return predicted_efficiency[0]

#
# def plot_age_vs_efficiency(model, scaler, default_values):
#     ages = range(10, 71)
#     male_efficiency = [predict_sleep_efficiency(age, 0, model, scaler, default_values) for age in ages]
#     female_efficiency = [predict_sleep_efficiency(age, 1, model, scaler, default_values) for age in ages]
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(ages, male_efficiency, label='男性', color='blue')
#     plt.plot(ages, female_efficiency, label='女性', color='red')
#     plt.xlabel('年龄')
#     plt.ylabel('预测的睡眠效率')
#     plt.title('年龄与睡眠效率预测')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f'pic/年龄与睡眠效率预测.png', bbox_inches='tight')
#     plt.show()
#
#
# plot_age_vs_efficiency(model, scaler, default_values)