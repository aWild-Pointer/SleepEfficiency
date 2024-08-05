import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


def load_data(fpath):
    # 读取数据
    data = pd.read_csv(fpath)

    # 数据清洗，删除包含空字段的行
    data.dropna(inplace=True)

    # 特征工程
    features = ['Age', 'Gender', 'Sleep duration', 'REM sleep percentage', 'Deep sleep percentage',
                'Light sleep percentage', 'Awakenings', 'Caffeine consumption',
                'Alcohol consumption', 'Smoking status', 'Exercise frequency']
    # 将性别、吸烟状态转换为数值
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    data['Smoking status'] = data['Smoking status'].map({'No': 0, 'Yes': 1})

    # 提取特征和目标变量
    X = data[features].values
    y = data['Sleep efficiency'].values

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 计算默认值（使用训练集的均值）
    default_values = data[features].mean().to_dict()

    return X_train, X_test, y_train, y_test, scaler, default_values


# if __name__ == "__main__":
#     file_path = 'Sleep_Efficiency.csv'
#     X_train, X_test, y_train, y_test, scaler, default_values = load_data(file_path)
#
#     # print("多元线性回归")
#     linear_mse, linear_regression_r2 = linear_regression(X_train, X_test, y_train, y_test)
#
#     # print("神经网络回归")
#     neural_mse, neural_r2 = neural_network_regression(X_train, X_test, y_train, y_test)
#
#     # print("XGBoost回归")
#     xgboost_mse, xgboost_r2 = model_xgb = xgboost_regression(X_train, X_test, y_train, y_test)
#
#     # print("随机森林回归")
#     model_rf, rf_mse, rf_r2 = random_forest_regression(X_train, X_test, y_train, y_test)
#     model_save(model_rf, scaler, default_values)
#
#     mse_values = {
#         'Linear Regression': linear_mse,
#         'Neural Network': neural_mse,
#         'XGBoost': xgboost_mse,
#         'Random Forest': rf_mse
#     }
#
#     r2_values = {
#         'Linear Regression': linear_regression_r2,
#         'Neural Network': neural_r2,
#         'XGBoost': xgboost_r2,
#         'Random Forest': rf_r2
#     }
#
#     # Plot MSE values
#     plt.figure(figsize=(10, 5))
#     plt.bar(mse_values.keys(), mse_values.values(), color='skyblue')
#     plt.xlabel('Models')
#     plt.ylabel('MSE')
#     plt.title('不同模型MSE对比')
#     plt.savefig('1.png', bbox_inches='tight')
#     plt.show()
#
#     # Plot R² values
#     plt.figure(figsize=(10, 5))
#     plt.bar(r2_values.keys(), r2_values.values(), color='lightgreen')
#     plt.xlabel('Models')
#     plt.ylabel('R²')
#     plt.title('不同模型R²对比')
#     plt.savefig('2.png', bbox_inches='tight')
#     plt.show()
#
