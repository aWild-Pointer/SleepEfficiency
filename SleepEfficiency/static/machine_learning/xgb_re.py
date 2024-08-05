import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

from static.machine_learning.draw_plot import plot_results, plot_loss_curve


# XGBoost算法
def xgboost_regression(X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb):
    # 定义和训练XGBoost回归模型 objective=回归问题 n_estimators=200棵树 learning_rate=学习率0.01 max_depth=树的最大深度3
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.01, max_depth=3)
    model_xgb.fit(X_train_xgb, y_train_xgb, eval_set=[(X_train_xgb, y_train_xgb)], verbose=False)

    # 进行预测
    y_train_pred_xgb = model_xgb.predict(X_train_xgb)
    y_test_pred_xgb = model_xgb.predict(X_test_xgb)

    # 计算评估指标
    mse = mean_squared_error(y_train_xgb, y_train_pred_xgb)
    r2 = r2_score(y_train_xgb, y_train_pred_xgb)

    # print(f"均方误差 (MSE): {mse}")
    # print(f"决定系数 (R²): {r2}")

    plot_results(y_test_xgb, y_test_pred_xgb, 'XGBoost回归 真实值与预测值对比')

    # 绘制训练损失变化曲线
    results = model_xgb.evals_result()
    rmse_values = results['validation_0']['rmse']
    plot_loss_curve(rmse_values, 'XGBoost回归 训练损失变化')

    return mse, r2
