import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from static.machine_learning.draw_plot import plot_results


# 随机森林算法
def random_forest_regression(X_train_rf, X_test_rf, y_train_rf, y_test_rf):
    # 定义参数搜索空间
    """
    criterion: 均方误差和绝对误差的决策树分裂标准。
    n_estimators: 森林中的200 到 2000 的 10 个不同树木数量。
    max_features: 每次分裂时考虑平方根和对数两种最大特征数量。
    max_depth: 树的最大深度。定义了从 10 到 100 的 10 个不同深度，并不限制深度。
    min_samples_split: 内部节点再划分所需的最小样本数。
    min_samples_leaf: 叶节点所需的最小样本数。
    bootstrap: 是否使用自助法抽样。
    """
    criterion = ['squared_error', 'absolute_error']
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {
        'criterion': criterion,
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    # 构建随机森林模型
    clf = RandomForestRegressor(n_estimators=200, random_state=42)
    # 10 次随机搜索，进行 3 折交叉验证 verbose=0不打印
    clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                                    n_iter=10, cv=3, verbose=0, random_state=42, n_jobs=1)

    # 训练集上进行参数搜索
    clf_random.fit(X_train_rf, y_train_rf)
    print(clf_random.best_params_)

    # 使用最佳参数构建最终模型并训练
    best_params = clf_random.best_params_
    model_rf = RandomForestRegressor(**best_params)
    model_rf.fit(X_train_rf, y_train_rf)
    y_test_pred_rf = model_rf.predict(X_test_rf)

    mse = mean_squared_error(y_test_rf, y_test_pred_rf)
    r2 = r2_score(y_test_rf, y_test_pred_rf)

    plot_results(y_test_rf, y_test_pred_rf, '随机森林回归 真实值与预测值对比')

    return model_rf, mse, r2
