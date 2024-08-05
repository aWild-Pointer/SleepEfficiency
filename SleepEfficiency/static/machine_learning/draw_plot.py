import matplotlib
import matplotlib.pyplot as plt

# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_results(y_test_contrast, y_test_pred_contrast, title):
    # plt.figure(figsize=(8, 6))
    # plt.plot(y_test_contrast, label='真实值', marker='o', linestyle='-', color='green')
    # plt.plot(y_test_pred_contrast, label='预测值', marker='x', linestyle=':', color='blue')
    # plt.xlabel('测试样本索引')
    # plt.ylabel('睡眠效率')
    # plt.title(title)
    # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_contrast, y_test_pred_contrast, edgecolor='k', alpha=0.7)
    plt.plot([y_test_contrast.min(), y_test_contrast.max()], [y_test_contrast.min(), y_test_contrast.max()], 'k--', lw=3)
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(title)
    plt.savefig(f'static/machine_learning/pic/{title}.png', bbox_inches='tight')
    plt.show()


def plot_loss_curve(epoch_loss, title):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(epoch_loss)), epoch_loss, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'static/machine_learning/pic/{title}.png', bbox_inches='tight')
    plt.show()
