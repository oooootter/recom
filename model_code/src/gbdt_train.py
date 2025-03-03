# 创建LightGBM数据集
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_train, y_train, reference=lgb_train)
print('post process data finish)

# 设置参数 
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary', # 目标：二分类问题
    'metric': {'binary_logloss', 'auc'}, # 评估指标：二分类logloss和AUC
    'num_leaves': 31, # 叶子节点数:用于控制树的复杂度，避免过拟合
    'learning_rate': 0.2,
    'feature_fraction': 0.9, # 选择的特征比例
    'bagging_fraction': 0.8, # 采样的样本比例
    'bagging_freq': 5,  # 每5轮进行bagging
    'verbose': -1   # 日志显示，-1为不显示
}

# 保存AUC
train_aucs = []
test_aucs = []

# 开始训练，从5棵树开始，直到100棵树，每5棵树为一个间隔
for i in range(5，101，5):
    print(f"Training model with {i} trees...")
    # 训练模型
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=i,
                valid_sets=[lgb train, lgb eval],valid_names=['train'，'valid']) #关闭详细日志

    # 使用最后一棵树进行预测并计算 AUC
    y_train_pred= gbm.predict(X_train)
    y_test_pred= gbm.predict(X_test)

    train_auc =roc_auc_score(y_train,y_train_pred)test_auc=roc_auc_score(y_test,y_test_pred)
    
    # 打印 AUC
    print(f"Iteration {i}: Train Auc={train_auc:.4f}, Test Auc={test_auc:.4f}")
    
    # 保存模型
    gbm.save_model(f'./models_{i}.txt')
    
    # 保存 AUC
    train_aucs.append(train_auc)
    test_aucs.append(test_auc)
