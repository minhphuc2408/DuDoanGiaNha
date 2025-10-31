# 3. Danh sách mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    "Polynomial Regression (deg=2)": make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0)),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=1500, learning_rate=0.01, num_leaves=31, random_state=42)
}
#best model sau khi tunning 
best_xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.6,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    objective='reg:squarederror',
    n_jobs=-1,
    tree_method='hist'
)
#stacking xgboost+lightgbm+randomforest
best_lgb = models['LightGBM']
best_rf = RandomForestRegressor(n_estimators=500, random_state=42)

stack_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('lgb', best_lgb),
        ('rf', best_rf)
    ],
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

stack_model, results_stack = compile_and_evaluate_model(
    "Stacking_XGB_LGB_RF",
    stack_model,
    X_train, y_train,
    X_test, y_test
)

results.append(results_stack)
results_df = pd.DataFrame(results).drop(columns=["Preds"]).sort_values(by="R2", ascending=False)
display(results_df)