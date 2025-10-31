results = []
pred_dict = {}

for name, model in models.items():
    model, res = compile_and_evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results.append(res)
    pred_dict[name] = res["Preds"]