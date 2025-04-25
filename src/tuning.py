import optuna
from models import evaluate_model_across_sites, TraditionalMethod, DROMethod


def tune_model_with_optuna(
    model_class,
    X_train,
    y_train,
    hatdata,
    top_vars,
    data,
    filtered_list,
    param_ranges,
    fixed_params=None,
    n_trials=60,
):
    input_dim = X_train.shape[1]
    fixed_params = fixed_params or {}

    def objective(trial):
        params = fixed_params.copy()
        for param_name, config in param_ranges.items():
            if config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    config["low"],
                    config["high"],
                    step=config.get("step", 1),
                )
            elif config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )

        short_params = params.copy()
        if "num_epochs" in short_params:
            short_params["num_epochs"] = 300  # Shortened for tuning

        try:
            model = model_class(input_dim=input_dim, num_classes=4, **short_params)
        except TypeError as e:
            print("Failed to init model with params:", short_params)
            raise e

        dro_train_args = {k: v for k, v in params.items() if k in ["kappa"]}

        if isinstance(model, DROMethod):
            model.train(X_train, y_train, **dro_train_args)
        else:
            model.train(X_train, y_train)

        X_val = hatdata[top_vars].values
        y_val = hatdata["y_{t+1}"].values
        val_auc, _ = model.evaluate(X_val, y_val)
        return val_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    final_params = {**fixed_params, **study.best_params}
    final_dro_args = {
        k: v
        for k, v in final_params.items()
        if k in ["kappa", "kappacoef", "wasserstein"]
    }

    try:
        final_model = model_class(
            input_dim=input_dim,
            num_classes=4,
            **{k: v for k, v in final_params.items() if k not in ["kappa"]},
        )
    except TypeError as e:
        print("Failed to init final model with params:", final_params)
        raise e

    if isinstance(final_model, DROMethod):
        final_model.train(X_train, y_train, **final_dro_args)
    else:
        final_model.train(X_train, y_train)

    results = evaluate_model_across_sites(
        final_model,
        top_vars,
        data,
        filtered_list,
        evaluate_race=(model_class != TraditionalMethod),
    )
    return (final_model, results), study
