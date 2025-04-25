import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import optuna


# %% Multi-layer Perceptron Definition
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout_rate=0.3):
        super(MultiLayerPerceptron, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# %% Unified Base Class for Training
class ModelTrainer:
    def __init__(
        self,
        input_dim,
        num_classes,
        learning_rate,
        num_epochs,
        hidden_dim,
        dropout,
    ):
        self.model = MultiLayerPerceptron(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

    def _prepare_data(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        return X_tensor, y_tensor

    def train(self, X_train, y_train):
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            logits = self.model(X_train_tensor)
            loss = F.cross_entropy(logits, y_train_tensor.view(-1).long())
            loss.backward()
            self.optimizer.step()
        return self

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.tensor(X_test, dtype=torch.float32))
            y_prob = F.softmax(y_pred, dim=1).numpy()
        auc_value = roc_auc_score((y_test == 3).astype(int), y_prob[:, 3])
        return auc_value, y_prob[:, 3]


# %% Traditional Method
class TraditionalMethod(ModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


# %% Regularization Method
class RegularizationMethod(ModelTrainer):
    def __init__(self, lambda_val=0.01, **kwargs):
        super().__init__(**kwargs)
        self.lambda_val = lambda_val

    def train(self, X_train, y_train):
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            logits = self.model(X_train_tensor)
            loss = F.cross_entropy(logits, y_train_tensor.view(-1).long())
            l2_reg = self.lambda_val * sum(
                p.pow(2).sum() for p in self.model.parameters()
            )
            loss += l2_reg
            loss.backward()
            self.optimizer.step()
        return self


# %% Evaluate model across sites
def evaluate_model_across_sites(
    model, top_vars, data, filtered_list, evaluate_race=False
):
    results = {"auc_all": [], "prob_all": [], "auc_exclude_3": [], "prob_exclude_3": []}
    if evaluate_race:
        results.update({"auc_non_white": [], "auc_non_white_exclude_3": []})

    for site in filtered_list:
        df_site = data[data["site"] == site]
        X_site = df_site[top_vars].values
        y_site = df_site["y_{t+1}"].values
        auc, proba = model.evaluate(X_site, y_site)
        results["auc_all"].append(auc)
        results["prob_all"].append(proba.reshape(-1, 1))

        df_site_4 = df_site[df_site["y_t"] != 3]
        X_site_4 = df_site_4[top_vars].values
        y_site_4 = df_site_4["y_{t+1}"].values
        auc_4, proba_4 = model.evaluate(X_site_4, y_site_4)
        results["auc_exclude_3"].append(auc_4)
        results["prob_exclude_3"].append(proba_4.reshape(-1, 1))

        if evaluate_race:
            df_non_white = df_site[df_site["race_ethnicity"] != 1]
            X_non_white = df_non_white[top_vars].values
            y_non_white = df_non_white["y_{t+1}"].values
            auc_non_white, _ = model.evaluate(X_non_white, y_non_white)
            results["auc_non_white"].append(auc_non_white)

            df_non_white_4 = df_non_white[df_non_white["y_t"] != 3]
            X_non_white_4 = df_non_white_4[top_vars].values
            y_non_white_4 = df_non_white_4["y_{t+1}"].values
            auc_non_white_4, _ = model.evaluate(X_non_white_4, y_non_white_4)
            results["auc_non_white_exclude_3"].append(auc_non_white_4)

    return results


# %% Unified Optuna tuning framework
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

        short_params = {k: v for k, v in params.items() if k not in ["kappa"]}
        if "num_epochs" in short_params:
            short_params["num_epochs"] = 500

        try:
            filtered_params = {
                k: v for k, v in short_params.items() if k not in ["kappa"]
            }
            model = model_class(input_dim=input_dim, num_classes=4, **filtered_params)

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

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    final_params = {**fixed_params, **study.best_params}
    final_dro_args = {k: v for k, v in final_params.items() if k in ["kappa"]}

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
    print("[Final Results Summary]")
    print("Best Validation AUC:", study.best_value)
    print("Best Parameters:", study.best_params)
    print(
        "Average AUC across sites:", sum(results["auc_all"]) / len(results["auc_all"])
    )
    return (final_model, results), study


# %% DRO Method
class DROMethod(ModelTrainer):
    def __init__(self, kappacoef=1.0, wasserstein=18.0, **kwargs):
        super().__init__(**kwargs)
        self.kappacoef = kappacoef
        self.wasserstein = wasserstein

    def train(self, X_train, y_train, kappa=1e-6):
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        lambda_param_raw = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        optimizer_lambda = optim.Adam(
            [lambda_param_raw], lr=self.optimizer.param_groups[0]["lr"]
        )

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            optimizer_lambda.zero_grad()
            logits = self.model(X_train_tensor)
            loss = F.cross_entropy(logits, y_train_tensor.view(-1).long())

            l2_norm = torch.sqrt(sum(p.pow(2).sum() for p in self.model.parameters()))
            lambda_param = torch.exp(lambda_param_raw)

            y_onehot = torch.eye(logits.size(1))[y_train_tensor.view(-1).long()]
            true_logits = torch.sum(y_onehot * logits, dim=1)
            max_other_logits = torch.max(
                (1 - y_onehot) * logits - 1e9 * y_onehot, dim=1
            )[0]
            margins = true_logits - max_other_logits - lambda_param * kappa

            label_uncertainty_term = torch.relu(margins).mean()
            penalty = torch.relu(l2_norm - lambda_param).pow(2)

            loss += self.kappacoef * label_uncertainty_term
            loss += self.wasserstein * lambda_param
            loss += 500 * penalty

            loss.backward()
            self.optimizer.step()
            optimizer_lambda.step()

            with torch.no_grad():
                lambda_param_raw.data = torch.log(
                    torch.max(torch.exp(lambda_param_raw), l2_norm + 1e-6)
                )
                lambda_param_raw.data.clamp_(min=-10)
        return self
