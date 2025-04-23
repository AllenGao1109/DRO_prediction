import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


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


class ModelTrainer:
    def __init__(self, input_dim, num_classes=4, learning_rate=0.01, num_epochs=2000):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None
        self.best_model_params = None

    def _prepare_data(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        return X_tensor, y_tensor

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.tensor(X_test, dtype=torch.float32))
            y_prob = F.softmax(y_pred, dim=1).numpy()
        auc_value = roc_auc_score((y_test == 3).astype(int), y_prob[:, 3])
        return auc_value, y_prob[:, 3]


class TraditionalMethod(ModelTrainer):
    def __init__(self, input_dim, num_classes=4, learning_rate=0.01, num_epochs=4000):
        super().__init__(input_dim, num_classes, learning_rate, num_epochs)
        self.model = MultiLayerPerceptron(input_dim=input_dim, num_classes=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, X_train, y_train):
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            logits = self.model(X_train_tensor)
            loss = F.cross_entropy(
                logits, y_train_tensor.view(-1).long(), reduction="mean"
            )
            loss.backward()
            self.optimizer.step()
        self.best_model_params = {
            name: param.clone() for name, param in self.model.named_parameters()
        }
        self.model.load_state_dict(self.best_model_params)
        return self


class RegularizationMethod(ModelTrainer):
    def __init__(self, input_dim, num_classes=4, learning_rate=0.01, num_epochs=4000):
        super().__init__(input_dim, num_classes, learning_rate, num_epochs)
        self.model = MultiLayerPerceptron(input_dim=input_dim, num_classes=num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lambdas = [
            0.0001,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
        ]
        self.best_lambda = None

    def train(self, X_train, y_train, hatdata, top_vars):
        X_train_tensor, y_train_tensor = self._prepare_data(X_train, y_train)
        best_auc = -float("inf")
        for lambda_val in self.lambdas:
            model = MultiLayerPerceptron(
                input_dim=self.input_dim, num_classes=self.num_classes
            )
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            for epoch in range(self.num_epochs):
                optimizer.zero_grad()
                logits = model(X_train_tensor)
                loss = F.cross_entropy(
                    logits, y_train_tensor.view(-1).long(), reduction="mean"
                )
                l2_reg = lambda_val * sum(p.pow(2).sum() for p in model.parameters())
                loss += l2_reg
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                X_val = hatdata[top_vars].values
                y_val = hatdata["y_{t+1}"].values
                y_pred = model(torch.tensor(X_val, dtype=torch.float32))
                y_prob = F.softmax(y_pred, dim=1).numpy()
                val_auc = roc_auc_score((y_val == 3).astype(int), y_prob[:, 3])
                if val_auc > best_auc:
                    best_auc = val_auc
                    self.best_lambda = lambda_val
                    self.best_model_params = {
                        name: param.clone() for name, param in model.named_parameters()
                    }
                print(best_auc)
        self.model.load_state_dict(self.best_model_params)
        return self


class DROMethod(ModelTrainer):
    def __init__(self, input_dim, num_classes=4, learning_rate=0.01, num_epochs=6000):
        super().__init__(input_dim, num_classes, learning_rate, num_epochs)
        self.model = MultiLayerPerceptron(input_dim=input_dim, num_classes=num_classes)
        self.kappa_iter = [
            2e-6,
            3e-6,
            5e-6,
            7e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
        ]
        self.kappacoef = [1, 1.2]
        self.best_kappa = None
        self.was = [18]
        self.best_kappacoef = None
        self.best_lambda = None
        self.scaler = StandardScaler()

    def train_single_run(self, X_train, y_train, hatdata, top_vars):
        X_train_values = X_train
        X_train_tensor = torch.tensor(X_train_values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        best_auc = -float("inf")
        for kcoefiter in self.kappacoef:
            for kappa in self.kappa_iter:
                for wasserstein in self.was:
                    model = MultiLayerPerceptron(
                        input_dim=self.input_dim, num_classes=self.num_classes
                    )
                    optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
                    lambda_param_raw = torch.nn.Parameter(torch.tensor(1.0))
                    optimizer_lambda = optim.Adam(
                        [lambda_param_raw], lr=self.learning_rate
                    )
                    for epoch in range(self.num_epochs):
                        optimizer.zero_grad()
                        optimizer_lambda.zero_grad()
                        logits = model(X_train_tensor)
                        loss = F.cross_entropy(logits, y_train_tensor.view(-1).long())
                        l2_norm = torch.sqrt(
                            sum(p.pow(2).sum() for p in model.parameters())
                        )
                        lambda_param = torch.exp(lambda_param_raw)
                        y_onehot = torch.eye(logits.size(1))[
                            y_train_tensor.view(-1).long()
                        ].to(X_train_tensor.device)
                        true_logits = torch.sum(y_onehot * logits, dim=1)
                        max_other_logits = torch.max((1 - y_onehot) * logits, dim=1)[0]
                        margins = true_logits - max_other_logits - lambda_param * kappa
                        label_uncertainty_term = torch.relu(margins).mean()
                        loss += wasserstein * lambda_param
                        loss += kcoefiter * label_uncertainty_term
                        penalty = torch.relu(l2_norm - lambda_param).pow(2)
                        loss += 500 * penalty
                        loss.backward()
                        optimizer.step()
                        optimizer_lambda.step()
                        with torch.no_grad():
                            lambda_param_raw.data = torch.log(
                                torch.max(torch.exp(lambda_param_raw), l2_norm + 1e-6)
                            )
                            lambda_param_raw.data.clamp_(min=-10)
                    model.eval()
                    with torch.no_grad():
                        X_val = hatdata[top_vars].values
                        y_val = hatdata["y_{t+1}"].values
                        y_pred = model(torch.tensor(X_val, dtype=torch.float32))
                        y_prob = F.softmax(y_pred, dim=1).numpy()
                        val_auc = roc_auc_score((y_val == 3).astype(int), y_prob[:, 3])
                        if val_auc > best_auc:
                            best_auc = val_auc
                            self.best_kappa = kappa
                            self.best_kappacoef = kcoefiter
                            self.best_lambda = lambda_param.data
                            self.best_model_params = {
                                name: param.clone()
                                for name, param in model.named_parameters()
                            }
                            self.best_was = wasserstein
                    print(best_auc)
        self.model.load_state_dict(self.best_model_params)
        return self


def train_and_eval_traditional(X_train, y_train, data, top_vars, filtered_list):
    """训练并评估传统方法"""
    input_dim = X_train.shape[1]
    model = TraditionalMethod(input_dim=input_dim)
    model.train(X_train, y_train)

    auc_list, pro_list = [], []
    auc_list_4, pro_list_4 = [], []

    for site in filtered_list:
        df_site = data[data["site"] == site]
        X_site = df_site[top_vars].values
        y_site = df_site["y_{t+1}"].values
        auc, proba = model.evaluate(X_site, y_site)
        auc_list.append(auc)
        pro_list.append(proba.reshape(-1, 1))

        df_site_4 = df_site[df_site["y_t"] != 3]
        X_site_4 = df_site_4[top_vars].values
        y_site_4 = df_site_4["y_{t+1}"].values
        auc_4, proba_4 = model.evaluate(X_site_4, y_site_4)
        auc_list_4.append(auc_4)
        pro_list_4.append(proba_4.reshape(-1, 1))

    return model, auc_list, pro_list, auc_list_4, pro_list_4


def train_and_eval_regularization(
    X_train, y_train, hatdata, top_vars, data, filtered_list
):
    """训练并评估正则化方法"""
    input_dim = X_train.shape[1]
    model = RegularizationMethod(input_dim=input_dim)
    model.train(X_train, y_train, hatdata, top_vars)

    auc_list, pro_list = [], []
    auc_list_4, pro_list_4 = [], []
    white_list, white_list_4 = [], []

    for site in filtered_list:
        df_site = data[data["site"] == site]
        X_site = df_site[top_vars].values
        y_site = df_site["y_{t+1}"].values
        auc, proba = model.evaluate(X_site, y_site)
        auc_list.append(auc)
        pro_list.append(proba.reshape(-1, 1))

        df_site_4 = df_site[df_site["y_t"] != 3]
        X_site_4 = df_site_4[top_vars].values
        y_site_4 = df_site_4["y_{t+1}"].values
        auc_4, proba_4 = model.evaluate(X_site_4, y_site_4)
        auc_list_4.append(auc_4)
        pro_list_4.append(proba_4.reshape(-1, 1))

        df_white = df_site[df_site["race_ethnicity"] != 1]
        X_white = df_white[top_vars].values
        y_white = df_white["y_{t+1}"].values
        auc_white, _ = model.evaluate(X_white, y_white)
        white_list.append(auc_white)

        df_white_4 = df_white[df_white["y_t"] != 3]
        X_white_4 = df_white_4[top_vars].values
        y_white_4 = df_white_4["y_{t+1}"].values
        auc_white_4, _ = model.evaluate(X_white_4, y_white_4)
        white_list_4.append(auc_white_4)

    return model, auc_list, pro_list, auc_list_4, pro_list_4, white_list, white_list_4


def train_and_eval_dro_multiple(
    n_runs, X_train, y_train, hatdata, top_vars, data, filtered_list
):
    """多次训练并评估DRO方法，返回平均结果"""
    input_dim = X_train.shape[1]
    auc_list = None
    pro_list = None
    auc_list_4 = None
    pro_list_4 = None
    white_list = None
    white_list_4 = None
    best_params = []
    models = []

    for run in range(n_runs):
        print(run)
        model = DROMethod(input_dim=input_dim)
        model.train_single_run(X_train, y_train, hatdata, top_vars)
        models.append(model)

        run_auc, run_pro, run_auc_4, run_pro_4, run_white, run_white_4 = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for site in filtered_list:
            df_site = data[data["site"] == site]
            X_site = df_site[top_vars].values
            y_site = df_site["y_{t+1}"].values
            auc, proba = model.evaluate(X_site, y_site)
            run_auc.append(auc)
            run_pro.append(proba.reshape(-1, 1))

            df_site_4 = df_site[df_site["y_t"] != 3]
            X_site_4 = df_site_4[top_vars].values
            y_site_4 = df_site_4["y_{t+1}"].values
            auc_4, proba_4 = model.evaluate(X_site_4, y_site_4)
            run_auc_4.append(auc_4)
            run_pro_4.append(proba_4.reshape(-1, 1))

            df_white = df_site[df_site["race_ethnicity"] != 1]
            X_white = df_white[top_vars].values
            y_white = df_white["y_{t+1}"].values
            auc_white, _ = model.evaluate(X_white, y_white)
            run_white.append(auc_white)

            df_white_4 = df_white[df_white["y_t"] != 3]
            X_white_4 = df_white_4[top_vars].values
            y_white_4 = df_white_4["y_{t+1}"].values
            auc_white_4, _ = model.evaluate(X_white_4, y_white_4)
            run_white_4.append(auc_white_4)

        # 累加结果
        if auc_list is None:
            auc_list = run_auc
            pro_list = run_pro
            auc_list_4 = run_auc_4
            pro_list_4 = run_pro_4
            white_list = run_white
            white_list_4 = run_white_4
        else:
            auc_list = [a + b for a, b in zip(auc_list, run_auc)]
            pro_list = [a + b for a, b in zip(pro_list, run_pro)]
            auc_list_4 = [a + b for a, b in zip(auc_list_4, run_auc_4)]
            pro_list_4 = [a + b for a, b in zip(pro_list_4, run_pro_4)]
            white_list = [a + b for a, b in zip(white_list, run_white)]
            white_list_4 = [a + b for a, b in zip(white_list_4, run_white_4)]

        best_params.append(
            {
                "kappa": model.best_kappa,
                "kappacoef": model.best_kappacoef,
                "lambda": float(model.best_lambda)
                if hasattr(model.best_lambda, "item")
                else model.best_lambda,
                "was": model.best_was,
            }
        )

    # 求平均
    auc_list = [x / n_runs for x in auc_list]
    pro_list = [x / n_runs for x in pro_list]
    auc_list_4 = [x / n_runs for x in auc_list_4]
    pro_list_4 = [x / n_runs for x in pro_list_4]
    white_list = [x / n_runs for x in white_list]
    white_list_4 = [x / n_runs for x in white_list_4]

    return (
        models,
        auc_list,
        pro_list,
        auc_list_4,
        pro_list_4,
        white_list,
        white_list_4,
        best_params,
    )
