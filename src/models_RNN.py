import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import optuna


# ======================================
# 1. LSTM Network
# ======================================
class RNNNetwork(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=64, num_classes=4, num_layers=1, dropout=0.0
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            nonlinearity="tanh",  # 默认就是 tanh
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)

        packed_out_data = packed_out.data  # (total_valid_steps, hidden_dim)
        fc_out_data = self.fc(packed_out_data)  # (total_valid_steps, num_classes)

        packed_fc_out = nn.utils.rnn.PackedSequence(
            data=fc_out_data,
            batch_sizes=packed_out.batch_sizes,
            sorted_indices=packed_out.sorted_indices,
            unsorted_indices=packed_out.unsorted_indices,
        )

        return packed_fc_out


# ======================================
# 2. Base Trainer
# ======================================
class LSTMTrainer:
    def __init__(
        self,
        input_dim,
        num_classes=4,
        learning_rate=0.001,
        num_epochs=2000,
        hidden_dim=64,
        dropout=0.3,
    ):
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define LSTM model and move it to device
        self.model = RNNNetwork(input_dim, hidden_dim, num_classes, 2, dropout).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

    def train(self, train_loader):
        self.model.train()

        for X_batch, y_batch, lengths, _ in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(X_batch, lengths)  # logits is PackedSequence
            logits_flat = logits.data  # <-- just take data

            loss = F.cross_entropy(logits_flat, y_batch.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

    def evaluate(self, loader):
        self.model.eval()
        all_probs, all_targets, all_subject_ids = [], [], []

        with torch.no_grad():
            for X_batch, y_batch, lengths, subject_ids in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                lengths = lengths.to(self.device)

                logits = self.model(X_batch, lengths)  # logits is PackedSequence
                logits_flat = logits.data

                probs_flat = F.softmax(logits_flat, dim=-1)

                all_probs.append(probs_flat.cpu())
                all_targets.append(y_batch.cpu())
                all_subject_ids.extend(subject_ids)

        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        probs_class3 = all_probs[:, 3]
        targets_flat = all_targets

        if targets_flat.numel() == 0:
            auc_score = np.nan
        else:
            auc_score = roc_auc_score(
                (targets_flat == 3).numpy(),
                probs_class3.numpy(),
            )

        return auc_score, probs_class3.numpy(), all_subject_ids, targets_flat.numpy()


# ======================================
# 3. Traditional / Regularized / DRO Trainers
# ======================================
class TraditionalLSTMMethod(LSTMTrainer):
    pass


class RegularizationLSTMMethod(LSTMTrainer):
    def __init__(self, lambda_val=0.01, **kwargs):
        super().__init__(**kwargs)
        self.lambda_val = lambda_val

    def train(self, train_loader):
        self.model.train()

        for X_batch, y_batch, lengths, _ in train_loader:
            # Move to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(X_batch, lengths)
            logits_flat = logits.data

            loss = F.cross_entropy(logits_flat, y_batch.long())

            # L2 regularization
            l2_reg = self.lambda_val * sum(
                p.pow(2).sum() for p in self.model.parameters()
            )
            loss += l2_reg

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()


class DROLSTMMethod(LSTMTrainer):
    def __init__(self, kappacoef=1.0, wasserstein=18.0, kappa=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.kappacoef = kappacoef
        self.wasserstein = wasserstein
        self.kappa = kappa

    def train(self, train_loader):
        self.model.train()

        # Define an adaptive lambda parameter for DRO
        lambda_param_raw = nn.Parameter(
            torch.tensor(1.0, device=self.device, requires_grad=True)
        )
        optimizer_lambda = optim.Adam(
            [lambda_param_raw], lr=self.optimizer.param_groups[0]["lr"] * 0.1
        )

        for X_batch, y_batch, lengths, _ in train_loader:
            # Move to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()
            optimizer_lambda.zero_grad()

            logits = self.model(X_batch, lengths)
            logits_flat = logits.data

            loss = F.cross_entropy(logits_flat, y_batch.long())

            # DRO-specific terms
            l2_norm = torch.sqrt(sum(p.pow(2).sum() for p in self.model.parameters()))
            lambda_param = torch.exp(lambda_param_raw)

            if y_batch.numel() > 0:
                valid_logits = logits_flat
                valid_labels = y_batch.long()

                # Get true class logits and max other class logits
                y_onehot = F.one_hot(
                    valid_labels, num_classes=valid_logits.size(-1)
                ).float()
                true_logits = torch.sum(y_onehot * valid_logits, dim=1)
                max_other_logits = torch.max((1 - y_onehot) * valid_logits, dim=1)[0]

                # DRO margin constraint
                margins = true_logits - max_other_logits - lambda_param * self.kappa
                label_uncertainty_term = torch.relu(margins).mean()

                # Penalty for lambda constraint
                penalty = torch.relu(l2_norm - lambda_param).pow(2)

                loss += self.kappacoef * label_uncertainty_term
                loss += self.wasserstein * lambda_param
                loss += 500 * penalty

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            optimizer_lambda.step()

            # Projection to maintain stability
            with torch.no_grad():
                lambda_param_raw.data = torch.log(
                    torch.max(torch.exp(lambda_param_raw), l2_norm + 1e-6)
                )
                lambda_param_raw.data.clamp_(min=-10)


# ======================================
# 4. Optuna Tuning
# ======================================
def tune_lstm_model_with_optuna_whole(
    model_class,
    train_loader,
    val_loader,
    test_loader,
    param_ranges,
    fixed_params=None,
    n_trials=60,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        params["num_epochs"] = 300

        model = model_class(
            input_dim=train_loader.dataset.X_list[0].shape[-1], num_classes=4, **params
        )
        model.model = model.model.to(device)
        model.train(train_loader)

        val_auc, _, _, _ = model.evaluate(val_loader)
        return val_auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = {**fixed_params, **study.best_params}
    model = model_class(
        input_dim=train_loader.dataset.X_list[0].shape[-1], num_classes=4, **best_params
    )
    model.model = model.model.to(device)
    model.train(train_loader)

    test_auc, test_probs, test_subject_ids, test_targets = model.evaluate(test_loader)

    results = {
        "auc_all": test_auc,
        "prob_all": test_probs,
        "subject_ids": test_subject_ids,
        "targets_all": test_targets,
    }
    print("[Final Whole Test Results]")
    print("Best Validation AUC:", study.best_value)
    print("Final Test AUC:", test_auc)

    return (model, results), study


# ======================================
# 5. Dataset Definition
# ======================================
class RNNDataset(Dataset):
    def __init__(self, X_list, y_list, subject_id_list=None):
        self.X_list = X_list
        self.y_list = y_list
        self.subject_id_list = subject_id_list or [None] * len(X_list)

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        return (
            self.X_list[idx],
            self.y_list[idx],
            len(self.X_list[idx]),
            self.subject_id_list[idx],
        )


def collate_fn(batch):
    X_batch, y_batch, lengths, subject_ids = zip(*batch)
    lengths = torch.tensor(lengths)

    X_batch = pad_sequence(X_batch, batch_first=True)
    y_batch = torch.cat(y_batch)
    return X_batch, y_batch, lengths, subject_ids
