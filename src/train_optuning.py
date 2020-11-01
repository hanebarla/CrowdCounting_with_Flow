import optuna
import train


TRIAL_SIZE = 100


def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-2)
    loss = train.train(lr=lr, wd=weight_decay)

    return loss


def optuning_hyper(obj):
    study = optuna.create_study()
    study.optimize(obj, n_trials=TRIAL_SIZE)

    return study


if __name__ == "__main__":
    study = optuning_hyper(objective)
    print(study.best_params)
