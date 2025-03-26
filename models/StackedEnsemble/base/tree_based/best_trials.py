import optuna

study_name = "xgboost_optimization"
storage_url = "sqlite:///optuna_xgboost.db"
top_n = 10

def get_top_trials():
    # Load the Optuna study from persistent SQLite storage.
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # Optionally, filter trials that have finished.
    completed_trials = [trial for trial in study.trials if trial.state.is_finished()]
    
    # Sort the completed trials by their objective value in descending order
    top_trials = sorted(completed_trials, key=lambda trial: trial.value if trial.value is not None else -float("inf"), reverse=True)[:top_n]
    return top_trials

if __name__ == '__main__':
    top_trials = get_top_trials()
    print("Top 10 Trials:")
    for rank, trial in enumerate(top_trials, start=1):
        print(f"Rank {rank}: Trial #{trial.number}, Value: {trial.value}")
        print(f"         Parameters: {trial.params}")