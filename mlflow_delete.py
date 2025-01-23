import mlflow
from mlflow.tracking import MlflowClient

def delete_experiments(experiments_to_keep=None):
    if experiments_to_keep is None:
        experiments_to_keep = {"Default", "pollution_prediction"}
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Get all experiments
    experiments = client.search_experiments()
    
    # Delete experiments not in keep list
    for experiment in experiments:
        if experiment.name not in experiments_to_keep:
            try:
                print(f"Deleting experiment: {experiment.name}")
                experiment_id = experiment.experiment_id  # Using experiment_id instead of id
                client.delete_experiment(experiment_id)
                print(f"Successfully deleted experiment: {experiment.name}")
            except Exception as e:
                print(f"Error deleting experiment {experiment.name}: {e}")

if __name__ == "__main__":
    # Example usage
    experiments_to_keep = {"Default", "pollution_prediction"}  # Example names
    delete_experiments(experiments_to_keep)