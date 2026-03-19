from pathlib import Path
def get_config():
    return {
        "model_name": "t5-small",   # pretrained model

        "batch_size": 4,
        "num_epochs": 10,
        "lr": 3e-4,
        "max_len": 300,

        "train_file": "train.csv",
        "test_file": "test.csv",

        "output_dir": "outputs"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])  