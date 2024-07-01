import os
from pathlib import Path
from src.utilities.utils import read_yaml
# Paths
root_path = Path(__file__).parent.parent.parent.resolve()
config_path = root_path / "configs"
img_path = root_path / "img"
model_path = root_path / "model"
finbert_model_path = model_path / "finbert"
train_data_path = root_path / "data" / "train"
scrape_data_path = root_path / "data" / "scrape"
predicted_data_path = root_path / "data" / "predicted"
combined_data_path = root_path / "data" / "combined"
log_path = root_path / "logs"

class ConfigManager(object):
    """
    Config Manager to manage main configurations
    and store them as attributes depending on
    the environment
    """

    def __init__(self, config_file="main_config.yaml"):
        # load main_config
        self.params = read_yaml(os.path.join(config_path, "main_config.yaml"))