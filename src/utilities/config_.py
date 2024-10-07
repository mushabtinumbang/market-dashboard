import os
from pathlib import Path
import src.utilities.utils as utils
# Paths
root_path = Path(__file__).parent.parent.parent.resolve()
config_path = root_path / "configs"
img_path = root_path / "img"
model_path = root_path / "model"
distilbert_model_path = model_path / "distilbert"
train_data_path = root_path / "data" / "train"
scrape_data_path = root_path / "data" / "scrape"
summarized_data_path = root_path / "data" / "summarized"
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
        self.params = utils.read_yaml(os.path.join(config_path, "main_config.yaml"))