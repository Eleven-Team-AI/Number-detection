import logging.config
import os.path
import yaml
from pathlib import Path

logging.config.dictConfig(
    yaml.load(
        open(
            os.path.join(Path(__file__).parents[0], 'config', 'logging.conf')
        ),
        Loader=yaml.FullLoader
    )
)
