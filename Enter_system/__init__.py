import logging.config
import os.path
from pathlib import Path

import yaml

logging.config.dictConfig(
    yaml.load(
        open(
            os.path.join(Path(__file__).parents[0], 'config', 'logging.conf')
        ),
        Loader=yaml.FullLoader
    )
)
