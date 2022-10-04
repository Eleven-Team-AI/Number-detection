import os.path
from pathlib import Path
import logging, logging.config, yaml

logging.config.dictConfig(
    yaml.load(
        open(
            os.path.join(Path(__file__).parents[0], 'config', 'logging.conf')
        ),
        Loader=yaml.FullLoader
    )
)