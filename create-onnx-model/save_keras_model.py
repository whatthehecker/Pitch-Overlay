"""
Exports a CREPE TensorFlow model as a SavedModel.
"""

import argparse
import crepe.core
from pathlib import Path

OUTPUTS_PATH = Path('outputs')


def export_crepe_model(args: argparse.Namespace) -> None:
    model_capacity = args.size
    model = crepe.core.build_and_load_model(model_capacity)

    OUTPUTS_PATH.mkdir(exist_ok=True)
    model.export(str(OUTPUTS_PATH / 'keras_assets'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', required=True, choices=crepe.core.models.keys(), type=str,
                        help='Which CREPE model to export.')
    export_crepe_model(parser.parse_args())


if __name__ == '__main__':
    main()
