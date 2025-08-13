import os
import sys
import argparse
from typing import List

from psx_ai_advisor.logging_config import setup_logging, get_logger
from psx_ai_advisor.data_storage import DataStorage
from psx_ai_advisor.ml_predictor import MLPredictor
from psx_ai_advisor.exceptions import InsufficientDataError, MLPredictorError


def discover_symbols(data_dir: str) -> List[str]:
    symbols = []
    if not os.path.isdir(data_dir):
        return symbols
    for name in os.listdir(data_dir):
        if not name.endswith('_HISTORICAL_DATA.csv'):
            continue
        base = name[:-len('_HISTORICAL_DATA.csv')]
        if base:
            symbols.append(base)
    symbols.sort()
    return symbols


def main():
    parser = argparse.ArgumentParser(prog='train_models', add_help=True)
    parser.add_argument('--symbols', type=str, default='', help='Comma-separated list of symbols to train')
    parser.add_argument('--all', action='store_true', help='Train all symbols discovered in the data directory')
    parser.add_argument('--model-type', type=str, default='ensemble', choices=['ensemble', 'random_forest'], help='Model type to train')
    parser.add_argument('--optimize', action='store_true', default=True, help='Enable hyperparameter optimization')
    parser.add_argument('--no-optimize', action='store_false', dest='optimize', help='Disable hyperparameter optimization')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of symbols to train')
    args = parser.parse_args()

    setup_logging()
    logger = get_logger(__name__)

    storage = DataStorage()
    data_dir = storage.storage_config.get('data_directory', 'data')
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), data_dir)

    selected_symbols: List[str] = []
    if args.all:
        selected_symbols = discover_symbols(data_dir)
    if args.symbols:
        extra = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
        selected_symbols.extend(extra)
    seen = set()
    symbols = []
    for s in selected_symbols:
        if s not in seen:
            symbols.append(s)
            seen.add(s)
    if args.limit and args.limit > 0:
        symbols = symbols[:args.limit]
    if not symbols:
        logger.info('No symbols provided or discovered')
        sys.exit(0)

    predictor = MLPredictor(model_type='ensemble' if args.model_type.lower() == 'ensemble' else 'RandomForest')

    success = 0
    failures = 0
    for sym in symbols:
        try:
            logger.info(f'Training {sym} with model_type={predictor.model_type}, optimize={args.optimize}')
            result = predictor.train_model(sym, optimize_params=args.optimize)
            mt = result.get('model_type', predictor.model_type)
            acc = result.get('accuracy')
            logger.info(f'Trained {sym}: model_type={mt}, accuracy={acc}')
            success += 1
        except (InsufficientDataError, MLPredictorError, Exception) as e:
            logger.error(f'Failed training {sym}: {e}')
            failures += 1

    logger.info(f'Training completed: success={success}, failures={failures}, total={len(symbols)}')


if __name__ == '__main__':
    main()


