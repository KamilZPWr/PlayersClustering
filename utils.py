import glob
from typing import List, Dict, Any

import blosc
import joblib
import numpy as np


def _collect_annotation_paths(path='./team_classification/annotations/', extension='.csv'):
    return glob.glob(f'{path}*{extension}')


def _load_masks(path: str) -> np.array:
    return blosc.unpack_array(joblib.load(path))


def divide_paths_to_games(paths: List[str]) -> Dict[str, List[str]]:
    """
    Divides provided paths into groups based on game

    :param paths: paths to all annotations
    :return: Dict with groups like
                {
                    game_name: [path_to_frame_1.csv, path_to_frame_2.csv],
                    ...
                }
    """
    games = {}
    for annotation in paths:
        game = annotation.split('-')[1]
        game = game.replace('_1_po_owa', '').replace('_2_po_owa', '')
        if game in games:
            games[game].append(annotation)
        else:
            games[game] = [annotation]
    return games


def flatten_dicts_dict(features: Dict[str, Dict[int, Any]]) -> List[float]:
    """
    Converts Dict of Dict with features to list of features
    """
    features = [list(frame.values()) for frame in features.values()]
    return [item for sublist in features for item in sublist]
