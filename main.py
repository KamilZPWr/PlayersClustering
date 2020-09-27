from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

from consts import LINE_WIDTH, TEAM, DETECTION_ID, Y1, Y0, X1, X0, \
    LABELED_IMAGES, MASKS, IMAGES, ANNOTATIONS, COLORS
from utils import _collect_annotation_paths, _load_masks, divide_paths_to_games, flatten_dicts_dict


def extract_frame_features(masks: np.array, image: Image) -> Dict[int, float]:
    """
    Extracts features for given image based on given masks
    :param masks: array with shape (n_of_masks, image_width, image_height)
    :param image: image corresponding to the given masks
    :return: dict like
                {
                    mask_n: P-mode pixel value (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)
                    ...
                }
    """
    masks_n = masks.shape[0]
    image_array = np.asarray(image.convert('RGB'))
    peoples = {}
    for mask_id in range(masks_n):
        peoples[mask_id] = np.mean(image_array[masks[mask_id]], axis=0)
    return peoples


def extract_game_features(annotations: List[str]) -> Dict[str, Dict[int, float]]:
    """
    Extracts features for each game's frame
    :param annotations: List of all available annotations
    :return: Dict like
                {
                    frame_n_annotation_path: {
                        mask_n: P-mode pixel value
                        ...
                    }
                    ...
                }
    """
    results = {}
    for annotation in annotations:
        image_path = annotation.replace(ANNOTATIONS, IMAGES).replace('.csv', '.jpg')
        masks_path = annotation.replace(ANNOTATIONS, MASKS).replace('.csv', '.joblib')

        masks = _load_masks(masks_path)
        image = Image.open(image_path)

        results[annotation] = extract_frame_features(masks, image)

    return results


def process_predictions(features: Dict[int, float], model: KMeans,
                        annotation: pd.DataFrame, image: Image.Image) -> Tuple[pd.DataFrame, Image.Image]:
    """
    Predicts team for given frame, draw detection box on image, update annotation

    :param features: Dict with mask_id and feature value
    :param model: Trained KMeans model
    :param annotation: annotation DataFrame to update
    :param image: image to draw on
    :return: Tuple of updated annotation and image
    """
    for mask_id, feature in features.items():
        prediction = model.predict(feature.reshape(1, -1))[0]
        row_annotation = annotation.iloc[mask_id]
        row_id = row_annotation[DETECTION_ID]
        annotation.loc[annotation[DETECTION_ID] == row_id, TEAM] = prediction

        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (
                (row_annotation[X0], row_annotation[Y0]),
                (row_annotation[X1], row_annotation[Y1])
            ),
            outline=COLORS[prediction],
            width=LINE_WIDTH
        )
    return annotation, image


def main():
    annotation_paths = _collect_annotation_paths()
    games_annotations = divide_paths_to_games(annotation_paths)

    for game_alias, game_annotations in games_annotations.items():

        game_features = extract_game_features(
            game_annotations
        )

        flattened_features = flatten_dicts_dict(game_features)
        flattened_features_without_outliers = np.array(flattened_features)

        model = KMeans(n_clusters=2).fit(flattened_features_without_outliers)

        for frame_id, frame_features in game_features.items():
            annotation = pd.read_csv(frame_id)
            image_path = frame_id.replace(ANNOTATIONS, IMAGES).replace('.csv', '.jpg')
            image = Image.open(image_path)

            annotation, image = process_predictions(frame_features, model, annotation, image)

            annotation.to_csv(frame_id.replace(ANNOTATIONS, 'predictions'))
            image.save(frame_id.replace(ANNOTATIONS, LABELED_IMAGES).replace('.csv', '.jpg'), "JPEG")


if __name__ == '__main__':
    main()
