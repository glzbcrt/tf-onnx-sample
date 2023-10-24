from pathlib import Path
import getopt
import sys

from model import build_model, train_model, save_onnx, save_classes
from dataset import get_dataset, init, process_assets


def main(dataset_dir: Path, image_width: int, image_height: int, image_channels: int, batch_size: int, epochs: int):
    """_summary_

    Args:
        dataset_dir (Path): _description_
        image_width (int): _description_
        image_height (int): _description_
        image_channels (int): _description_
        batch_size (int): _description_
        epochs (int): _description_
    """

    init()
    process_assets(dataset_dir)

    training_ds = get_dataset(dataset_dir, "training",
                              batch_size, image_width, image_height)
    save_classes("output/classes", training_ds.class_names)

    validation_ds = get_dataset(
        dataset_dir, "validation", batch_size, image_width, image_height)

    model = build_model(
        training_ds.class_names,
        img_width=image_width,
        img_height=image_height,
        img_channels=image_channels
    )

    train_model(
        model=model,
        epochs=epochs,
        training_ds=training_ds,
        validation_ds=validation_ds,
        batch_size=batch_size
    )

    save_onnx(
        model,
        "output/model.onnx",
        (1, image_height, image_width, image_channels)
    )


def usage():
    """_summary_
    """

    print("Usage: python model.py --dataset-dir <path>")
    sys.exit(2)


if __name__ == '__main__':
    print("Image Classification Model v1.0")
    print("MIT License")
    print()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "", [
                                   "dataset-dir=", "image-width=", "image-height=", "image-channels=", "batch-size=", "epochs="])

        for o, a in opts:
            if o in ("--dataset-dir"):
                dataset_dir = Path(a)
            elif o in ("--image-width"):
                image_width = int(a)
            elif o in ("--image-height"):
                image_height = int(a)
            elif o in ("--image-channels"):
                image_channels = int(a)
            elif o in ("--batch-size"):
                batch_size = int(a)
            elif o in ("--epochs"):
                epochs = int(a)

        if (not Path.exists(Path(dataset_dir))):
            usage()

        main(
            dataset_dir=dataset_dir,
            image_width=image_width,
            image_height=image_height,
            image_channels=image_channels,
            batch_size=batch_size,
            epochs=epochs
        )
    except getopt.GetoptError as err:
        usage()
