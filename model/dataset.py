from pillow_heif import register_heif_opener
from pathlib import Path
from PIL import Image
import cv2

from keras.utils import image_dataset_from_directory


def extract_frames_from_mp4(file: Path, frame: int = 30, stride: int = 10):
    """_summary_

    Args:
        file (Path): _description_
        frame (int, optional): _description_. Defaults to 30.
        stride (int, optional): _description_. Defaults to 10.
    """

    done = f"{file.parent}/{file.stem}.frames"

    if (Path.exists(Path(done))):
        return

    video = cv2.VideoCapture(str(file))
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(0, int(frames), 10):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = video.read()

        cv2.imwrite(f"{file.parent}/{file.stem}_{i}.jpg", frame)

    video.release()

    f = open(done, "a")
    f.close()


def convert_heic_to_jpg(file: Path):
    """_summary_

    Args:
        file (Path): _description_
    """

    new = f"{file.parent}/{file.stem}.jpg"

    if (Path.exists(Path(new))):
        return

    image = Image.open(file)
    image.save(new)


def get_dataset(directoy: Path, ds_type: str, batch_size: int, image_width: int, image_height: int):
    """_summary_

    Args:
        directoy (Path): _description_
        ds_type (str): _description_
        batch_size (int): _description_
        image_width (int): _description_
        image_height (int): _description_

    Returns:
        _type_: _description_
    """

    return image_dataset_from_directory(
        str(directoy),
        validation_split=0.2,
        subset=ds_type,
        seed=123,
        image_size=(image_height, image_width),
        batch_size=batch_size
    )


def process_assets(directory: Path):
    """_summary_

    Args:
        directory (Path): _description_
    """

    for file in directory.glob("**/*.mp4"):
        extract_frames_from_mp4(file)

    for file in directory.glob("**/*.heic"):
        convert_heic_to_jpg(file)


def init():
    """_summary_
    """

    register_heif_opener()
