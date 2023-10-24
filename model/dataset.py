from keras.utils import image_dataset_from_directory
from pillow_heif import register_heif_opener
from pathlib import Path
from PIL import Image
import cv2


def extract_frames_from_mp4(file: Path, stride: int = 10):
    """Extract N frames as JPG files at every @stride.

    Args:
        file (Path): MP4 file path.
        stride (int, optional): step to increment the frame to be extracted.
    """

    done = f"{file.parent}/{file.stem}.frames"

    if (Path.exists(Path(done))):
        return

    video = cv2.VideoCapture(str(file))
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(0, int(frames), stride):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = video.read()

        cv2.imwrite(f"{file.parent}/{file.stem}_{i}.jpg", frame)

    video.release()

    f = open(done, "a")
    f.close()


def convert_heic_to_jpg(file: Path):
    """Convert a HEIC file to JPG. The file will be saved in the same directory as the HEIC file, but with the JPG extension.

    Args:
        file (Path): HEIC file path.
    """

    new = f"{file.parent}/{file.stem}.jpg"

    if (Path.exists(Path(new))):
        return

    image = Image.open(file)
    image.save(new)


def get_dataset(directoy: Path, ds_type: str, batch_size: int, image_width: int, image_height: int):
    """Create a Tensorflow dataset from the specified directory.

    Args:
        directoy (Path): directory where the dataset is located.
        ds_type (str): dataset type. Can be training or validation.
        batch_size (int): how many images we should return each time.
        image_width (int): the width of the image to be returned.
        image_height (int): the height of the image to be returned.
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
    """Process HEIC and MP4 files located in the dataset directory.

    Args:
        directory (Path): dataset directory.
    """

    for file in directory.glob("**/*.mp4"):
        extract_frames_from_mp4(file)

    for file in directory.glob("**/*.heic"):
        convert_heic_to_jpg(file)


def init():
    """Initialize the dataset module."""

    register_heif_opener()
