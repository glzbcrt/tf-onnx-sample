import tensorflow as tf
import tf2onnx
import onnx


def save_classes(file: str, classes: list):
    """Save the image classes to a file for later use.

    Args:
        file (str): file where the classes should be saved.
        classes (list): list of classes to save.
    """
    with open(file, "w") as tmp:
        for className in classes:
            tmp.write(className + "\n")


def build_model(output_classes: int, img_width: int, img_height: int, img_channels: int):
    """Build the Tensorflow model to be used for training.

    Args:
        output_classes (int): number of output classes.
        img_width (int): input image width.
        img_height (int): input image heigth.
        img_channels (int): input image channels.

    Returns:
        _type_: _description_
    """
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(
            1./255, input_shape=(img_height, img_width, img_channels)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(output_classes), name="output")
    ])


def train_model(model, epochs: int, batch_size: int, training_ds, validation_ds):
    """Train the model.

    Args:
        model (_type_): model to be trainned.
        epochs (int): number of epochs to train.
        training_ds (_type_): trainning dataset to use.
        validation_ds (_type_): validation dataset to use.
    """
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=epochs
    )


def save_onnx(model, path: str, shape):
    """Save the trainned model as ONNX.

    Args:
        model (_type_): model to be saved.
        path (str): path where the model should be saved.
        shape (_type_): shape of the model input.
    """
    input_signature = [
        tf.TensorSpec(
            shape,
            tf.float32,
            name='input'
        )
    ]

    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature,
        opset=13
    )

    onnx.save(
        onnx_model,
        path
    )
