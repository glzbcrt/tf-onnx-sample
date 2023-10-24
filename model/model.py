import tensorflow as tf
import tf2onnx
import onnx


def save_classes(file: str, classes: list):
    """_summary_

    Args:
        file (str): _description_
        classes (list): _description_
    """
    with open(file, "w") as tmp:
        for className in classes:
            tmp.write(className + "\n")


def build_model(output_classes: int, img_width: int, img_height: int, img_channels: int):
    """_summary_

    Args:
        output_classes (int): _description_
        img_width (int): _description_
        img_height (int): _description_
        img_channels (int): _description_

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
    """_summary_

    Args:
        model (_type_): _description_
        epochs (int): _description_
        training_ds (_type_): _description_
        validation_ds (_type_): _description_
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
    """_summary_

    Args:
        model (_type_): _description_
        path (str): _description_
        shape (_type_): _description_
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
