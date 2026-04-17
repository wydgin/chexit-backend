import os
import json
import random
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

import tensorflow as tf


# -------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_DIR = os.path.join(BASE_DIR, "Training")

# Montgomery dataset
MONTGOMERY_IMAGE_DIR = os.path.join(TRAINING_DIR, "montgomery_cxr")
MONTGOMERY_MASK_COMB_DIR = os.path.join(
    TRAINING_DIR, "montgomery_mask", "combineMask"
)

# Shenzhen dataset
SHENZHEN_IMAGE_DIR = os.path.join(TRAINING_DIR, "shenzhen_cxr")
SHENZHEN_MASK_DIR = os.path.join(TRAINING_DIR, "shenzhen_mask")

# Preprocessed output (mirrors the Kaggle notebook structure)
SEGMENTATION_DIR = os.path.join(BASE_DIR, "segmentation")
SEGMENTATION_TEST_DIR = os.path.join(SEGMENTATION_DIR, "test")
SEGMENTATION_TRAIN_DIR = os.path.join(SEGMENTATION_DIR, "train")
SEGMENTATION_AUG_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "augmentation")
SEGMENTATION_IMAGE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "image")
SEGMENTATION_MASK_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "mask")
SEGMENTATION_DILATE_DIR = os.path.join(SEGMENTATION_TRAIN_DIR, "dilate")

# Model / training configuration
IMG_SIZE = 512
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)
BATCH_SIZE = 2
EPOCHS = 56
TRAIN_FRACTION = 0.8  # train / test split on preprocessed data

DILATE_KERNEL = np.ones((15, 15), np.uint8)

# Output artifacts
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_lung_seg_best.h5")
BEST_WEIGHTS_PATH = os.path.join(MODELS_DIR, "unet_lung_seg_best_weights.weights.h5")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "unet_lung_seg_final.h5")
HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.json")


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def setup_directories() -> None:
    ensure_dir(SEGMENTATION_DIR)
    ensure_dir(SEGMENTATION_TEST_DIR)
    ensure_dir(SEGMENTATION_TRAIN_DIR)
    ensure_dir(SEGMENTATION_AUG_DIR)
    ensure_dir(SEGMENTATION_IMAGE_DIR)
    ensure_dir(SEGMENTATION_MASK_DIR)
    ensure_dir(SEGMENTATION_DILATE_DIR)
    ensure_dir(MODELS_DIR)


# -------------------------------------------------------------------
# Data preparation (following the Kaggle notebook structure)
# -------------------------------------------------------------------

def collect_montgomery_pairs() -> List[Tuple[str, str, str]]:
    """
    Returns list of (image_path, mask_path, base_filename) for Montgomery.
    base_filename is the original image name (e.g., MCUCXR_0001_0.png).
    """
    image_files = sorted(glob(os.path.join(MONTGOMERY_IMAGE_DIR, "*.png")))
    pairs = []

    for img_path in image_files:
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        mask_name = f"{name}_comb{ext}"
        mask_path = os.path.join(MONTGOMERY_MASK_COMB_DIR, mask_name)
        if not os.path.exists(mask_path):
            # Skip if mask is missing
            continue
        pairs.append((img_path, mask_path, base))

    return pairs


def collect_shenzhen_pairs() -> List[Tuple[str, str, str]]:
    """
    Returns list of (image_path, mask_path, base_filename) for Shenzhen.
    base_filename is the original image name (e.g., CHNCXR_0001_0.png).
    """
    mask_files = sorted(glob(os.path.join(SHENZHEN_MASK_DIR, "*.png")))
    pairs = []

    for mask_path in mask_files:
        mask_base = os.path.basename(mask_path)
        base = mask_base.replace("_mask", "")
        img_path = os.path.join(SHENZHEN_IMAGE_DIR, base)
        if not os.path.exists(img_path):
            continue
        pairs.append((img_path, mask_path, base))

    return pairs


def preprocess_and_split() -> Tuple[int, int]:
    """
    - Resize images and masks to 512x512
    - Dilate masks
    - Split into train/test (TRAIN_FRACTION)
    - Write to segmentation directory following the notebook's structure.
    Returns: (num_train, num_test) counts.
    """
    setup_directories()

    mont_pairs = collect_montgomery_pairs()
    shen_pairs = collect_shenzhen_pairs()

    all_pairs = mont_pairs + shen_pairs
    if not all_pairs:
        raise RuntimeError("No image/mask pairs found in Training directories.")

    seed_everything(42)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * TRAIN_FRACTION)
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]

    # Write training data
    for img_path, mask_path, base in train_pairs:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)

        cv2.imwrite(os.path.join(SEGMENTATION_IMAGE_DIR, base), image)
        cv2.imwrite(os.path.join(SEGMENTATION_MASK_DIR, base), mask)
        cv2.imwrite(os.path.join(SEGMENTATION_DILATE_DIR, base), mask_dilate)

    # Write test data
    for img_path, mask_path, base in test_pairs:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask_dilate = cv2.dilate(mask, DILATE_KERNEL, iterations=1)

        filename, fileext = os.path.splitext(base)

        cv2.imwrite(os.path.join(SEGMENTATION_TEST_DIR, base), image)
        cv2.imwrite(
            os.path.join(
                SEGMENTATION_TEST_DIR,
                f"{filename}_mask{fileext}",
            ),
            mask,
        )
        cv2.imwrite(
            os.path.join(
                SEGMENTATION_TEST_DIR,
                f"{filename}_dilate{fileext}",
            ),
            mask_dilate,
        )

    return len(train_pairs), len(test_pairs)


# -------------------------------------------------------------------
# Generators and helper functions (adapted from the notebook)
# -------------------------------------------------------------------

def adjust_data(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1.0
    mask[mask <= 0.5] = 0.0
    return img, mask


def train_generator(
    batch_size: int,
    train_path: str,
    image_folder: str,
    mask_folder: str,
    aug_dict: dict,
    image_color_mode: str = "grayscale",
    mask_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    mask_save_prefix: str = "mask",
    save_to_dir: str = None,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 1,
):
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed,
    )

    train_gen = zip(image_generator, mask_generator)

    for img, mask in train_gen:
        img, mask = adjust_data(img, mask)
        yield img, mask


def test_load_image(test_file: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)
    return img


def test_generator(test_files: List[str], target_size: Tuple[int, int] = (256, 256)):
    for test_file in test_files:
        yield test_load_image(test_file, target_size)


def save_result(save_path: str, npyfile: np.ndarray, test_files: List[str]) -> None:
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        img = (item[:, :, 0] * 255.0).astype(np.uint8)

        filename, fileext = os.path.splitext(os.path.basename(result_file))
        result_file = os.path.join(save_path, f"{filename}_predict{fileext}")

        cv2.imwrite(result_file, img)


def add_suffix(base_file: str, suffix: str) -> str:
    filename, fileext = os.path.splitext(base_file)
    return f"{filename}_{suffix}{fileext}"


def load_validation_data(
    target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads the test set (images and dilated masks) into memory for validation.
    """
    test_files = [
        f
        for f in glob(os.path.join(SEGMENTATION_TEST_DIR, "*.png"))
        if ("_mask" not in f and "_dilate" not in f and "_predict" not in f)
    ]
    if not test_files:
        raise RuntimeError("No test images found in segmentation/test.")

    X = []
    y = []
    for img_path in test_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, target_size)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)

        dilate_path = add_suffix(img_path, "dilate")
        mask = cv2.imread(dilate_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, target_size)
        mask = mask.astype("float32") / 255.0
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        y.append(mask)

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    return X, y, test_files


# -------------------------------------------------------------------
# U-Net model and metrics (as in the notebook, extended with IoU)
# -------------------------------------------------------------------

def dice_coef(y_true, y_pred):
    K = tf.keras.backend
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + 1.0
    )


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred):
    K = tf.keras.backend
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + 1.0) / (union + 1.0)


def unet(input_size: Tuple[int, int, int] = (256, 256, 1)) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(input_size)

    conv1 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(inputs)
    conv1 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    )(pool1)
    conv2 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    )(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(pool2)
    conv3 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", padding="same"
    )(pool3)
    conv4 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", padding="same"
    )(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same"
    )(pool4)
    conv5 = tf.keras.layers.Conv2D(
        512, (3, 3), activation="relu", padding="same"
    )(conv5)

    up6 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                256, (2, 2), strides=(2, 2), padding="same"
            )(conv5),
            conv4,
        ],
        axis=3,
    )
    conv6 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", padding="same"
    )(up6)
    conv6 = tf.keras.layers.Conv2D(
        256, (3, 3), activation="relu", padding="same"
    )(conv6)

    up7 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                128, (2, 2), strides=(2, 2), padding="same"
            )(conv6),
            conv3,
        ],
        axis=3,
    )
    conv7 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(up7)
    conv7 = tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(conv7)

    up8 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                64, (2, 2), strides=(2, 2), padding="same"
            )(conv7),
            conv2,
        ],
        axis=3,
    )
    conv8 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    )(up8)
    conv8 = tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    )(conv8)

    up9 = tf.keras.layers.concatenate(
        [
            tf.keras.layers.Conv2DTranspose(
                32, (2, 2), strides=(2, 2), padding="same"
            )(conv8),
            conv1,
        ],
        axis=3,
    )
    conv9 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(up9)
    conv9 = tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    )(conv9)

    conv10 = tf.keras.layers.Conv2D(
        1, (1, 1), activation="sigmoid"
    )(conv9)

    return tf.keras.Model(inputs=[inputs], outputs=[conv10])


# -------------------------------------------------------------------
# Training and inference
# -------------------------------------------------------------------

def train_unet() -> None:
    seed_everything(42)

    num_train, num_test = preprocess_and_split()

    train_files = glob(os.path.join(SEGMENTATION_IMAGE_DIR, "*.png"))
    if not train_files:
        raise RuntimeError("No preprocessed training images found.")

    train_generator_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    train_gen = train_generator(
        BATCH_SIZE,
        SEGMENTATION_TRAIN_DIR,
        "image",
        "dilate",
        train_generator_args,
        target_size=(IMG_SIZE, IMG_SIZE),
        save_to_dir=os.path.abspath(SEGMENTATION_AUG_DIR),
    )

    X_val, y_val, _ = load_validation_data(target_size=(IMG_SIZE, IMG_SIZE))

    model = unet(input_size=INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=dice_coef_loss,
        metrics=[dice_coef, iou_coef, "binary_accuracy"],
    )

    steps_per_epoch = max(1, len(train_files) // BATCH_SIZE)

    checkpoint_best_model = tf.keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_dice_coef",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    checkpoint_best_weights = tf.keras.callbacks.ModelCheckpoint(
        BEST_WEIGHTS_PATH,
        monitor="val_dice_coef",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=[checkpoint_best_model, checkpoint_best_weights],
        validation_data=(X_val, y_val),
    )

    # Save final model and training history
    model.save(FINAL_MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f)

    # Load best weights before inference
    if os.path.exists(BEST_WEIGHTS_PATH):
        model.load_weights(BEST_WEIGHTS_PATH)

    # Inference on test set and save predicted masks
    test_files = [
        f
        for f in glob(os.path.join(SEGMENTATION_TEST_DIR, "*.png"))
        if ("_mask" not in f and "_dilate" not in f and "_predict" not in f)
    ]
    test_gen = test_generator(test_files, target_size=(IMG_SIZE, IMG_SIZE))
    results = model.predict(test_gen, len(test_files), verbose=1)
    save_result(SEGMENTATION_TEST_DIR, results, test_files)


# -------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------

def add_colored_mask(image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
    """
    Overlay a mask in red over the input image.
    """
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    if len(mask_image.shape) == 2:
        mask_color = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
    else:
        mask_color = mask_image.copy()

    mask_gray = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(mask_color, mask_color, mask=mask_gray)

    mask_coord = np.where(mask != [0, 0, 0])
    mask[mask_coord[0], mask_coord[1], :] = [255, 0, 0]

    ret = cv2.addWeighted(image_color, 0.7, mask, 0.3, 0)
    return ret


def visualize_result(
    image_path: str,
    mask_path: str,
    model_path: str = BEST_MODEL_PATH,
    target_size: Tuple[int, int] = (IMG_SIZE, IMG_SIZE),
) -> None:
    """
    Visualizes input image, ground truth mask, and predicted mask side by side.
    """
    custom_objects = {
        "dice_coef": dice_coef,
        "dice_coef_loss": dice_coef_loss,
        "iou_coef": iou_coef,
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image_resized = cv2.resize(image, target_size)
    x = image_resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0, :, :, 0]
    pred_bin = (pred > 0.5).astype(np.uint8) * 255

    pred_mask = cv2.resize(pred_bin, (image.shape[1], image.shape[0]))
    gt_mask_resized = cv2.resize(gt_mask, (image.shape[1], image.shape[0]))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.axis("off")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.imshow(add_colored_mask(image, gt_mask_resized))

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.axis("off")
    plt.imshow(add_colored_mask(image, pred_mask))

    plt.tight_layout()
    plt.show()


def main() -> None:
    train_unet()


if __name__ == "__main__":
    main()

