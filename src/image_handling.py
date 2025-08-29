# Packages import
import requests
from pathlib import Path
from PIL import Image
import base64
import io
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score

# Contains all label index used by the model.
CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17,
}


def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size


def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros(
        (height, width), dtype=np.uint8
    )  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result["mask"], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            # Apply background only where no other class has been assigned yet
            # This logic might need adjustment based on how the model defines 'Background'
            # For this model, it seems safer to just let non-background overwrite it first.
            # A simple application like this should be fine: if Background mask says pixel is BG, set it to 0.
            # However, a more robust way might be to only set to background if combined_mask is still 0 (initial value)
            combined_mask[mask_array > 0] = 0  # Class ID for Background is 0

    return combined_mask


def read_image(image_path: Path) -> tuple[bytes, str] | tuple[None, None]:
    """
    Returns the image data and it's extension.

    Args:
      image_path (Path): Path to the image file.

      Returns:
        tuple: (image byte data, image extension), or None if the image does not exists or if it's not an image.
    """
    try:
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            # Raise an error if the file is not an image.
            raise ValueError(
                f"{os.path.basename(image_path)} is not a supported image format."
            )

        return (image_path.read_bytes(), image_path.suffix[1:])

    except FileNotFoundError:
        print(f"File not found: {os.path.basename(image_path)}")
        return (None, None)
    except Exception as e:
        print(f"Error reading {os.path.basename(image_path)}: {e}")
        return (None, None)


def segmentation_query(
    data: bytes, api_url: str, headers: dict[str, str]
) -> dict | None:
    """
    Sends the image data to the segformer_b3_clothes API.
    Gets the masks and data results.

    Args:
      data (bytes): Image binary data.
      api_url (str): URL to send the data to.
      headers (dict): Headers of the request. Must contains Authorization and Content-Type.

    Returns:
        dict: JSON received from the API. None in case of failure.
    """
    # Trying a maximum of 3 times
    for attempt in range(3):
        try:
            response = requests.post(
                url=api_url, data=data, headers=headers, timeout=30
            )

            # Raise an error in case of failure
            response.raise_for_status()

            return response.json()

        except requests.HTTPError as e:
            # Too many requests error
            if response.status_code == 429:
                print(f"Too many requests. Retrying in 5s. (attempt {attempt}/3)")
                time.sleep(5)
                continue
            # Return None otherwise
            else:
                print(f"HTTP error {response.status_code}: {e}")
                return None

        except requests.RequestException as e:
            print(f"Request error: {e}")
            if attempt < 2:
                time.sleep(2)
                continue
            return None

        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    print("All retry attemps failed.")
    return None


def resize_image(
    image_path_str: str, image_directory: str, resized_directory: str
) -> Path:
    """
    Resize an image to reduce it's size.

    Args:
        image_path (Path): Path to the image file.
        resized_directory (str): Directory that receive the resized images.

    Returns:
        Path: Path to the resized image.
    """
    max_size = 512, 512

    image_path = Path(image_directory) / image_path_str

    filename, ext = os.path.splitext(os.path.basename(image_path))

    with Image.open(image_path) as image:
        image.thumbnail(max_size)
        image.save(resized_directory + "/" + filename + "_resized" + ext, ext[1:])

    return Path(resized_directory) / str(filename + "_resized" + ext)


def show_result(
    original_image: Path, mask_image: Path, combined_masks: np.ndarray
) -> None:
    """
    Shows the result of the segmentation in a plot.

    Args:
        original_image (Path): Original image path to display.
        mask_image (Path): Original mask path to display.
        combined_masks (np.ndarray): Combined segmentation mask with class indices.
    """
    iou, dice = mask_evaluation(mask_image, combined_masks)
    
    with Image.open(original_image) as image:
        with Image.open(mask_image) as mask:            
            plt.figure(figsize=(15, 5))

            # Position 1
            plt.subplot(1, 4, 1)
            plt.imshow(image)
            plt.title("Original image")
            plt.axis("off")

            # Position 2
            combined_masks_wo_background = np.ma.masked_where(
                combined_masks == 0, combined_masks
            )
            plt.subplot(1, 4, 2)
            plt.imshow(image)
            plt.imshow(combined_masks_wo_background, alpha=0.7)
            plt.title("Image mask with original background")
            plt.axis("off")

            # Position 3
            plt.subplot(1, 4, 3)
            plt.imshow(combined_masks)
            plt.title("Predicted mask")
            plt.axis("off")

            # Position 4
            plt.subplot(1, 4, 4)
            plt.imshow(mask)
            plt.title("Original mask")
            plt.axis("off")

            plt.suptitle(f"IoU (Jaccard): {iou:.3f} | Dice (F1): {dice:.3f}", fontsize=16)
            plt.tight_layout()
            plt.show()


def segment_images_batch(
    list_of_images_paths: list[str], image_directory: str, resized_directory: str
) -> tuple[list[np.ndarray], list[str]]:
    """
    Handles all the logic to: 
    - Optimize the images to send to the API.
    - Read the images.
    - Send to the API.
    - Create the masks.
    - Returns the results.

    Args:
        list_of_images_paths (list[str]): List of the images to process.
        image_directory (str): Original images directory path.
        resized_directory (str): Directory that receive the resized images.
    
    Returns:
        tuple[list[np.ndarray], list[str]]: List of the calculated masks from the API, and the list of resized images.
    """
    list_of_resized_images_paths = []
    batch_segmentations = []
    API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

    # Loading .env file
    load_dotenv()
    API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

    print(f"Start batching {len(list_of_images_paths)} images.")

    for image_path in tqdm(list_of_images_paths, desc="Segmenting images"):
        # Resizing images to reduce process time.
        resized_image_path = resize_image(
            image_path, image_directory, resized_directory
        )
        list_of_resized_images_paths.append(resized_image_path)

        # Get the image data and extension
        image_data, image_extension = read_image(resized_image_path)

        # Preparing the requests headers.
        headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": f"image/{image_extension}",
        }

        # Query the API.
        api_result = segmentation_query(
            data=image_data, api_url=API_URL, headers=headers
        )

        # Get image size
        image_width, image_height = get_image_dimensions(resized_image_path)
        # Create masks
        combined_masks = create_masks(api_result, image_width, image_height)

        batch_segmentations.append(combined_masks)
        time.sleep(1)

    return batch_segmentations, list_of_resized_images_paths


def mask_evaluation(
    original_mask: Path, calculated_mask_data: np.ndarray
) -> tuple[float, float]:
    """
    Evaluates the precision of the model results.

    Args:
        original_mask (Path): Path to the mask file.
        calculated_mask_data (np.ndarray): Calculated mask from the model.

    Returns:
        tuple[float, float]: Tuple with the IoU and Dice score.

    """
    with Image.open(original_mask) as original_mask_data:
        y_true = np.array(original_mask_data).ravel()
        y_pred = calculated_mask_data.ravel()

        iou = jaccard_score(y_true, y_pred, average="micro")
        dice = f1_score(y_true, y_pred, average="micro")

    return iou, dice
