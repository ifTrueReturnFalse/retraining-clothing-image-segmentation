# Packages import
import requests
from pathlib import Path
from PIL import Image
import base64
import io
import numpy as np

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


def read_image(dir_path: str, img_name: str) -> tuple[bytes, str] | tuple[None, None]:
    """
    Returns the image data and it's extension.

    Args:
      dir_path (str): Directory path to the image.
      img_name (str): Name of the image.

      Returns:
        tuple: (image byte data, image extension), or None if the image does not exists or if it's not an image.
    """
    try:
        file_path = Path(dir_path) / img_name
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            # Raise an error if the file is not an image.
            raise ValueError(f"{img_name} is not a supported image format.")

        return (file_path.read_bytes(), file_path.suffix[1:])

    except FileNotFoundError:
        print(f"File not found: {img_name}")
        return (None, None)
    except Exception as e:
        print(f"Error reading {img_name}: {e}")
        return (None, None)


def segmentation_query(data: bytes, api_url: str, headers: dict[str, str]):
    """
    Sends the image data to the segformer_b3_clothes API.
    Gets the masks and data results.

    Args:
      data (bytes): Image binary data.
      api_url (str): URL to send the data to.
      headers (dict): Headers of the request. Must contains Authorization and Content-Type.
    """
    try:
        response = requests.post(url=api_url, data=data, headers=headers)
        return response.json()
    except Exception as e:
        print(e)
