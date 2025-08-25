# Packages import
import requests
from pathlib import Path

def read_image(dir_path: str, img_name:str) -> tuple[bytes, str] | tuple[None, None]:
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
    if file_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
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
    response = requests.post(url = api_url, data = data, headers = headers)
    return response.json()
  except Exception as e:
    print(e)