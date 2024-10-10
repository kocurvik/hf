import cv2
from PIL import Image

import os
import piexif


from dataset_utils.data import is_image


def load_rotated_image(image_path):
    img = cv2.imread(image_path)
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img

# def load_rotated_image(image_path):
#     # Load image using PIL to get EXIF data
#     image_pil = Image.open(image_path)
#
#     # Try to get EXIF orientation tag
#     try:
#         exif = image_pil._getexif()
#         orientation = exif.get(274)  # Orientation tag is 274
#     except (AttributeError, KeyError, IndexError):
#         # No EXIF data or no orientation tag
#         orientation = None
#
#     # Load the image using OpenCV
#     # image_cv = cv2.iread(imagde_path, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#     image_cv = cv2.imread(image_path)#, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#
#     print(orientation)
#     # Rotate image based on orientation if needed
#     # Correct orientation based on the EXIF orientation tag
#     if orientation == 2:
#         image_cv = cv2.flip(image_cv, 1)  # Flip horizontally
#     elif orientation == 3:
#         image_cv = cv2.rotate(image_cv, cv2.ROTATE_180)
#     elif orientation == 4:
#         image_cv = cv2.flip(image_cv, 0)  # Flip vertically
#     elif orientation == 5:
#         image_cv = cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         image_cv = cv2.flip(image_cv, 1)  # Rotate 90° CCW and flip horizontally
#     elif orientation == 6:
#         image_cv = cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
#     elif orientation == 7:
#         image_cv = cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
#         image_cv = cv2.flip(image_cv, 1)  # Rotate 90° CW and flip horizontally
#     elif orientation == 8:
#         image_cv = cv2.rotate(image_cv, cv2.ROTATE_90_COUNTERCLOCKWISE)
#
#     return image_cv

# def load_rotated_image(image_path):
#     image_cv = cv2.imread(image_path)
#     heigth, width = image_cv.shape[:2]
#
#     if heigth > width:
#         image_cv = cv2.rotate(image_cv, cv2.ROTATE_90_CLOCKWISE)
#
#     return image_cv

# Folder containing the .jpeg images
image_folder = "/path/to/your/folder"

# Function to update EXIF orientation
def update_exif_orientation(image_path, target):
    try:
        # Open the image
        image = Image.open(image_path)

        # Get the current EXIF data
        exif_dict = piexif.load(image.info["exif"])

        # Change the orientation to "Right Top"
        exif_dict["0th"][piexif.ImageIFD.Orientation] = target

        # Convert the updated EXIF back to bytes
        exif_bytes = piexif.dump(exif_dict)

        piexif.insert(exif_bytes, image_path)
        print(f"Updated orientation for {image_path}")

    except Exception as e:
        print(f"Failed to update {image_path}: {e}")

# Iterate over all .jpeg images in the folder
def fix_folder(image_folder, target):
    for filename in os.listdir(image_folder):
        if is_image(filename):
            image_path = os.path.join(image_folder, filename)
            update_exif_orientation(image_path, target)

if __name__ == '__main__':
    fix_folder('D:/Research/data/H3vf/exif_rot_scenes/IPhoneZBHBack/Book', 1)
