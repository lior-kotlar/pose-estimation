
from PIL import Image
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import matplotlib.image as mpimg
from scipy.ndimage import shift
from skimage.color import rgb2gray
from scipy import ndimage
import imageio
rotation_range = 45
xy_shift = 10


# def augment(img, h_fl, v_fl, rotation_angle, shift_y_x):
#     if h_fl:
#         img = np.fliplr(img)
#     if v_fl:
#         img = np.flipud(img)
#     img = shift(img, shift_y_x)
#     img_pil = Image.fromarray(img)
#     img_pil = img_pil.rotate(rotation_angle, 3)
#     img = np.asarray(img_pil)
#     return img
#
#
# def do_augmentations(img, mask):
#     img_aug = img[:]
#     do_horizontal_flip = bool(np.random.randint(2))
#     do_vertical_flip = bool(np.random.randint(2))
#     rotation_angle = np.random.randint(-rotation_range, rotation_range)
#     shift_y_x = np.random.randint(-xy_shift, xy_shift, 2)
#     num_channels = img_aug.shape[-1]
#     for channel in range(num_channels):
#         img_aug[:, :, channel] = augment(img_aug[:, :, channel], do_horizontal_flip, do_vertical_flip, rotation_angle, shift_y_x)
#     mask_aug = augment(mask, do_horizontal_flip, do_vertical_flip, rotation_angle, shift_y_x)
#     return img_aug, mask_aug

# define a function to flip an image horizontally


def flip_horizontal(image):
    return image[:, ::-1]


# define a function to flip an image vertically
def flip_vertical(image):
    return image[::-1, :]


# define a function to translate an image by a random offset

def translate(image, shift_y_x):
    if len(image.shape) == 2:
        image = translate_channel(image, shift_y_x)
    else:
        for i in range(3):
            image[:, :, i] = translate_channel(image[:, :, i], shift_y_x)
    return image


def translate_channel(image, shift_y_x):
    # generate a random offset in x and y direction
    return shift(image, shift_y_x)


# define a function to rotate an image by a random angle
def rotate_channel(image, rotation_angle):
    # get the image shape
    img_pil = Image.fromarray(image)
    img_pil = img_pil.rotate(rotation_angle, 3)
    image = np.asarray(img_pil)
    return image


def rotate(image, rotation_angle):
    if len(image.shape) == 2:
        image = rotate_channel(image, rotation_angle)
    else:
        for i in range(3):
            image[:, :, i] = rotate_channel(image[:, :, i], rotation_angle)
    return image


def do_augmentations(image_aug, mask_aug):
    mask_aug = (mask_aug * 255).astype(np.uint8)
    mask1 = np.logical_and(mask_aug > 100, mask_aug < 200)
    mask2 = mask_aug > 200

    do_horizontal_flip = bool(np.random.randint(2))
    do_vertical_flip = bool(np.random.randint(2))

    if do_horizontal_flip:
        image_aug = flip_horizontal(image_aug)
        mask_aug = flip_horizontal(mask_aug)
        mask1 = flip_horizontal(mask1)
        mask2 = flip_horizontal(mask2)

    if do_vertical_flip:
        image_aug = flip_vertical(image_aug)
        mask_aug = flip_vertical(mask_aug)
        mask1 = flip_vertical(mask1)
        mask2 = flip_vertical(mask2)

    shift_y_x = np.random.randint(-xy_shift, xy_shift, 2)
    image_aug = translate(image_aug, shift_y_x)
    mask_aug = translate(mask_aug, shift_y_x)
    mask1 = translate(mask1, shift_y_x)
    mask2 = translate(mask2, shift_y_x)

    rotation_angle = np.random.randint(-rotation_range, rotation_range)
    image_aug = rotate(image_aug, rotation_angle)
    mask_aug = rotate(mask_aug, rotation_angle)
    mask1 = rotate(mask1, rotation_angle)
    mask2 = rotate(mask2, rotation_angle)

    return image_aug, mask1, mask2


def show_image_and_mask(image, mask):
    disp_img = image
    disp_img[:, :, 0] += mask
    disp_img[:, :, 1] += mask
    disp_img[:, :, 2] += mask
    plt.imshow(disp_img)
    plt.show()


def create_augmentations(path_of_images, path_of_masks, new_path_images, new_path_masks, n):
    """
    this fucntion takes each image from the train-set and apply some form of augmentation to it:
    a rotation, a flip, a translation and then applies the augmentation to the train_masks.
    ot creates n augmentations
    then it saves it with new names
    Args:
        path_of_images:
        path_of_masks:
        new_path_images:
        new_path_masks:
        n: number of augmentations for each image
    """
    for file in os.listdir(path_of_images):
        # get the file name
        file_name = os.path.basename(file)

        # for i in range(n):
            # image_aug = mpimg.imread(f"{path_of_images}\\{file_name}")
        image_aug = imageio.v2.imread(f"{path_of_images}\\{file_name}")
        mask_aug = rgb2gray(mpimg.imread(f"{path_of_masks}\\{file_name}", format='L')[..., :-1])

        image_aug, mask1, mask2 = do_augmentations(image_aug, mask_aug)
        mask1 = ndimage.binary_closing(mask1, iterations=3).astype(float)
        mask1 = (mask1 * 255).astype(np.uint8)
        mask2 = ndimage.binary_closing(mask2, iterations=3).astype(float)
        mask2 = (mask2 * 255).astype(np.uint8)
        # show_image_and_mask(image_aug, mask1 + mask2)
        # save_path_mask1 = f"{new_path_masks}\\masks1\\{i}{file_name}"
        # save_path_mask2 = f"{new_path_masks}\\masks2\\{i}{file_name}"
        # save_path_image = f"{new_path_images}\\{i}{file_name}"

        save_path_mask1 = f"{new_path_masks}\\wings1\\{file_name}"
        save_path_mask2 = f"{new_path_masks}\\wings2\\{file_name}"
        save_path_image = f"{new_path_images}\\{file_name}"

        imageio.imwrite(save_path_mask1, mask1)
        imageio.imwrite(save_path_mask2, mask2)
        imageio.imwrite(save_path_image, image_aug)

        print(f"saved image {file_name} augmentation number")




if __name__ == '__main__':
    path_of_images =  r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\annotations\new_images_folder"
    path_of_masks =  r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\annotations\new_train_masks"
    new_path_images = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\annotations\augmnetations\aug_1\train_images"
    new_path_masks = r"C:\Users\amita\PycharmProjects\pythonProject\vision\train_nn_project\utils\wings_segmentation\create_annotations\input\annotations\augmnetations\aug_1\train_masks"

    create_augmentations(path_of_images, path_of_masks, new_path_images, new_path_masks, 3)
