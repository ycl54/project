import torch
from dataset import Data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity
from autoencoder import AEModel
from loss_ssim import SSIM
import copy
from torch.autograd import Variable

def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        # resmap = np.expand_dims(resmap, axis=-1)
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return resmaps
def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2

    return resmaps

def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if(imgs_input.shape[1]==3):
        imgs_input_gray = imgs_input[:,0,:,:].reshape(imgs_input.shape[0], imgs_input.shape[2], imgs_input.shape[3])
        imgs_pred_gray = imgs_pred[:,0,:,:].reshape(imgs_pred.shape[0], imgs_pred.shape[2], imgs_pred.shape[3])
    else:
        imgs_pred_gray = imgs_pred.reshape(imgs_pred.shape[0], imgs_pred.shape[2], imgs_pred.shape[3])
        imgs_input_gray = imgs_input.reshape(imgs_input.shape[0], imgs_input.shape[2], imgs_input.shape[3])
    # calculate remaps
    if method == "l2":
        resmaps = resmaps_l2(imgs_input_gray, imgs_pred_gray)
    elif method in ["ssim", "mssim"]:
        resmaps = resmaps_ssim(imgs_input_gray, imgs_pred_gray)

    return resmaps

def label_images(images_th):
    """
    Segments images into images of connected components (regions).
    Returns segmented images and a list of lists, where each list
    contains the areas of the regions of the corresponding image.

    Parameters
    ----------
    images_th : array of binary images
        Thresholded residual maps.
    Returns
    -------
    images_labeled : array of labeled images
        Labeled images.
    areas_all : list of lists
        List of lists, where each list contains the areas of the regions of the corresponding image.
    """
    images_labeled = np.zeros(shape=images_th.shape)
    areas_all = []
    for i, image_th in enumerate(images_th):
        # close small holes with binary closing
        # bw = closing(image_th, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(image_th)

        # label image regions
        image_labeled = label(cleared)

        # image_labeled = label(image_th)

        # append image
        images_labeled[i] = image_labeled

        # compute areas of anomalous regions in the current image
        regions = regionprops(image_labeled)

        if regions:
            areas = [region.area for region in regions]
            areas_all.append(areas)
        else:
            areas_all.append([0])

    return images_labeled, areas_all

def is_defective(areas, min_area):
    """Decides if image is defective given the areas of its connected components"""
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0

def predict_class(resmap, min_area, threshold):
    resmap_th  = resmap > threshold
    _, areas_all = label_images(resmap_th)
    y_pred = [is_defective(areas, min_area) for areas in areas_all]

    return y_pred


transform2 = transforms.Compose([transforms.Resize((928, 928)),
                                 transforms.Grayscale(1),
                                 transforms.ToTensor()])

test_ds = Data('test','./mvtec/bottle', transform2)
test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)
min_area = 400
threshold = 0.88
for sample in test_dl:
    y_true = sample['class'].numpy()
    imgs_input = sample['image'].numpy()
    print(imgs_input.shape)
    model = torch.load("trained_model.pt")
    input = sample['image'].cuda()
    model.eval()
    with torch.no_grad():
        imgs_pred = model(input)

    imgs_pred = imgs_pred.cpu().numpy()
    print(imgs_pred.shape)
    resmaps = calculate_resmaps(imgs_input, imgs_pred, "ssim")

    pred = predict_class(resmap=resmaps, min_area=min_area, threshold=threshold)
    print(pred)
