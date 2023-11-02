from scipy import ndimage
from scipy import signal
import  numpy as np
from  matplotlib import  pyplot as plt
from skimage.color import rgb2gray
import  imageio
import os

RGB = 2
def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    im_conv = ndimage.convolve(im,blur_filter)
    im_conv = ndimage.convolve(im_conv,blur_filter.T)
    return im_conv[::2 , ::2]



def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    k, m = im.shape
    new_arr = np.zeros((2*k, 2 * m), dtype=im.dtype)
    new_arr[::2, ::2] = im
    im_conv = ndimage.convolve(new_arr, 2*(blur_filter))
    im_conv = ndimage.convolve(im_conv, 2*(blur_filter.T))
    return  im_conv





def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    k = (filter_size - 1)/2
    conv_res = np.array([[1,1]])
    for i in range(1,filter_size -1):
        conv_res = signal.convolve2d(np.array([[1,1]]),conv_res)
    conv_res = conv_res/(np.power(2,2*k))
    pyr = [im]
    for j in range(max_levels-1):
        if pyr[j].shape[1]/2 <16 or pyr[j].shape[0]/2 <16 :
            break
        pyr.append(reduce(pyr[j], conv_res))
    return pyr, conv_res









def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    k = (filter_size - 1) / 2
    conv_res = np.array([[1, 1]])
    for i in range(1, filter_size - 1):
        conv_res = signal.convolve2d(np.array([[1, 1]]), conv_res)
    conv_res = conv_res / (np.power(2, 2 * k))
    g_pyr,g_con_res = build_gaussian_pyramid(im,max_levels,filter_size)
    pyr= [g_pyr[len(g_pyr)-1]]
    for i in range(len(g_pyr)-1,0,-1):
        pyr.insert(0,g_pyr[i-1] - expand(g_pyr[i],conv_res))

    return pyr , conv_res








def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """
    reconstructed_image = lpyr[-1]*coeff[-1]
    for i in range(len(lpyr)-1,0,-1):
            reconstructed_image = expand(reconstructed_image,filter_vec) +\
                                  (lpyr[i-1]*coeff[i-1])
    return reconstructed_image












def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    for j in range(levels):
        pyr[j] =np.subtract(pyr[j], np.min(pyr[j]))/np.subtract(np.max(pyr[j]),
                                                                np.min(pyr[j]))
    res = pyr[0]
    for i in range(1,levels):
        arr = np.zeros(((pyr[0].shape[0]- pyr[i].shape[0]),pyr[i].shape[1]))
        new_arr = np.concatenate((pyr[i],arr),axis=0)
        res = np.concatenate((res,new_arr),axis=1)
    return res








def display_pyramid(pyr, levels):
    """

    display the rendered pyramid
    """
    plt.figure()
    plt.imshow(render_pyramid(pyr,levels), cmap='gray')
    plt.show()



def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    l_1,filter_vec_1 = build_laplacian_pyramid(im1,max_levels,filter_size_im)
    l_2,filter_vec_2 = build_laplacian_pyramid(im2,max_levels,filter_size_im)
    mask = mask.astype(np.float64)
    g_m, filter_vec_g= build_gaussian_pyramid(mask,max_levels,filter_size_mask)

    coeff = [1] * len(l_1)
    l_out = []
    for k in range(len(l_1)):
        l_out.append(np.add(np.multiply(g_m[k],l_1[k]),
                            np.multiply(np.subtract(1,g_m[k]),l_2[k])))
    return np.clip( laplacian_to_image(l_out,filter_vec_1,coeff),0,1)


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('externals/cat_1.jpg'), 2)
    im2 = read_image(relpath('externals/leo.jpg'), 2)
    mask = read_image(relpath('externals/mask_1.jpg'), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool_)
    im_size = im1.shape
    blended_im = np.empty(im_size)
    channels = np.arange(3)
    for i in channels:
        blended_im[:, :, i] = pyramid_blending(im1[:,:,i],im2[:,:,i],mask,7,7,7)
    plt.subplot(2,2,1)
    plt.imshow(im1)
    plt.subplot(2,2,2)
    plt.imshow(im2)
    plt.subplot(2,2,3)
    plt.imshow(mask,cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(blended_im)
    plt.show()

    return im1,im2, mask,blended_im




def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath('externals/flower.jpg'), 2)
    im2 = read_image(relpath('externals/dress.jpg'), 2)
    mask = read_image(relpath('externals/mask_2.jpg'), 1)
    mask = np.round(mask)
    mask = mask.astype(np.bool_)
    im_size = im1.shape
    blended_im = np.empty(im_size)
    channels = np.arange(3)
    for i in channels:
        blended_im[:, :, i] = pyramid_blending(im1[:,:,i],im2[:,:,i],mask,5,5,5)
    plt.subplot(2,2,1)
    plt.imshow(im1)
    plt.subplot(2,2,2)
    plt.imshow(im2)
    plt.subplot(2,2,3)
    plt.imshow(mask,cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(blended_im)
    plt.show()
    return im1, im2, mask, blended_im


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    matrix_picture = imageio.imread(filename).astype('float64')
    if matrix_picture.ndim == RGB and representation==1:
        return np.divide(matrix_picture, 255)

    if representation == RGB:
        return np.divide(matrix_picture, 255)

    return np.divide(rgb2gray(matrix_picture), 255)


