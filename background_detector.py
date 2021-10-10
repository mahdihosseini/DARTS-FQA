import numpy as np

def background_detector(image_patch, config):
    '''
    :param image_patch: numpy array with RGB pixel values for the image patch.
    :return: do_process: boolean equal to true if not part of background, and false if part of background.
    '''

    block_size = np.shape(image_patch)
    block_size = block_size[:-1]

    threshold = [170, 170, 170]
    r_mask = image_patch[:,:,0] > threshold[0]
    g_mask = image_patch[:,:,1] > threshold[1]
    b_mask = image_patch[:,:,2] > threshold[2]
    r_mask = r_mask.astype(int)
    g_mask = g_mask.astype(int)
    b_mask = b_mask.astype(int)
    rgb_mask = r_mask & g_mask & b_mask
    background_ratio = np.sum(rgb_mask[:])/np.prod(block_size)

    diff_channel = 0
    diff_channel = diff_channel + abs(image_patch[:,:,0] - image_patch[:,:,1])
    diff_channel = diff_channel + abs(image_patch[:,:,0] - image_patch[:,:,2])
    diff_channel = diff_channel + abs(image_patch[:,:,1] - image_patch[:,:,2])
    diff_channel = diff_channel/3
    diff_channel = np.mean(diff_channel[:])

    min_channel_difference = 10
    background_threshold = 0.3
    if diff_channel > min_channel_difference:
        do_process = True
    else:
        if background_ratio > background_threshold:
            do_process = False
        else:
            do_process = True

    return do_process
