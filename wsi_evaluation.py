import openslide
import math
import numpy as np
import torch
import torchvision
import time
from alive_progress import alive_bar

import skimage.color
from background_detector import background_detector



def wsi_evaluation(svs_image_path, stride, patch_size, model, use_cuda, config):
    '''
    :param svs_image_path: string corresponding to the .svs whole slide image in question.
    :return: svs image in an acceptable format.
    '''


    block_size = [patch_size, patch_size]
    wsi = openslide.open_slide(svs_image_path)
    grid_step = stride
    image_width = wsi.dimensions[0]
    image_height = wsi.dimensions[1]
    # image_shape = (image_height, image_width)

    image_scores = [[]]

    i_lower, i_upper, j_lower, j_upper = config.i_lower, config.i_upper, config.j_lower, config.j_upper

    image_shape = (i_upper - i_lower, j_upper - j_lower)

    i_index = np.arange(i_lower, i_upper, grid_step)
    j_index = np.arange(j_lower, j_upper, grid_step)

    # Mapping the .SVS image as a 2D coordinate system of start points to sample from.
    J, I = np.meshgrid(j_index, i_index)
    j_mesh = np.arange(0, np.size(j_index), 1)
    i_mesh = np.arange(0, np.size(i_index), 1)
    J_mesh_index, I_mesh_index = np.meshgrid(j_mesh, i_mesh)
    N1_index, N2_index = np.shape(J)

    I_linear = I.reshape(-1, order="F")
    J_linear = J.reshape(-1, order="F")
    I_mesh_index_linear = I_mesh_index.reshape(-1, order="F")
    J_mesh_index_linear = J_mesh_index.reshape(-1, order="F")

    i_shift = grid_step
    j_shift = grid_step

    iter_pos = 0
    with alive_bar(np.size(J_mesh_index_linear[:]), theme='smooth', spinner='waves2') as bar:
        for iter in range(0, np.size(J_mesh_index_linear[:])):
            bar()
            iter_pos = iter_pos + 1
            pos = [I_linear[iter], J_linear[iter]]
            i_start = pos[0] - math.floor((block_size[0] - 1) / 2)
            i_end = pos[0] + math.ceil((block_size[0] - 1) / 2)
            j_start = pos[1] - math.floor((block_size[1] - 1) / 2)
            j_end = pos[1] + math.ceil((block_size[0] - 1) / 2)

            RGBA = np.array(wsi.read_region((j_start - 1, i_start - 1), 0, (j_end - j_start + 1, i_end - i_start + 1)))
            image_patch = RGBA[:, :, :-1]

            do_process = background_detector(image_patch, config)

            j_index = J_mesh_index_linear[iter]
            i_index = I_mesh_index_linear[iter]

            if do_process:
                to_tensor = torchvision.transforms.ToTensor()
                tensor_patch = to_tensor(image_patch)
                tensor_patch = torch.autograd.Variable(tensor_patch)
                if use_cuda:
                    tensor_patch = tensor_patch.cuda()
                score_predict = torch.squeeze(model(tensor_patch[None, :, :, :])[0].cpu().data).numpy()
                image_scores.append([i_index, j_index, score_predict.item()])

    del(image_scores[0])
    return np.array(image_scores), image_shape