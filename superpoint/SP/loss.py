import torch 
import torch.nn.functional as F

def warp_points(points, homographies, device='cpu'):
    """
    :param points: (N,2), tensor
    :param homographies: [B, 3, 3], batch of homographies
    :return: warped points B,N,2
    """
    if len(points)==0:
        return points

    #TODO: Part1, the following code maybe not appropriate for your code
    points = torch.fliplr(points)
    if len(homographies.shape)==2:
        homographies = homographies.unsqueeze(0)
    B = homographies.shape[0]
    ##TODO: uncomment the following line to get same result as tf version
    # homographies = torch.linalg.inv(homographies)
    points = torch.cat((points, torch.ones((points.shape[0], 1),device=device)),dim=1)
    ##each row dot each column of points.transpose
    warped_points = torch.tensordot(homographies, points.transpose(1,0),dims=([2], [0]))#batch dot
    ##
    warped_points = warped_points.reshape([B, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    #TODO: Part2, the flip operation is combinated with Part1
    warped_points = torch.flip(warped_points,dims=(2,))
    #TODO: Note: one point case
    warped_points = warped_points.squeeze(dim=0)
    return warped_points

def pixel_shuffle_inv(tensor, scale_factor):
    """
    Implementation of inverted pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to down-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, (r*r)*C, H/r, W/r],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert height % scale_factor == 0
    assert width % scale_factor == 0

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute(0, 1, 3, 5, 2, 4)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor

def descriptor_loss(descriptors, warped_descriptors, homographies, valid_mask=None, device='cpu', name= ''):
    """
    :param descriptors: [B,C,H/8,W/8]
    :param warped_descriptors: [B,C.H/8,W/8]
    :param homographies: [B,3,3]
    :param config:
    :param valid_mask:[B,H,W]
    :param device:
    :return:
    """
    grid_size = 8
    positive_margin = 1.0
    negative_margin = 0.2 
    lambda_d = 650
    lambda_loss = 1

    (batch_size, _, Hc, Wc) = descriptors.shape
    coord_cells = torch.stack(torch.meshgrid([torch.arange(Hc,device=device),
                                              torch.arange(Wc,device=device)]),dim=-1)#->[Hc,Wc,2]
    coord_cells = coord_cells * grid_size + grid_size // 2  # (Hc, Wc, 2)
    warped_coord_cells = warp_points(coord_cells.reshape(-1, 2), homographies, device=device)
    coord_cells = torch.reshape(coord_cells, [1,1,1,Hc,Wc,2]).type(torch.float32)
    warped_coord_cells = torch.reshape(warped_coord_cells, [batch_size, Hc, Wc, 1, 1, 2])
    cell_distances = torch.norm(coord_cells - warped_coord_cells, dim=-1, p=2)
    s = (cell_distances<=(grid_size-0.5)).float()#
    descriptors = torch.reshape(descriptors, [batch_size, -1, Hc, Wc, 1, 1])
    descriptors = F.normalize(descriptors, p=2, dim=1)
    warped_descriptors = torch.reshape(warped_descriptors, [batch_size, -1, 1, 1, Hc, Wc])
    warped_descriptors = F.normalize(warped_descriptors, p=2, dim=1)
    dot_product_desc = torch.sum(descriptors * warped_descriptors, dim=1)
    dot_product_desc = F.relu(dot_product_desc)
    ##l2_normalization
    dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
                                                 p=2,
                                                 dim=3), [batch_size, Hc, Wc, Hc, Wc])
    dot_product_desc = torch.reshape(F.normalize(torch.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
                                                 p=2,
                                                 dim=1), [batch_size, Hc, Wc, Hc, Wc])

    positive_dist = torch.maximum(torch.tensor(0.,device=device), positive_margin - dot_product_desc)
    negative_dist = torch.maximum(torch.tensor(0.,device=device), dot_product_desc - negative_margin)
    loss = lambda_d * s * positive_dist + (1 - s) * negative_dist
    # Mask the pixels if bordering artifacts appear
    valid_mask = torch.ones([batch_size, Hc*grid_size, Wc*grid_size],
                             dtype=torch.float32, device=device) if valid_mask is None else valid_mask
    valid_mask = valid_mask.unsqueeze(dim=1).type(torch.float32)  # [B, H, W]->[B,1,H,W]
    valid_mask = pixel_shuffle_inv(valid_mask, grid_size)# ->[B,64,Hc,Wc]
    valid_mask = torch.prod(valid_mask, dim=1)
    valid_mask = torch.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

    normalization = torch.sum(valid_mask)*(Hc*Wc)
    loss = lambda_loss*torch.sum(valid_mask * loss)/normalization

    return loss
