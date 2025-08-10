import torch 
import numpy as np 
from torch.utils.data import Dataset
import torchvision.transforms.functional as trsF 
import torch.nn.functional as F 
import bz2 
import pickle
import os 
import glob 
from sklearn.cluster import DBSCAN
import random 
from datasets.constants import(
    ChatGLM_DistanceMatrix,
    ChatGLM_ConfidenceMatrix,
    DeepSeek_DistanceMatrix,
    DeepSeek_ConfidenceMatrix,
    ChatGPT_DistanceMatrix,
    ChatGPT_ConfidenceMatrix,
)
class SemanticMapDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()

        if args.dataset == 'joint':
            self.data_paths = glob.glob(
                os.path.join(args.data_root, '*', split, '*', '*.pbz2')
            )
        else:
            self.data_paths = glob.glob(
                os.path.join(args.data_root, args.dataset, split, '*', '*.pbz2')
            )

        self.args = args
    
    def __getitem__(self, index):
        with bz2.BZ2File(self.data_paths[index], 'rb') as fp:
            data = pickle.load(fp)
        if self.args.prior:
            if self.args.llm == 'chatglm':
                DistanceMatrix = ChatGLM_DistanceMatrix
                ConfidenceMatrix = ChatGLM_ConfidenceMatrix
            elif self.args.llm == 'deepseek':
                DistanceMatrix = DeepSeek_DistanceMatrix
                ConfidenceMatrix = DeepSeek_ConfidenceMatrix
            elif self.args.llm == 'chatgpt':
                DistanceMatrix = ChatGPT_DistanceMatrix
                ConfidenceMatrix = ChatGPT_ConfidenceMatrix
            else:
                raise ValueError(f"No such llm available {self.args.llm}")
            
            
            prior = get_priors(data['in_semmap'], DistanceMatrix, ConfidenceMatrix, k = self.args.k,
                               distance_thre = self.args.dist_thr, confidence_thre = self.args.conf_thr,
                               min_std = self.args.min_std, max_std = self.args.max_std)
            data['prior'] = prior 
            in_semmap, semmap, prior = Crop(data, self.args.expand_ratio, self.args.input_size)
            return in_semmap, semmap, prior 
        in_semmap, semmap = Crop(data, self.args.expand_ratio, self.args.input_size)
        return in_semmap, semmap

    def __len__(self):
        return len(self.data_paths)


def Crop(data, epsilon = 1.4, target_size = 256):
    """
    Cropping semantic map into minimum bounding box and rescale it based
    on scale factor epsilon and target_size
    """
    in_semmap = torch.from_numpy(data['in_semmap'])
    semmap = torch.from_numpy(data['semmap']) # gt
    nz_c = torch.nonzero(torch.sum(in_semmap, dim=0))
    x_min,x_max,y_min,y_max = nz_c[:,0].min(), nz_c[:,0].max(), nz_c[:,1].min(), nz_c[:,1].max()
    wh = max(x_max-x_min,y_max-y_min)

    # Extend the cropped area to be larger than minimum bounding box
    x_min = int(max(0, x_min - (wh * epsilon - wh) // 2)) 
    y_min = int(max(0, y_min - (wh * epsilon - wh) // 2))
    x_max = int(min(x_min + wh * epsilon, 479))
    y_max = int(min(y_min + wh * epsilon, 479))

    wh = max(x_max-x_min, y_max-y_min)


    # Resize
    input_map = trsF.resize(trsF.crop(in_semmap, x_min, y_min, wh, wh), target_size, interpolation=trsF.InterpolationMode.NEAREST)
    gt_map = trsF.resize(trsF.crop(semmap, x_min, y_min, wh, wh), target_size, interpolation=trsF.InterpolationMode.NEAREST)
    
    if 'prior' in data:
        prior = trsF.resize(trsF.crop(data['prior'], x_min, y_min, wh, wh), target_size, interpolation=trsF.InterpolationMode.NEAREST)
        return input_map, gt_map, prior
    return input_map, gt_map

def calculate_frontiers(x):
    # x - semantic map of shape (N, H, W)
    free_map = (x[0] >= 0.5).float()  # (H, W)
    exp_map = torch.max(x, dim=0).values >= 0.5  # (H, W)
    unk_map = (~exp_map).float()  # (H, W)

    # Compute frontiers
    unk_map_shiftup = F.pad(unk_map, (0, 0, 0, 1))[1:, :]
    unk_map_shiftdown = F.pad(unk_map, (0, 0, 1, 0))[:-1, :]
    unk_map_shiftleft = F.pad(unk_map, (0, 1, 0, 0))[:, 1:]
    unk_map_shiftright = F.pad(unk_map, (1, 0, 0, 0))[:, :-1]

    frontiers = (
        (free_map == unk_map_shiftup)
        | (free_map == unk_map_shiftdown)
        | (free_map == unk_map_shiftleft)
        | (free_map == unk_map_shiftright)
    ) & (free_map == 1)  # (H, W)

    # Dilate the frontiers
    frontiers = frontiers.unsqueeze(0).float()  # (1, H, W) for pooling
    frontiers = F.max_pool2d(frontiers.unsqueeze(0), 7, stride=1, padding=3).squeeze(0)

    return frontiers  # (H, W)

def extract_objects(semmap, merge_dist = 5):
    """
    A single object often occupy connected grids rather than a single grid. This 
    is for extracting the centroids of each objects in a semantic map.
    """

    semmap = semmap[2:, :, :] # exclude wall, floor 
    nc, _, _  = semmap.shape

    objects_map = torch.zeros_like(semmap)

    for c in range(nc):
        if (semmap[c] > 0).any().item():
            coords = np.column_stack(np.nonzero(semmap[c])).T
            assert coords.shape[0] > 0
            if coords.shape[0] == 1:
                centers = coords 
            else:
                clustering = DBSCAN(eps = merge_dist, min_samples = 1).fit(coords)

                labels = clustering.labels_ 

                unique_labels = np.unique(labels)


                centers = np.array([coords[labels == k][0] for k in unique_labels])
            

            for (y, x) in centers:
                objects_map[c, y, x] = 1  # Mark center
    
    return objects_map

def get_priors(semmap, DistanceMatrix, ConfidenceMatrix, distance_thre, confidence_thre, k = 3, min_std = 5., 
               max_std = 10.):
    
    semmap = torch.from_numpy(semmap)
    objects_map = extract_objects(semmap)
    nc, h, w = objects_map.shape

    frontiers = calculate_frontiers(semmap).expand(nc, h, w)

    prior = torch.zeros_like(objects_map).float()
    
    registered_objects = []
    for c in range(nc):
        # Object corresponding to current channel isn't present or no frontiers 
        if not objects_map[c].any().item() or not frontiers[c].any().item():
            continue 

        object_coords = torch.nonzero(objects_map[c], as_tuple = False)
        frontier_coords = torch.nonzero(frontiers[c], as_tuple = False)

        distances = torch.cdist(object_coords.float(), frontier_coords.float())

        min_indices = torch.argmin(distances, dim = 1)

        closest_frontiers = frontier_coords[min_indices]

        directions = (closest_frontiers - object_coords).float()

        norm = torch.norm(directions, dim = 1, keepdim = True)
        norm = torch.clamp(norm, min = 1e-6)

        unit = directions / norm

        distance_valid = np.array(DistanceMatrix[c]) < distance_thre
        confidence_valid = np.array(ConfidenceMatrix[c]) > confidence_thre
        indices = np.argwhere(distance_valid | confidence_valid).tolist()
        indices = [x[0] for x in indices if x[0]!=c]

        # For objects like 'chair' and 'table', ChatGLM tends to generate relatively small distance from all
        # other objects, pick the top k 
        if len(indices) >= 8 or len(indices) == 0 :
            _, indices = torch.topk(-torch.tensor(DistanceMatrix[c]), k + 1)
            indices = indices[1:]
        # Prioritize the objects already registered to take accumulate object co-occurence probs
        common = list(set(indices) & set(registered_objects))
        if len(common) >= 1:
            prior_no = common[random.randint(0, len(common) - 1)]
        else:
            prior_no = indices[random.randint(0, len(indices) - 1)]
        
        registered_objects.append(prior_no)

        i, d, c = prior_no, DistanceMatrix[c][prior_no], ConfidenceMatrix[c][prior_no]
        for j, object in enumerate(object_coords):
            # for i, d, c in zip(top_k_i, top_k_d, top_k_c):
            centroid = object + unit[j] * d 
            y, x  = centroid 
            y = min(h - 1, max(0, int(y)))
            x = min(w - 1, max(0, int(x)))
            i = int(i)
            std = min_std  + (max_std - min_std) * (1 - c)
            gauss_2d = isotropic_gaussian_2d((h, w), (y, x), std)
            prior[i - 2] += gauss_2d 
    
    return prior
            
def isotropic_gaussian_2d(shape, mean, std):
    h, w = shape
    cy, cx = mean

    y = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
    x = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)

    d2 = (x - cx) ** 2 + (y - cy) ** 2

    gauss = torch.exp(-d2 / (2 * std ** 2)) / (2 * torch.pi * std ** 2)

    return gauss