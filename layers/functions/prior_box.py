import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.img_wh = cfg['img_wh']
        self.num_priors = len(cfg['aspect_ratios'])
        self.feature_maps = cfg["feature_maps"]
        self.use_extra_prior = cfg['use_extra_prior']
        self.variance = cfg['variance'] or [0.1]
        self.min_sizes = cfg['min_sizes']
        self.use_max_sizes = cfg["use_max_sizes"]
        if cfg["use_max_sizes"]:
            self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        if self.use_extra_prior:
            self.aspect_ratios = cfg['aspect_ratios_extra']
        if 'refine' in cfg and cfg['refine'] == True:
            self.feature_maps = self.feature_maps[:6]
            self.aspect_ratios = self.aspect_ratios[:6]
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            grid_h, grid_w = f[1], f[0]
            for i in range(grid_h):
                for j in range(grid_w):
                    f_k_h = self.img_wh[1] / self.steps[k][1]
                    f_k_w = self.img_wh[0] / self.steps[k][0]
                    # unit center x,y
                    cx = (j + 0.5) / f_k_w
                    cy = (i + 0.5) / f_k_h

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k_h = self.min_sizes[k] / self.img_wh[1]
                    s_k_w = self.min_sizes[k] / self.img_wh[0]
                    mean += [cx, cy, s_k_w, s_k_h]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    if self.use_max_sizes:
                        s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k] /self.img_wh[0]))
                        s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k] /self.img_wh[1]))
                        mean += [cx, cy, s_k_prime_w, s_k_prime_h]

                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k_w*sqrt(ar), s_k_h/sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        # print(output.size())
        return output
