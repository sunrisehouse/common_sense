import torch
from option import get_args
import random
import numpy as np

from train import train
from test import test
from subjective import subjective

if __name__ == '__main__':
    args = get_args()

    #### Random Seed ####
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.mission == "train":
        train(args)

    elif args.mission == 'test':
        test(args)

    elif args.mission == 'subjective':
        subjective(args)