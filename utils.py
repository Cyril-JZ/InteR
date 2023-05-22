import sys
import torch
import random
from datetime import datetime
import numpy as np
import csv


csv.field_size_limit(sys.maxsize)

def setup_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark  = False
    else:   # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark  = True


def format_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S | ')


def load_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def remove_multiple_strings(cur_string, replace_list):
    for cur_word in replace_list:
        cur_string = cur_string.replace(cur_word, '')
    return cur_string
