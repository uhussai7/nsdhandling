import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nsdhandling.utils.nsd_orig.mask import inflate_masks
from nsdhandling import config

for subj in range(1,9):
    print('Inflating mask for subject...: ',subj)
    inflate_masks(subj,config.MASK_ROOT)
print('Done')