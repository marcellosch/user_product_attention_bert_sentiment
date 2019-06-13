#! /usr/bin/python3

### Imports 
from utils.data import SentimentDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

### config variables
out_file = "figures/label_distributions.png"
path_template = "data/{}"
train = 'train.txt'
wordlist = "wordlist.txt"
productlist = "prdlist.txt"
userlist = "usrlist.txt"

datasets = ["IMDB", "yelp13", "yelp14"]

### Load each datset and add a plot for its label distribution
fig, ax = plt.subplots(1,len(datasets), figsize=(12,3))
for i, s in enumerate(datasets):
    path = Path(path_template.format(s))
    dat = SentimentDataset(str(path/train),
                       str(path/userlist),
                       str(path/productlist),
                       str(path/wordlist))
    x = np.array(dat.documents["label"])
    ax[i].hist(x, density=True, align='left', bins=range(max(x)+2), color=f"C{i}") 
    ax[i].set_xticks(range(max(x)+1))
    ax[i].set_facecolor((1.0, 1.00, 1.00))
    ax[i].set_title(s)
    
### Save image to folder
fig.savefig(out_file, dpi=200, pad_inches=0, bbox_inches='tight')
