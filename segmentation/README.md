# U-net Segmentation

The code in this directory is adapted from [this repo](https://github.com/switchablenorms/CelebAMask-HQ).

We pre-train the U-net encoder using PDR (Physically Disentangled Representations) and then use it for segmentation, as compared to the supervised U-net baseline.

To train:

```
python -u main.py --batch_size 8 --imsize 64 --version exptName
```

To test:

```
python -u main.py --batch_size 4 --imsize 64 --version exptName --train False
```

## Face parsing
A Pytorch implementation face parsing model trained by CelebAMask-HQ
## Dependencies
* Pytorch 0.4.1
* numpy
* Python3
* Pillow
* opencv-python
* tenseorboardX
## Preprocessing
* Move the mask folder, the image folder, and `CelebA-HQ-to-CelebA-mapping.txt` ( remove 1st line in advance ) under `./Data_preprocessing`
* Run `python g_mask.py`
* Run  `python g_partition.py` to split train set and test set.
## Training
* Run `bash run.sh #GPU_num`
## Testing & Color visualization
* Run `bash run_test.sh #GPU_num`
* Results will be saved in `./test_results`
* Color visualized results will be saved in `./test_color_visualize`
* Another way for color visualization without using GPU: Run `python ./Data_preprocessing/g_color.py` 
