# Iterative Polygon Deformation for Building Extraction

## Installation

```
conda create -n ipda python=3.8
conda activate ipda

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 11.8, install torch 2.0 built from cuda 11.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install Cython==3.0.5
pip install -r requirements.txt

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# Compile cuda extensions under 'lib/csrc'
cd lib/csrc
cd poly_utils
python setup.py build_ext --inplace
cd ../dcn_v2
python setup.py build_ext --inplace
cd ../extreme_utils
python setup.py build_ext --inplace
cd ../roi_align_layer
python setup.py build_ext --inplace
```

## Prepare datasets

1. Organize the dataset as the following structure:
    ```
   dataset
    ├── train/test
    │   ├── Images
    │   │   ├── file1.png
    │   │   ├── file2.png
    │   │   ├── ...
    │   ├── anns
    │   │   ├── file1.txt
    │   │   ├── file2.txt
    │   │   ├── ...
    ```
   Each line in the f.txt file corresponds to the outline annotation of a building instance in the image f.png,
   and the organization of each line is in the format "x y x y ...". 
   Here, x and y represent the horizontal and vertical coordinates of the instance outline points, respectively.
	
2. Add the dataset information to "lib/datasets/dataset_catalog.py".

## Training

```
python train_net.py --cfg_file configs/vegas.yaml model vegas
```

## Inference
1. visualize:
    ```
    python test_net.py --type visualize --cfg_file configs/vegas.yaml model vegas
    ```
3. evaluate (APs and PolySim):
    ```
    python test_net.py --type evaluate --cfg_file configs/vegas.yaml model vegas
    ```
4. Speed:
    ```
    python test_net.py --type speed --cfg_file configs/vegas.yaml model vegas
    ```