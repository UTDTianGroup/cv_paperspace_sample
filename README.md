# cv_paperspace_sample
Sample code with instructions to run on paperspace gradient

This code has been tested on paperspace gradient with python==3.8, cuda==11.7, and pytorch==1.13.1

Please create a notebook on [paperspace gradient](https://www.paperspace.com) and start the machine.

The following commands can be run on a [terminal in a gradient notebook](https://docs.digitalocean.com/products/paperspace/notebooks/how-to/use-terminal/).
## Environment

Miniconda installation commands
```
mkdir -p /notebooks/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /notebooks/miniconda3/miniconda.sh
bash /notebooks/miniconda3/miniconda.sh -b -u -p /notebooks/miniconda3
rm /notebooks/miniconda3/miniconda.sh
```

To deactivate auto activate of the base conda environment, please run the following command
```
conda config --set auto_activate_base false
```

Environment installation commands
```
conda create -n cv_paperspace python=3.8
conda activate cv_paperspace
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

To use an existing installation of miniconda3, please run the command
```
source /notebooks/miniconda3/etc/profile.d/conda.sh
```

## Train the model

python train.py --save_path /path/to/save/model/

## Test the model
python train.py --evaluation --eval_model_path /path/to/saved/model/