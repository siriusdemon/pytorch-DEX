# DEX: Deep EXpectation of apparent age from a single image

This is a pytorch version of DEX. Refer to its [Home Page](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) for more details


## Getting Started

A separate Python environment is recommended.
+ Python3.5+ (Python3.5, Python3.6 are tested)
+ Pytorch == 1.0
+ opencv4 (opencv3.4.5 is tested also)
+ numpy

install dependences using `pip`
```bash
pip3 install numpy opencv-python
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision (optional)
```
or install using `conda`
```bash
conda install opencv numpy
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## Usage
```bash
git clone https://github.com/siriusdemon/pytorch-DEX.git
cd pytorch-DEX
python demo.py path/to/image 
```

## Citation
    @InProceedings{Rothe-ICCVW-2015,
      author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
      title = {DEX: Deep EXpectation of apparent age from a single image},
      booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
      year = {2015},
      month = {December},
    }