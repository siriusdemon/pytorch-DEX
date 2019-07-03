# DEX: Deep EXpectation of apparent age from a single image

This is a pytorch version of DEX. Refer to its [Home Page](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) for more details

You can refer to [insight](https://github.com/siriusdemon/hackaway/tree/master/projects/insight) if you want a much smaller model but it uses `mxnet` instead of `pytorch`. I haven't convert it to `pytorch` yet.

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

## Results
<img src="imgs/2.png">

```
predict image: imgs/2.png
woman: 0.994, man: 0.006
age: 21.433
```
<img src="imgs/5.png">

```bash
predict image: imgs/5.png
woman: 0.010, man: 0.990
age: 42.896
```

## Installation
You can use dex as a separate Python package right now!
```
cd pytorch-DEX
pip install .
```
See [demo.py](demo.py) for example.

## Citation
    @InProceedings{Rothe-ICCVW-2015,
      author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
      title = {DEX: Deep EXpectation of apparent age from a single image},
      booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
      year = {2015},
      month = {December},
    }
