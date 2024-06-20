# dataset cifar10
# Download:
# cifar10_raw
wget -c https://figshare.com/ndownloader/files/29138334?private_link=0c1dfc3be66eb622cf85
# cifar-10-python
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# xtract:
python -m zipfile -e cifar10-raw-images.zip ./data
python -m zipfile -e cifar-10-python.tar.gz ./data
# or
#!unzip ./data/cifar10-raw-images.zip
#!unzip ./data/cifar-10-python.tar.gz
