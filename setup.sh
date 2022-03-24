conda update -y conda
conda install -y pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 ogb==1.3.0 git cmake make -c pytorch -c conda-forge
git clone --recurse-submodules https://github.com/chwan-rice/dgl.git
cd dgl
git submodule update --init --recursive
mkdir build
cd build
cmake -DUSE_CUDA=ON -DBUILD_TORCH=ON ..
make -j4
cd ../python
python setup.py install
cd ../../
rm -rf dgl/
