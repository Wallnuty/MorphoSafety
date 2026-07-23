echo "1) Preparing environment" 
module purge
module load stack/2024-06
module load gcc/12.2.0

module load cmake/3.27.7

# Madrona dependencies 
module load libx11/1.8.4-ns5x2da
module load libxrandr/1.5.3-acspwjp
module load libxinerama/1.1.3
module load libxcursor/1.2.1
module load libxi/1.7.6-qeazdpn
module load mesa/23.0.3

# Hidden Madrona dependencies :^) 
module load libxrender/0.9.10-kss2t7k
module load libxext/1.3.3-e74gj2z
module load libxfixes/5.0.2-5fbeidb

# Cuda + Python 
module load python_cuda/3.11.6

echo "2) Installing mujoco in Python environment" 
pip3 install -U mujoco mujoco-mjx

echo "3) Installing Madrona" 
git clone https://github.com/shacklettbp/madrona_mjx.git
cd madrona_mjx
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd ..
pip3 install -e .

echo "4) Installation Done - check by running 'from madrona_mjx import BatchRenderer' in a python shell"