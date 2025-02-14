conda create -n GenVDM2 python=3.10 -y
conda activate GenVDM2
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install huggingface-hub==0.23.0
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch