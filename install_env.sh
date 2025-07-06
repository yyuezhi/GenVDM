pip install -r requirements.txt
conda install -c nvidia cuda-toolkit=12.1 -y
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install huggingface-hub==0.23.0
pip install git+https://github.com/NVlabs/tiny-cuda-nn@c91138b#subdirectory=bindings/torch