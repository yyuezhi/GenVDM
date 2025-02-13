conda create -n VDM_prepare python=3.9 -y
pip install open3d trimesh[all] scikit-learn shapely scipy pillow pymeshlab plyfile omegaconf libigl matplotlib cython opencv-python blenderproc==2.7.1
python setup.py build_ext --inplace