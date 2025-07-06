# GenVDM
Pytorch implementation of [[CVPR 2025] GenVDM:Generating Vector Displacement Maps From a Single Image](www.google.com)  [Yuezhi Yang](https://yyuezhi.github.io/), [Qimin Chen](https://qiminchen.github.io/), [Vladimir G. Kim](http://www.vovakim.com/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/), [Qixing Huang](https://www.cs.utexas.edu/~huangqx/), [Zhiqin Chen](https://czq142857.github.io/)

<img src="asset/Teaser.png" style="width:400px;" />

### [Paper](https://www.arxiv.org/abs/2503.00605)  |  [Project page](https://yyuezhi.github.io/GenVDM/)

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{yang2025genvdm,
  title={GenVDM: Generating Vector Displacement Maps From a Single Image},
  author={Yang, Yuezhi and Chen, Qimin and Kim, Vladimir G and Chaudhuri, Siddhartha and Huang, Qixing and Chen, Zhiqin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
}
```


## Environment
Please install the environment by running:
```
conda create -n GenVDM python=3.10 -y
conda activate GenVDM
bash install_env.sh
```

## Pre-trained weights and Datasets
We provide the pre-trained network weights. Please put it in ./checkpoints/example_run directory.

- [GenVDM checkpoint](https://huggingface.co/datasets/yzyang/VDM_Dataset)

Please see dataset directory for instruction on how to download dataset and how to make your own data

## Inference
To generate VDM image, put your image in ./input and run:
```
bash generate.sh <image name> <exp name> <checkpoint name>
```

For example:
```
bash generate.sh ear2.png example_run example
```
Notice that your image has to be in png format RGBA image. The object needs to lie in the center of the image. See example images in ./input as an example

## Training
To train the network, please put rendered images in ./dataset/rendering_result
```
python train.py --base config/example_run.yaml --gpus 0,1,2,3,4,5,6,7 --num_nodes 1
```

## Interactive Modeling
You can download demo.blend and precomputed result from ./demo or directly load *.exr file from outputVDM directory to play around VDM. An example VDM has been loaded for you. We highly recomend to use blender 3.6 to open demo.blend since higher version might cause loading errors.

You can learn VDM related blender instruction from [here](https://docs.blender.org/manual/en/3.5/sculpt_paint/sculpting/tools/draw.html#vdm-displacement) and [here](https://www.blender.org/download/releases/3-5/).

## Acknowledgement
We have borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [Wonder3D](https://github.com/xxlong0/Wonder3D)
- [InstantMesh](https://github.com/TencentARC/InstantMesh)