# Dataset for GenVDM

## Ready to use dataset
Please visit [huggingface](https://huggingface.co/datasets/yzyang/VDM_Dataset) to download our dataset, which contains approximately 1200 VDM shapes.

## Create your own VDM data
To install preprocessing environment:
```
bash env_set.sh
```

Please put your own data in ./mesh directory. You need to first voxelize your data before launching our labeling GUI. You need a monitor to perform interactive labeling.
```
python voxelize.py
python launch_UI.py
```

The basic labeling routine goes like this:
```
1. In label mode, left click to mark keypoints in shapes
2. Mark keypoints around part which you wish to extract and form a circle using ~10 keypoints. These keypoints have to be strictly clockwise/anti-clockwise
3. Press F to find loop that connects these keypoints. You need to make sure the loop is formed to cut intended part and the rest of geometry
4. Press A/D to switch label to extract other parts within the same shape, Press S to move on to next shape
```


Some basic operations of GUI:
```
"W"/"S" for prev/next shape, "s" also automatically save label to ./label
"Z" for switch to view mode, you can use left click to pan the view/move the view up/down
"X" for switch to label mode, left click to label points
"A/D" to swtich label ID, there is 0(green), 1(blue), 2(red) ..., upto 10 labels in total.
"F" to find loop given the label keypoints, it would only find loop for current label ID. Do not press twice for each label ID
"C" clear all the label points and loop for current label ID
```

To generate VDM mesh from labels:
```
python patch_gen.py
```
Note that due to complex geometry of some input shapes, this algorithm may fail/produce inferior result. It would be better to check generated VDM mesh and remove those with poor quality.

To render these VDM meshes:
```
blenderproc run ./renderNormal_training.py 0 1  
```
You can also launch multi-process rendering:
```
python multi_rendering.py
```

