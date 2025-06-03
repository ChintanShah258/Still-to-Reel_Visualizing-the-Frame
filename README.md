# Still-to-Reel_Visualizing-the-Frame
We improve upon the CVPR paper 3D Cinemagraphy from a Single Image (CVPR 2023)

[Project](https://xingyi-li.github.io/3d-cinemagraphy/) | [Paper](https://github.com/xingyi-li/3d-cinemagraphy/blob/main/pdf/3d-cinemagraphy-paper.pdf) | [arXiv](https://arxiv.org/abs/2303.05724) | [Video](https://youtu.be/sqCy7ffTEEY) | [Supp](https://github.com/xingyi-li/3d-cinemagraphy/blob/main/pdf/3d-cinemagraphy-supp.pdf) | [Poster](https://github.com/xingyi-li/3d-cinemagraphy/blob/main/pdf/3d-cinemagraphy-poster.pdf)

We introduce a novel technique for 3D cinemagraph generation that synthesizes realistic animations by seamlessly combining visual content animation with camera motion. Starting from a single still image, our method addresses key challenges such as depth inaccuracies, motion field errors, and artifacts that arise when naively merging existing 2D animation and 3D photography techniques. Leveraging advanced
depth estimation and inpainting, our approach converts input images into layered depth representations and adapts them into robust 3D motion fields using a hybrid Runge-Kutta Eulerian flow. To address issues like depth discontinuities and the emergence of visual artifacts, we implement customized smoothing, depth clipping, and bilateral filtering to ensure consistent and natural results.
By bidirectionally displacing features in the 3D scene and synthesizing novel views through projection and blending, our method achieves visually coherent and artifact-free animations. Quantitative evaluations and user studies demonstrate the effectiveness and robustness
of our approach, offering a significant step forward in 3D cinemagraphy.

## Installation
```
git clone https://github.com/ChintanShah258/Still-to-Reel_Visualizing-the-Frame.git
cd Still-to-Reel_Visualizing-the-Frame
bash requirements.sh
```

## Usage
Download pretrained models from [Google Drive](https://drive.google.com/file/d/1ROxvB7D-vNYl4eYmIzZ5Gitg84amMd19/view?usp=sharing), then unzip and put them in the directory `ckpts`.

To achieve better motion estimation results and controllable animation, here we provide the controllable version. 

Firstly, use [labelme](https://github.com/wkentaro/labelme) to specify the target regions (masks) and desired movement directions (hints): 
```shell
conda activate 3d-cinemagraphy
cd demo/0/
labelme image.png
```
A screenshot here:
![labelme](Still-to-Reel_Visualizing-the-Frame/assets/labelme.png)

It is recommended to specify **short** hints rather than long hints to avoid artifacts. Please follow [labelme](https://github.com/wkentaro/labelme) for detailed instructions if needed.

After that, we can obtain an image.json file. Our next step is to convert the annotations stored in JSON format into datasets that can be used by our method:
```shell
labelme_json_to_dataset image.json  # this will generate a folder image_json
cd ../../
python scripts/generate_mask.py --inputdir demo/0/image_json
```

To generate a video using our Methods (modifications to the original code), you can use:
```shell
python demo_final.py -c configs/config.yaml --input_dir demo/0/ --ckpt_path ckpts/model_150000.pth --flow_scale 1.0 --ds_factor 1.0 --video_path up-down
```
- `input_dir`: input folder that contains src images.
- `ckpt_path`: checkpoint path.
- `flow_scale`: scale that used to control the speed of fluid, > 1.0 will slow down the fluid.
- `ds_factor`: downsample factor for the input images.

Results will be saved to the `input_dir/output`.

## Acknowledgement
This code borrows heavily from [3D-Cinemagraphs](https://github.com/xingyi-li/3d-cinemagraphy). We thank the respective authors for open sourcing their methods.
