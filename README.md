# Bayes' Rays
### [Project Page](https://bayesrays.github.io/) | [Paper](https://arxiv.org/abs/2309.03185)
<img src="https://github.com/BayesRays/bayesrays.github.io/raw/main/video/demo1.gif" height=400>

### Installation
Bayes' Rays is built on top of [Nerfstudio](https://docs.nerf.studio/en/latest/).
After cloning Bayes' Rays repository, install Nerfstudio as a package by following the installation guide on [Nerfstudio installation page](https://docs.nerf.studio/en/latest/quickstart/installation.html)

Specifically, perform the following steps
1. [Create environment](https://docs.nerf.studio/en/latest/quickstart/installation.html#create-environment)
2. [Dependencies](https://docs.nerf.studio/en/latest/quickstart/installation.html#dependencies)
3. [Installing nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html#installing-nerfstudio). Follow the **From pip** guidelines, no need to clone Nerfstudio.


Then install Bayes' Rays as a package using `pip install -e .`
This will allow you to run uncertainty-related commands in the terminal like `ns-uncertainty`.

The code is tested to run with Nerfstudio version 0.3.1 (commit hash `e2e5637d05fc281a28abe7c4f9c86a93e130a085`) and version 0.3.3.

For experiments on clean-up task and comparing with [Nerfbusters](https://github.com/ethanweber/nerfbusters) the `nerfbusters-changes` [branch](https://github.com/nerfstudio-project/nerfstudio/tree/nerfbusters-changes) of Nerfstudio is used, to ensure fair comparison with Nerfbusters code. Therefore, if you wish to compare directly to Nerfbusters, install `nerfbusters-changes` branch of Nerfstudio instead.

### Running

First train a NeRF model using `ns-train` command in Nerfstudio. Currently our code only has support for Nerfacto, instant-NGP and Mip-NeRF models. Here is the demo example on the 'Loader Truck' scene. Download the dataset from [here](https://drive.google.com/file/d/1ZVjiVAQM7VhngGloi_5FdFy4MwKR8SzL/view?usp=sharing).

1) Train a Nerfacto model normally:
```
ns-train nerfacto  --vis viewer --data {PATH_TO_DATA} --experiment-name {EXP_NAME} --output-dir {OUTPUT_DIR} --timestamp main  --relative-model-dir=nerfstudio_models  --max-num-iterations=30000  nb-dataparser --eval-mode eval-frame-index  --train-frame-indices 0  --eval-frame-indices 1 --downscale-factor 2 --center_method focus
```
Note in this example we use `nb-dataparser` (instead of the usual `nerfstudio-data`), which is implemented based on nerfbusters-changes branch, and enables using a small part of the dataset as train set and the rest as test.

2) Then extract uncertainty:

```
ns-uncertainty --load-config={PATH_TO_CONFIG} --output-path={PATH_TO_UNCERTAINTY} --lod 8
```
Note `{PATH_TO_UNCERTAINTY}` must be a path to `.npy` file and 2^`lod` denotes the uncertainty grid length (i.e. the grid length in the example above is 256).

3) Render or view the uncertainty!

To run viewer in the browser and visualize uncertainty:

```
ns-viewer-u --load-config={PATH_TO_CONFIG} --unc_path {PATH_TO_UNCERTAINTY}
```
Note that  `ns-viewer-u` is a modified version of `ns-viewer` command in Nerfstudio and works with the same options. However it additionally takes `--unc_path` option as the path to the extracted uncertainty `.npy` file.

In the browser, toggle `Output Render` to uncertainty and optionally set colormap to `inferno`. For performing clean-up task interactively, use the `Filter Threshold` slider to gradually filter out uncertain parts!

To render uncertainty use `ns-render-u`. :
```
ns-render-u camera-path --load-config={PATH_TO_CONFIG} --unc_path {PATH_TO_UNCERTAINTY} --output-path {PATH_TO_VIDEO} --downscale-factor 2  --rendered_output_names rgb depth uncertainty  --filter-out True --filter-thresh 1. --white-bg True --black-bg False  --camera_path_filename {PATH_TO_CAMERA_PATH}
```
For the 'Loader Truck' scene, `{PATH_TO_CAMERA_PATH}` is inclued as `camera_path.json` in the downloaded dataset.

Note that `ns-render-u` is a modified version of `ns-render` command in Nerfstudio and works with the same options. However it additionally takes `--unc_path` option as the path to the extracted uncertainty `.npy` file.  If the option `--filter-out True` is set and the threshold `--filter-thresh` is set to a value less than 1, then the clean-up task is performed with the given threshold. With `--white-bg` and `--black-bg` control the color of rendered empty parts after clean-up.

### Uncertainty Evaluation
The uncertainity evaluation is done on the Light Field (LF) dataset and ScanNet Scenes by correlating depth error with uncertainity, computing via the AUSE metric.

For Light Field (LF) dataset download the data [here](https://drive.google.com/file/d/1U-Hly00DmqtAIGaPkF-Eu_B_q0Frsbh1/view?usp=sharing) and for ScanNet scenes use the data [here](https://drive.google.com/file/d/17j0l6vD1YLY0F9ghWDszyCuiZkuoyWvS/view?usp=sharing). Note that these folders contain the ground truth depth files for test views, alongside the approximate scale in `scale_parameters.txt`. The scale parameter is computed using `utils/scale_solver.py` and is used for the purpose of evaluation to solve the scale ambiguity of NeRF for depth comparison with GT depth. To run `scale_solver.py` on another scene or to verify the provided scales, you need to have the sparse COLMAP generated point cloud and the ground truth depthmaps then you can run the script as:

```
python /path/to/scale_solver.py --dataset [scannet, LF] --scene [scene's name]
```

Train Nerfacto models using the [provided settings](#training-settings), and then evaluate by:

```
ns-eval-u --load-config {PATH_TO_CONFIG} --output-path {PATH_TO_METRICS} --unc_path {PATH_TO_UNCERTAINTY} --dataset_path {PATH_TO_DATA}
```
where `{PATH_TO_METRICS}` is a json file.

<a name="training-settings"></a>
<details>
  <summary>Training Settings</summary>
For ScanNet dataset (setting {SCENE_NAME} to scene_001, scene_079, scene_316 or scene_158):

```
ns-train nerfacto --vis viewer --data {PATH_TO_DATA} --experiment-name {SCENE_NAME} --output-dir {OUTPUT_DIR} --timestamp main --relative-model-dir=nerfstudio_models/ --steps-per-save=2000 --max-num-iterations=30000 --logging.local-writer.enable=False --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.proposal-initial-sampler uniform --pipeline.model.use-average-appearance-embedding True --pipeline.model.background-color random --pipeline.model.disable-scene-contraction True  --pipeline.model.distortion-loss-mult 0.001 --pipeline.model.max-res 4096 sparse-nerfstudio --dataset-name {SCENE_NAME}
```

For LF dataset (setting {SCENE_NAME} to statue, torch, basket or africa):

```
ns-train nerfacto  --vis viewer --data {PATH_TO_DATA} --experiment-name {SCENE_NAME} --output-dir {OUTPUT_DIR} --timestamp main --relative-model-dir=nerfstudio_models/ --steps-per-save=2000 --max-num-iterations=30000 --logging.local-writer.enable=False --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.disable-scene-contraction True  --pipeline.model.distortion-loss-mult 0.0  --pipeline.model.near-plane 1 --pipeline.model.far-plane 100. --pipeline.model.use-average-appearance-embedding True --pipeline.model.proposal-initial-sampler uniform --pipeline.model.background-color random  --pipeline.model.max-res 4096 sparse-nerfstudio --dataset-name {SCENE_NAME}
```
</details>


### NeRF Clean Up Evaluation
The NeRF clean up task is performed on the Nerfbusters dataset, which can be downloaded [here](https://drive.google.com/uc?id=197bfxxvDEJr9lPf5_QZzbItsBnNfChOt).

First make sure that you have `nerfbuster-changes` branch of `nerfstudio` installed.

Train Nerfacto models using the [provided settings](#training-settings-nerfbusters), and then evaluate by:
```
ns-eval-u --load-config {PATH_TO_CONFIG} --output-path {PATH_TO_METRICS} --unc_path {PATH_TO_UNCERTAINTY} --filter-out
```

Optionally, you can use `--nb-mask` option to have the exact same metric definition as Nerfbusters (i.e. with the pseudo-ground-truth visibility masks applied.). The visibility masks can be downloaded [here](https://drive.google.com/file/d/1Wy77GlKCF4V7Z4wb0lSfI7iFBTNOqHfu/view?usp=sharing).
If running a comparison on scene `x` against Nerfbusters, you should run the uncertainty commands (extraction and evaluation) on the `x---nerfacto` model in the Nerfbusters outputs (i.e Nerfbusters "baseline" output model that does **not** use the Nerfbusters postprocessing techniques). Evaluation can be run as:
```
ns-eval-u --load-config {PATH_TO_CONFIG} --output-path {PATH_TO_METRICS} --unc_path {PATH_TO_UNCERTAINTY} --filter-out --nb-mask True --visibility-path {PATH_TO_VISIBILITY_MASKS}
```
where {PATH_TO_VISIBILITY_MASKS} are the paths to the scene specific visibility masks.

<a name="training-settings-nerfbusters"></a>
<details>
  <summary>Training Settings</summary>
For Nerfbusters initial training model (using `nerfbusters-changes` branch of Nerfstudio):

```
ns-train nerfacto --vis viewer --data {PATH_TO_DATA} --experiment-name nerfbusters --output-dir {OUTPUT_DIR} --timestamp base --relative-model-dir=nerfstudio_models/ --max-num-iterations=30000  nerfstudio-data --eval-mode eval-frame-index --train-frame-indices 0 --eval-frame-indices 1
```

which is then passed to Nerfbusters pipeline to get baseline and Nerfbusters postprocessed models (Baseline model is just the same Nerfacto model trained for 5K longer).
</details>

### Citation
```
@article{goli2023,
    title={{Bayes' Rays}: Uncertainty Quantification in Neural Radiance Fields},
    author={Lily Goli, Cody Reading, Silvia Sellán, Alec Jacobson, Andrea Tagliasacchi},
    journal={arXiv},
    year={2023}
}
```
