# MOT

## Description

## Installation

```shell
env_name=MOT
conda create -n ${env_name} python=3.9 -y

conda activate ${env_name}

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
pip install cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

pip install -r requirements.txt

python setup.py build_ext --inplace
```

## Data preparation

### Download SMD,SDS and put them under ./datasets in the following structure

```
datasets
|——————smd
|        └——————NIR
|        └——————VIS_Onboard
|        └——————VIS_Onshore
└——————sds
|        └——————train
|        └——————val
|        └——————test
```

### Prepare SMD dataset

```shell
export PYTHONPATH="$(pwd):$PYTHONPATH"
python tools/convert_smd_to_coco.py --convert_to_mot --split_data \
    --generate_seqmaps --convert_gt_to_coco_ultralytics --process_datasets ALL
```

### Prepare SDS dataset

```shell
export PYTHONPATH="$(pwd):$PYTHONPATH"
python tools/convert_sds_to_coco.py --convert_to_mot \
    --generate_seqmaps --convert_gt_to_coco_ultralytics --process_datasets Object
```

### Prepare ReID datasets

```shell
export PYTHONPATH="$(pwd):$PYTHONPATH"
# SMD
python tools/fastreid/data/generate_smd_patches.py --dataset_type VIS --save_patches
python tools/fastreid/data/generate_smd_patches.py --dataset_type NIR --save_patches
python tools/fastreid/data/generate_smd_patches.py --dataset_type VISNIR --save_patches

# SDS
python tools/fastreid/data/generate_sds_patches.py --save_patches --dataset_type Object
```

## Model Zoo

Download and store the [checkpoints](https://drive.google.com/drive/folders/1hjV3ic43EcMeNfAICRIa7BagJ_KJYL9_?usp=sharing) in ./pretrained folder

Download the [detector and ReID cache files](https://drive.google.com/drive/folders/1hjV3ic43EcMeNfAICRIa7BagJ_KJYL9_?usp=sharing) to directly use them for tracking without inference detection and ReID feature extraction.

## Training

You can directly use the checkpoint from model zoo for inference or directly use the cache files from model zoo for tracking without inference.
If you want to train your own model, please refer to the following commands.

## Detector Training

```shell
export PYTHONPATH="$(pwd):$PYTHONPATH"
# YOLO v5/6/8/9/10/11
export CUDA_VISIBLE_DEVICES=0
full_model_name=yolov5x # yolov5x/yolov6x/yolov8x/yolov9e/yolov10x/yolov11x
config_name=${full_model_name} # ${full_model_name}_CAG
data_name=SMDNIR # SMDVIS/SMDNIR/SMDVISNIR/SDSObject
output_path="Outputs/YOLO/${config_name}/train/${data_name}"
batch_size=8
device=0
num_workers=8
python tools/yolo/train.py \
    --model "${full_model_name}".yaml \
    --pretrained \
    --weights ./pretrained/"${full_model_name}".pt \
    --data exps/yolo/data/"${data_name}".yaml \
    --epochs 100 \
    --batch "${batch_size}" \
    --imgsz 640 \
    --save \
    --save_period 20 \
    --device "${device}" \
    --workers "${num_workers}" \
    --project "${output_path}" \
    --verbose \
    --deterministic \
    --amp \
    --val \
    --plots
    
# YOLOX
export CUDA_VISIBLE_DEVICES=0
config_name=yolox_x_SMDNIR # yolox_x_SMDNIR/yolox_x_SMDNIR_CAG/yolox_x_SMDVIS/yolox_x_SMDVIS_CAG/yolox_x_SMDVISNIR/yolox_x_SMDVISNIR_CAG/yolox_x_SDSObject/yolox_x_SDSObject_CAG
full_model_name=yolox_x
experiment_name=${config_name}/train
batch_size=8
num_workers=8
num_device=1
python tools/yolox/train.py \
    -f exps/yolox/mot/"${config_name}".py \
    -b "${batch_size}" 
    -d "${num_device}" 
    --workers "${num_workers}" \
    --output_dir "${output_path}" \
    --experiment_name "${experiment_name}" \
    --fp16 -o \
    -c pretrained/${full_model_name}.pth.tar 
```

## ReID Training

```shell
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
batch_size=8
num_workers=8
num_device=1
data_name=SMDNIR # SMDVIS/SMDNIR/SMDVISNIR/SDSObject
config_file=exps/fastreid/configs/SDSObject/sbs_S50.yml
python tools/fastreid/tools/train_net.py \
    --config-file ${config_file} \
    --num-gpus "${num_device}" \
    TEST.IMS_PER_BATCH "${batch_size}" \
    SOLVER.IMS_PER_BATCH "${batch_size}" \
    DATALOADER.NUM_WORKERS "${num_workers}" \
    SOLVER.MAX_EPOCH 100
```

## Tracker Evaluation

```shell
export PYTHONPATH="$(pwd):$PYTHONPATH"

python tools/batch_run/batch_eval.py --params tools/batch_run/tracker_params_sds.yaml --tracker_names sort,deepsort,bytetrack,ocsort,strongsort,hybridsort,botsort,GNN_NonProb_StoneSoup,GNN_Prob_StoneSoup,JPDAwithNBest,JPDAwithLBP,JPDAwithEHM,JPDAwithEHM2,MHTStoneSoup --dataset_names SDSObject --detector_categories  single_cls --detector_name yolox_x --data_split_names val --num_processes 2

python tools/batch_run/batch_eval.py --params tools/batch_run/tracker_params_smd.yaml --tracker_names sort,deepsort,bytetrack,ocsort,strongsort,hybridsort,botsort,GNN_NonProb_StoneSoup,GNN_Prob_StoneSoup,JPDAwithNBest,JPDAwithLBP,JPDAwithEHM,JPDAwithEHM2,MHTStoneSoup --dataset_names SMDNIR,SMDVIS --detector_categories  single_cls --detector_name yolox_x --data_split_names val,test --num_processes 2


```

## Acknowledgement

This codebase builds upon, but is not limited to, the following open-source projects:

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX.git)
- [ultralytics](https://github.com/ultralytics/ultralytics.git)
- [FastReID](https://github.com/JDAI-CV/fast-reid.git)
- [HybridSORT](https://github.com/ymzis69/HybridSORT.git)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [murty](https://github.com/erikbohnsack/murty.git)
- [StoneSoup](https://stonesoup.readthedocs.io)
