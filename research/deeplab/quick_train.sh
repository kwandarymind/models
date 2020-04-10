#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Set the Model backbone variant
#MODEL_VAR="mobilenet_v2"
#MODEL_VAR="mobilenet_v3_large_seg"
#MODEL_VAR="mobilenet_v3_small_seg"
#MODEL_VAR="nas_hnasnet"
#MODEL_VAR="nas_pnasnet"
#MODEL_VAR="resnet_v1_101"
#MODEL_VAR="resnet_v1_101_beta"
#MODEL_VAR="resnet_v1_18"
#MODEL_VAR="resnet_v1_18_beta"
#MODEL_VAR="resnet_v1_50"
#MODEL_VAR="resnet_v1_50_beta"
#MODEL_VAR="resnet_v1_50_beta"
#MODEL_VAR="xception_41"
#MODEL_VAR="xception_65"
#MODEL_VAR="xception_71"

#func_train_v3 PICK_MODEL
func_train_v3() {
  echo "*******************************************************************************"
  echo "*"
  echo "*  Deeplab v3: ${1}"
  echo "*"
  echo "*******************************************************************************"

  # Use 0.007 when training on PASCAL augmented training set, train_aug. When
  # fine-tuning on PASCAL trainval set, use learning rate=0.0001.
  #      --base_learning_rate="${LEARNING_RATE}" \
  if [ FINETUNE_MODEL == "" ]
  then
     LEARNING_RATE=0.007
  else
     LEARNING_RATE=0.0001
  fi

  # For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
  ## Use 0.0001 for ResNet model variants.
  #WEIGHT_DECAY=0.00004
  #      --weight_decay="${WEIGHT_DECAY}" \


  case $1 in
    "resnet_v1_101" | \
    "resnet_v1_101_beta" | \
    "resnet_v1_18" | \
    "resnet_v1_18_beta" | \
    "resnet_v1_50" | \
    "resnet_v1_50_beta" | \
    "resnet_v1_50_beta")

      python "${MODEL_SRC_DIR}"/train.py \
              --logtostderr \
              --train_split="trainval" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --train_crop_size="${CROP_SIZE}" \
              --train_batch_size="${BATCH_SIZE}" \
              --training_number_of_steps="${NUM_ITERATIONS}" \
              --fine_tune_batch_norm=true \
              --tf_initial_checkpoint="${FINETUNE_MODEL}" \
              --train_logdir="${TRAIN_LOGDIR}" \
              --save_summaries_images=true \
              --dataset_dir="${PASCAL_DATASET}" \
              --weight_decay="${WEIGHT_DECAY}" \
              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
              --depth_multiplier="${DEPTH_MULTIPLIER}" \
              --multi_grid=1 --multi_grid=2 --multi_grid=4

      ;;
    "xception_65" )

      python "${MODEL_SRC_DIR}"/train.py \
              --logtostderr \
              --train_split="trainval" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --train_crop_size="${CROP_SIZE}" \
              --train_batch_size="${BATCH_SIZE}" \
              --training_number_of_steps="${NUM_ITERATIONS}" \
              --fine_tune_batch_norm=true \
              --tf_initial_checkpoint="${FINETUNE_MODEL}" \
              --train_logdir="${TRAIN_LOGDIR}" \
              --save_summaries_images=true \
              --dataset_dir="${PASCAL_DATASET}" \
              --weight_decay="${WEIGHT_DECAY}"
      ;;
    "xception_41" | \
    "xception_71" | \
    "mobilenet_v2" | \
    "mobilenet_v3_large_seg" | \
    "mobilenet_v3_small_seg" | \
    "nas_hnasnet" | \
    "nas_pnasnet")

      python "${MODEL_SRC_DIR}"/train.py \
              --logtostderr \
              --train_split="trainval" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --train_crop_size="${CROP_SIZE}" \
              --train_batch_size="${BATCH_SIZE}" \
              --training_number_of_steps="${NUM_ITERATIONS}" \
              --fine_tune_batch_norm=true \
              --tf_initial_checkpoint="${FINETUNE_MODEL}" \
              --train_logdir="${TRAIN_LOGDIR}" \
              --save_summaries_images=true \
              --dataset_dir="${PASCAL_DATASET}" \
              --weight_decay="${WEIGHT_DECAY}"
#              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18
#              --depth_multiplier="${DEPTH_MULTIPLIER}"
      ;;
    *)
      echo " ##### NONE ##########"
      exit
      ;;

  esac;
}


#func_train_v3plus PICK_MODEL
func_train_v3plus() {
  echo "*******************************************************************************"
  echo "*"
  echo "*  Deeplab v3 Plus: ${1}"
  echo "*"
  echo "*******************************************************************************"

  case $1 in
    "resnet_v1_101" | \
    "resnet_v1_101_beta" | \
    "resnet_v1_18" | \
    "resnet_v1_18_beta" | \
    "resnet_v1_50" | \
    "resnet_v1_50_beta")

      python "${MODEL_SRC_DIR}"/train.py \
              --logtostderr \
              --train_split="trainval" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --train_crop_size="${CROP_SIZE}" \
              --train_batch_size="${BATCH_SIZE}" \
              --training_number_of_steps="${NUM_ITERATIONS}" \
              --fine_tune_batch_norm=true \
              --tf_initial_checkpoint="${FINETUNE_MODEL}" \
              --train_logdir="${TRAIN_LOGDIR}" \
              --save_summaries_images=true \
              --dataset_dir="${PASCAL_DATASET}" \
              --weight_decay="${WEIGHT_DECAY}" \
              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
              --depth_multiplier="${DEPTH_MULTIPLIER}" \
              --multi_grid=1 --multi_grid=2 --multi_grid=4 \
              --decoder_output_stride="${DECODER_OUTPUT_STRIDE}"
      ;;
    "mobilenet_v2" | \
    "mobilenet_v3_large_seg" | \
    "mobilenet_v3_small_seg" | \
    "nas_hnasnet" | \
    "nas_pnasnet")

      python "${MODEL_SRC_DIR}"/train.py \
              --logtostderr \
              --train_split="trainval" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --train_crop_size="${CROP_SIZE}" \
              --train_batch_size="${BATCH_SIZE}" \
              --training_number_of_steps="${NUM_ITERATIONS}" \
              --fine_tune_batch_norm=true \
              --tf_initial_checkpoint="${FINETUNE_MODEL}" \
              --train_logdir="${TRAIN_LOGDIR}" \
              --save_summaries_images=true \
              --dataset_dir="${PASCAL_DATASET}" \
              --weight_decay="${WEIGHT_DECAY}" \
              --depth_multiplier="${DEPTH_MULTIPLIER}" \
              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
              --decoder_output_stride="${DECODER_OUTPUT_STRIDE}"
      ;;
    "xception_41" | \
    "xception_65" | \
    "xception_71")

      python "${MODEL_SRC_DIR}"/train.py \
              --logtostderr \
              --train_split="trainval" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --train_crop_size="${CROP_SIZE}" \
              --train_batch_size="${BATCH_SIZE}" \
              --training_number_of_steps="${NUM_ITERATIONS}" \
              --fine_tune_batch_norm=true \
              --tf_initial_checkpoint="${FINETUNE_MODEL}" \
              --train_logdir="${TRAIN_LOGDIR}" \
              --save_summaries_images=true \
              --dataset_dir="${PASCAL_DATASET}" \
              --weight_decay="${WEIGHT_DECAY}" \
              --depth_multiplier="${DEPTH_MULTIPLIER}" \
              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
              --decoder_output_stride="${DECODER_OUTPUT_STRIDE}"
      ;;
    *)
      echo " ##### NONE ##########"
      exit
      ;;

  esac;
}

func_eval_v3() {
  echo "*******************************************************************************"
  echo "*"
  echo "*  Deeplab v3 Eval: ${1}"
  echo "*"
  echo "*******************************************************************************"

  case $1 in
    "xception_41" | \
    "xception_65" | \
    "xception_71" | \
    "resnet_v1_101" | \
    "resnet_v1_101_beta" | \
    "resnet_v1_18" | \
    "resnet_v1_18_beta" | \
    "resnet_v1_50" | \
    "resnet_v1_50_beta" | \
    "resnet_v1_50_beta" | \
    "nas_hnasnet" | \
    "nas_pnasnet")

      python "${MODEL_SRC_DIR}"/eval.py \
              --logtostderr \
              --eval_split="val" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --eval_crop_size="${CROP_SIZE}" \
              --checkpoint_dir="${TRAIN_LOGDIR}" \
              --eval_logdir="${EVAL_LOGDIR}" \
              --dataset_dir="${PASCAL_DATASET}" \
              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
              --max_number_of_evaluations=1 \
              --save_summaries_images=true
      ;;

    "mobilenet_v2" | \
    "mobilenet_v3_large_seg" | \
    "mobilenet_v3_small_seg")

      python "${MODEL_SRC_DIR}"/eval.py \
              --logtostderr \
              --eval_split="val" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --eval_crop_size="${CROP_SIZE}" \
              --checkpoint_dir="${TRAIN_LOGDIR}" \
              --eval_logdir="${EVAL_LOGDIR}" \
              --dataset_dir="${PASCAL_DATASET}" \
              --max_number_of_evaluations=1 \
              --save_summaries_images=true
      ;;
    *)
      echo " ##### NONE ##########"
      exit
      ;;

  esac;
}

func_eval_v3plus() {
  echo "*******************************************************************************"
  echo "*"
  echo "*  Deeplab v3+ Eval: ${1}"
  echo "*"
  echo "*******************************************************************************"

  case $1 in
    "resnet_v1_101" | \
    "resnet_v1_101_beta" | \
    "resnet_v1_18" | \
    "resnet_v1_18_beta" | \
    "resnet_v1_50" | \
    "resnet_v1_50_beta" | \
    "resnet_v1_50_beta")

      python "${MODEL_SRC_DIR}"/eval.py \
              --logtostderr \
              --eval_split="val" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --eval_crop_size="${CROP_SIZE}" \
              --checkpoint_dir="${TRAIN_LOGDIR}" \
              --eval_logdir="${EVAL_LOGDIR}" \
              --dataset_dir="${PASCAL_DATASET}" \
              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
              --max_number_of_evaluations=1 \
              --decoder_output_stride="${DECODER_OUTPUT_STRIDE}" \
              --save_summaries_images=true
      ;;
    "xception_41" | \
    "xception_65" | \
    "xception_71" | \
    "mobilenet_v2" | \
    "mobilenet_v3_large_seg" | \
    "mobilenet_v3_small_seg" | \
    "nas_hnasnet" | \
    "nas_pnasnet")

      python "${MODEL_SRC_DIR}"/eval.py \
              --logtostderr \
              --eval_split="val" \
              --model_variant="${1}" \
              --output_stride="${OUTPUT_STRIDE}" \
              --eval_crop_size="${CROP_SIZE}" \
              --checkpoint_dir="${TRAIN_LOGDIR}" \
              --eval_logdir="${EVAL_LOGDIR}" \
              --dataset_dir="${PASCAL_DATASET}" \
              --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 \
              --max_number_of_evaluations=1 \
              --decoder_output_stride="${DECODER_OUTPUT_STRIDE}" \
              --save_summaries_images=true
      ;;
    *)
      echo " ##### NONE ##########"
      exit
      ;;

  esac;
}

# --------------------------------------------------------------------------------------------------------

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)

CROP_SIZE="513,513"
MODEL_SRC_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="/home/mltrain/datasets/pascal_voc/VOCdevkit/VOC2012"
PASCAL_DATASET="${DATASET_DIR}/tfrecord"
TRAINING_DIR="/home/mltrain/deeplabv3"
CHECKPOINT_DIR="/home/mltrain/init_models"
mkdir -p "${CHECKPOINT_DIR}"

cd "${CURRENT_DIR}"

export CUDA_VISIBLE_DEVICES=0

# --------------------------------------------------------------------------------------------------------
# MODEL Configs

PICK_MODEL=1

BATCH_SIZE=2

if [ "$PICK_MODEL" == 0 ];
then
  # xception_65_imagenet_coco
  MODEL_VAR="xception_65"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/deeplabv3_pascal_train_aug_2018_01_04/model.ckpt"
  OUTPUT_STRIDE=16
  DECODER_OUTPUT_STRIDE=4
  WEIGHT_DECAY=0.00004
  DEPTH_MULTIPLIER=1.0
  BATCH_SIZE=4

elif [ "$PICK_MODEL" == 1 ];
then
  # xception_65_imagenet_coco
  MODEL_VAR="xception_65"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/deeplabv3_pascal_train_aug_2018_01_04/model.ckpt"
  OUTPUT_STRIDE=8
  DECODER_OUTPUT_STRIDE=None
  WEIGHT_DECAY=0.00004
  DEPTH_MULTIPLIER=1.0
  BATCH_SIZE=4
elif [ "$PICK_MODEL" == 2 ];
then
  # mobilenetv2_coco_voc_trainval
  MODEL_VAR="mobilenet_v2"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/deeplabv3_mnv2_pascal_trainval_2018_01_29/model.ckpt-30000"
  OUTPUT_STRIDE=8
  DECODER_OUTPUT_STRIDE=None
  WEIGHT_DECAY=0.00004
  DEPTH_MULTIPLIER=0.5
elif [ "$PICK_MODEL" == 2.1 ];
then
  # mobilenetv2_coco_voc_trainval
  MODEL_VAR="mobilenet_v2"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/deeplabv3_mnv2_pascal_train_aug_2018_01_29/model.ckpt-30000"
  OUTPUT_STRIDE=8
  DECODER_OUTPUT_STRIDE=None
  WEIGHT_DECAY=0.00004
  BATCH_SIZE=8
  DEPTH_MULTIPLIER=0.5
elif [ "$PICK_MODEL" == 3 ];
then
  # deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01
  MODEL_VAR="mobilenet_v2"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/deeplabv3_mnv2_dm05_pascal_trainval_2018_10_01/model.ckpt"
  OUTPUT_STRIDE=8
  DECODER_OUTPUT_STRIDE=8
  WEIGHT_DECAY=0.00004
  DEPTH_MULTIPLIER=0.5
  BATCH_SIZE=4
elif [ "$PICK_MODEL" == 4 ];
then
  # resnet_v1_50_2018_05_04
  MODEL_VAR="resnet_v1_50_beta"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/resnet_v1_50_2018_05_04/model.ckpt"
  OUTPUT_STRIDE=8
  DECODER_OUTPUT_STRIDE=8
  WEIGHT_DECAY=0.0002
  DEPTH_MULTIPLIER=1.0
  BATCH_SIZE=4
elif [ "$PICK_MODEL" == 4.1 ];
then
  # resnet_v1_50_2018_05_04
  MODEL_VAR="resnet_v1_50_beta"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/resnet_v1_50_2018_05_04/model.ckpt"
  OUTPUT_STRIDE=16
  DECODER_OUTPUT_STRIDE=None
  WEIGHT_DECAY=0.0001
  DEPTH_MULTIPLIER=1.0
elif [ "$PICK_MODEL" == 5 ];
then
  # resnet_v1_101
  MODEL_VAR="resnet_v1_101_beta"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/resnet_v1_101_2018_05_04/model.ckpt"
  OUTPUT_STRIDE=16
  DECODER_OUTPUT_STRIDE=4
  WEIGHT_DECAY=0.0001
  DEPTH_MULTIPLIER=1.0
  BATCH_SIZE=4
elif [ "$PICK_MODEL" == 6 ];
then
  # resnet_v1_101
  MODEL_VAR="resnet_v1_101_beta"
  FINETUNE_MODEL="${CHECKPOINT_DIR}/resnet_v1_101_2018_05_04/model.ckpt"
  OUTPUT_STRIDE=16
  DECODER_OUTPUT_STRIDE=4
  WEIGHT_DECAY=0.0001
  DEPTH_MULTIPLIER=1.0
  BATCH_SIZE=4
else
  echo "No model selected"
  exit
fi

EXP_FOLDER="$PICK_MODEL-$MODEL_VAR"

TRAIN_LOGDIR="$TRAINING_DIR/${EXP_FOLDER}/train"
EVAL_LOGDIR="$TRAINING_DIR/${EXP_FOLDER}/eval"
VIS_LOGDIR="$TRAINING_DIR/${EXP_FOLDER}/vis"
EXPORT_DIR="$TRAINING_DIR/${EXP_FOLDER}/export"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Comment this out to train from scratch
#FINETUNE_MODEL=""

# Train iterations.
NUM_ITERATIONS=100

case $1 in
  clean)
      echo "cleaning directory: $TRAIN_LOGDIR"
      rm -f $TRAIN_LOGDIR/*
      echo "cleaning directory: $EVAL_LOGDIR"
      rm -f $EVAL_LOGDIR/*
    ;;
  train)

    if [ "$2" == "clean" ];
    then
      echo "cleaning directory: $TRAIN_LOGDIR"
      rm -f $TRAIN_LOGDIR/*
    elif [ $DECODER_OUTPUT_STRIDE == "None" ]
    then
      # deeplab v3
      func_train_v3 $MODEL_VAR

    else
      # deeplab v3+
      func_train_v3plus $MODEL_VAR
    fi
    ;;
  eval)

    if [ "$2" == "clean" ];
    then
      echo "cleaning directory: $EVAL_LOGDIR"
      rm -f $EVAL_LOGDIR/*
    elif [ $DECODER_OUTPUT_STRIDE == "None" ]
    then
      # deeplab v3
      func_eval_v3 $MODEL_VAR

    else
      # deeplab v3+
      func_eval_v3plus $MODEL_VAR
    fi
    ;;
  vis)

    # Visualize the results.
    python "${MODEL_SRC_DIR}"/vis.py \
      --logtostderr \
      --vis_split="val" \
      --model_variant="${MODEL_VAR}" \
      --vis_crop_size="${CROP_SIZE}" \
      --checkpoint_dir="${TRAIN_LOGDIR}" \
      --vis_logdir="${VIS_LOGDIR}" \
      --dataset_dir="${PASCAL_DATASET}" \
      --max_number_of_iterations=1

    ;;
  export)

    # Export the trained checkpoint.
    CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
    EXPORT_PATH="${EXPORT_DIR}/${MODEL_VAR}-${PICK_MODEL}.pb"

    python "${MODEL_SRC_DIR}"/export_model.py \
      --logtostderr \
      --checkpoint_path="${CKPT_PATH}" \
      --export_path="${EXPORT_PATH}" \
      --model_variant="${MODEL_VAR}" \
      --num_classes=21 \
      --crop_size=513 \
      --crop_size=513 \
      --atrous_rates=6 \
      --atrous_rates=12 \
      --atrous_rates=18 \
      --output_stride=16 \
      --decoder_output_stride=4 \
      --inference_scales=1.0

    ;;
  *)
    echo "Usage: $0 train / eval / vis / export / clean"
    ;;
esac

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.