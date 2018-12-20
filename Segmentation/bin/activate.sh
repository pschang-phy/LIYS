#!/bin/bash


if [ -z ${WORKENV+x} ]
then

    echo "****** Enter Segmentation environment ******"
    BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    export PATH=${PATH}:${BINDIR}

    DIR=$BINDIR
    while ! [ -f ${DIR}/.top ]
    do
        DIR=${DIR%/*}
    done

    TOP=${DIR}
    export PYTHONPATH=$PYTHONPATH:${TOP}/Libs/Tensorflow/models:${TOP}/Libs/Tensorflow/models/slim

    ln -s ${TOP}/Libs/Tensorflow/models/deeplab/train.py ${BINDIR}
    ln -s ${TOP}/Libs/Tensorflow/models/deeplab/datasets/build_voc2012_data.py ${BINDIR}/convert2tfrecord.py
    ln -s ${TOP}/Libs/Tensorflow/models/deeplab/datasets/remove_gt_colormap.py ${BINDIR}

    unset BINDIR
    unset DIR

    export WORKENV="Segmentation"
fi
