#!/bin/bash


if [ -z ${WORKENV+x} ]
then

    echo "****** Enter Style Transfer environment ******"
    BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
    export PATH=${PATH}:${BINDIR}

    DIR=$BINDIR
    while ! [ -f ${DIR}/.top ]
    do
        DIR=${DIR%/*}
    done

    TOP=${DIR}
    PYTHONPATH=$PYTHONPATH:${TOP}/Libs/Tensorflow/fast-style-transfer
    PYTHONPATH=$PYTHONPATH:${TOP}/Libs/Tensorflow/fast-style-transfer/src
    export PYTHONPATH

    ln -s ${TOP}/Libs/Tensorflow/fast-style-transfer/evaluate.py ${BINDIR}
    ln -s ${TOP}/Libs/Tensorflow/fast-style-transfer/transform_video.py ${BINDIR}
    ln -s ${TOP}/Libs/Tensorflow/fast-style-transfer/style.py ${BINDIR}

    unset BINDIR
    unset DIR

    export WORKENV="StyleTransfer"
fi
