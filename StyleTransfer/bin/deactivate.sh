#!/bin/bash


if [[ -v WORKENV ]]
then
    echo "****** Exit $WORKENV environment ******"

    BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

    REPLACESTR=$(echo ${BINDIR} | sed 's/\//\\\//g')
    export PATH=$(echo ${PATH} | sed "s/:${REPLACESTR}//")
    unset REPLACESTR

    DIR=$BINDIR
    while ! [ -f ${DIR}/.top ]
    do
        DIR=${DIR%/*}
    done

    unset PYTHONPATH

    rm -f ${BINDIR}/eval.py \
          ${BINDIR}/train.py

    unset BINDIR
    unset DIR

    unset TOP
    unset WORKENV
fi
