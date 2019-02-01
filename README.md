AIA Project - Live In Your Style
===========================
This is AIA project **Live In Your Style**, which has objective
* Segment person in an image frame and style transfer background
* Integrate with cloud computing and edge application

This repository is the source code for training required deep learning model.

![](/img/pipeline.jpg)

****

## Code base structure

In this code base, it maintains the code for two model-related tasks
* Segmentation (2 classes, background and person)
* Style transfer

In the top folder, it has subdirectories
* Libs
    > The required libraries or model code will be in here. For example, because we use DeeplabV3+ model and Slim library from Tensorflow, a copy of their code base will be placed in here.
* Segmentation
    > This could be a workspace for the preparation of segmentation model. It may contains the training, validation, test tools.
* StyleTransfer
    > Similar to Segmentation folder, this folder is a workspace for the preparation of style transfer model.

## Workspace Environment

For each workspace folder, *Segmentation* and *StyleTransfer*, it has 2 shell scripts, under **bin** folder, to **activate** and **deactivate** the working environment.
* activate.sh
* deactivate.sh

After launching the activate.sh script
```Bash
$ source activate.sh
```
current environment will be activated to facilitate working on model-related tasks by
* Set proper environment variables
    * TOP

        It stores the full path of this code base.
        You could inspect it by
        ```Bash
        $ echo $TOP 
        /home/jovyan/AT073_10_Orig_Style
        ```
        Or use it to refer your file in your code base, for example
        ```Base
        $ python ${TOP}/Segmentation/bin/train.py
        ```
    * WORKENV

        It stores string about the variant of activated working environment, it may be one of **Segmentation** or **StyleTransfer**. You could inspect it by
        ```Bash
        $ echo $WORKENV
        Segmentation
        ```
* Add the path of required Python libraries in **PYTHONPATH**
* Create shortcuts, in bin folder, for useful tools

To reset the activated working environment, just run as below
```Bash
$ source ${TOP}/${WORKENV}/bin/deactivate.sh
```
It will clear stuff that the activate.sh setup.