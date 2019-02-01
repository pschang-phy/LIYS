Segmentation Model Training
===========================
This guide gives the training SOP and description of our models used in LIYS project.

We use segmentation model called **DeepLabv3+** for our background/person segmentation tasks. The detail of this model could refer to
* The open source GitHub
* Related papers
    * **Rethinking Atrous Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.
    * **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.


****

## Segmentation Dataset Preparation

This section describes dataset preparation to train our segmentation model.

### Annotation Convention

For each **train / validation** data, it has a pair of **(image, mask)** files, where
* **image** is a JPEG image file
* **mask** is a PNG image file with **8 bits color depth**, gives segmentation labels

The labels in ‘mask’ have the color map, (R,G,B), defined as
* Class 0 - (0, 0, 0)
* Class 1 - (128, 0, 0)
* Class 2 - (0, 128, 0)
* Class 3 - (128, 128, 0)
* ...

Below are examples of masks (Only contain Class 0 and Class 1, Person)

|Image|Mask|
|:------:|:----:|
|![](/img/COCO_sample1.jpg)|![](/img/COCO_sample1-mask.png)|
|![](/img/COCO_sample2.jpg)|![](/img/COCO_sample2-mask.png)|

#### Suggested Archive Layout

Here gives a suggested layout for archiving a prepared dataset.
For example, we have a dataset with folder name **‘xxx_dataset’**, with below layout
```
xxx_dataset
├── ImageSets
│   ├── train.txt
│   └── val.txt
│
├── JPEGImages
│   ├── 2009_000347.jpg
│   ├── 2009_000350.jpg
│   ├── 2009_000351.jpg
│   ├── 2009_000354.jpg
│   ├── 2009_000356.jpg
│   ├── 2009_000366.jpg
│   ├── 2009_000367.jpg
│   ...
│
└── SegmentationClass
    ├── 2009_000347.png
    ├── 2009_000350.png
    ├── 2009_000351.png
    ├── 2009_000354.png
    ├── 2009_000356.png
    ├── 2009_000366.png
    ├── 2009_000367.png
    ...
```
Where,
* JPEGImages

    Contains all the images with JPEG format
* SegmentationClass

    Contains all the segmentation masks with PNG format
* ImageSets

    Contains the list files to split sets of images, ex. **train.txt** and **val.txt**

The list files have the format as below example, the **train.txt**
```
2009_000347
2009_000350
2009_000351
2009_000354
...
```
In each line, a filename without the extension.

### Convert Dataset Into Training Format

To feed our dataset for training model, we have to convert dataset into compatible formats, first. Here we assume that dataset have prepared, as previous section mentioned. That is, it has images, segmentation masks and list files under folders **JPEGImages**, **SegmentationClass** and **ImageSets** respectively.

In general, it only needs 2 steps as below
1. Remove color map from segmentation masks
2. Pack images and masks into tfrecords

To achieve the dataset converting task, we need to activate **‘Segmentation’** working environment by running
```Bash
$ source Segmentation/bin/activate.sh
```
After activating, it provides the required tools under **Segmentation/bin**
* remove_gt_colormap.py
* convert2tfrecord.py


#### Remove color map
The prepared mask images is a **8 bit color depth PNG files**, which have RGB values for each pixel. However, the labels fed into training only need a class number, in range **[0, num_classes)**, for each pixel. Thus, we have to convert mask images into raw PNG images contain only class number by removing their color map.

To remove color map for each mask imagas, we could use a Python script **remove_gt_colormap.py**, as below
```Bash
$ python ${TOP}/${WORKENV}/bin/remove_gt_colormap.py \
    --original_gt_folder=SegmentationClass             \
    --output_dir=SegmentationClassRaw
```
The **SegmentationClass** is the folder contains all the mask PNG images as above mentioned, and the **SegmentationClassRaw** is the output folder where the **raw mask PNG images** will be stored into.

The raw mask PNG images is what we want to pack into tfrecord format and feed into training.

#### Pack into tfrecord
TensorFlow supports feeding compact data format into training, which is called **tfrecord**. It packs training data and its labels into binary files.

We could use a Python script **Segmentation/bin/convert2tfrecord.py** to achieve this as below
```Bash
$ python ${TOP}/Segmentation/bin/convert2tfrecord.py     \
    --image_folder=JPEGImages                            \
    --list_folder=ImageSets                              \
    --output_dir=tfrecord                                \
    --semantic_segmentation_folder=SegmentationClassRaw  \
    --num_shards=20                                      \
    --image_format="jpg"
```
Where the arguments,
* --image_folder=JPEGImages

    Gives the folders where the original images are located.
* --image_format="jpg"

    The format of images. We use JPEG image format.
* --list_folder=ImageSets

    This folder contains the list files to split sets of images. The convert tool will generate a set of tfrecords for each list file. For example, a **train.txt** list file will generate a tfrecord set with file names **train-***.
* --semantic_segmentation_folder=SegmentationClassRaw

    Gives the foler that contains all the raw mask images.
* --output_dir=tfrecord

    Gives the output folder of generated tfrecord files, ex. tfrecord. After converting, this folder will contains files with filename pattern as below
    ```
    train-00000-of-00020.tfrecord
    train-00001-of-00020.tfrecord
    train-00002-of-00020.tfrecord
    ...
    ```
    where the **‘train’** string in filename pattern is from the filename of list file **‘train.txt’**, and the 00000-of-00020 means it’s the first part of whole dataset which is splitted into 20 parts in total, and so on.
* --num_shards=20

    This gives the number of shards the whole dataset will be splitted into.

The packed tfrecords are the only required data we need to feed into training.


## Prepare COCO Dataset

We use [COCO dataset](http://cocodataset.org) (2017) as our main dataset to train.

Here we show how to prepare COCO 2017 dataset that has
* Image which has only person on it
* Its corresponding segmentation mask

### Prerequisite
We exploit [COCO API](https://github.com/cocodataset/cocoapi) to access and process the COCO dataset to acquire necessary information for us.

#### How to install COCO API?
1. Download COCO API
    ```Bash
    $ git clone https://github.com/cocodataset/cocoapi.git
    ```
2. Change to **PythonAPI** folder
    ```Bash
    $ cd cocoapi/PythonAPI
    ```
3. Modify the Makefile as below, to install module into your local path
    ```Bash
    python setup.py build_ext install --user
    ```
4. Install COCO API Python module
    ```Bash
    $ make install
    ```

#### Prepare the annotation files for COCO 2017 dataset
1. Download annotation files from
    ```Bash
    $ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    ```
2. Unzip it
    ```Bash
    $ unzip annotations_trainval2017.zip
    ```

### Download Dataset through Python script cocoDownload.py
We use the ***cocoDownload.py*** tool in our model code base to download dataset.

The script tool is at **Segmentation/bin/cocoDownload.py**.

The steps is as below
* Modify the related pathes in cocoDownload.py script
    * The path for annotation file
        ```
        annFile = 'annotations/instances_train2017.json'
        ```
    * The output path for JPEG image
        ```
        JPEGImagesPath = 'COCO17_Person/JPEGImages'
        ```
    * The output path for mask PNG image
        ```
        SegmentationClassPath = 'COCO17_Person/SegmentationClass'
        ```
* Start to download by runing script
    ```Bash
    $ python cocoDownload.py
    ```

Note that the tool will output information about each processed data as below format
```
# filename,height,width,count,ratio
000000381472,640,480,3,1.8645833333333335
000000250401,640,480,1,4.239583333333333
000000381475,612,612,1,12.689029860310136
…
```
Where,
* **count** is the number of person in image
* **ratio** is the ratio of pixel contains person instance

### Data Statistics
For COCO 2017 dataset with only person, it has
* Training set 64115
* Validation set 2693

Below gives the ratio statistics about the training set

![](/img/coco17_person.png)


## Model Training SOP
For now, we have dataset in **tfrecord** format, and could start to train.
The detail training SOP could refer to <a href='sampleTrain/README.md'>Training sample</a>.