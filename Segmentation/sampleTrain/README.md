Sample Training
===============

Here we show you a detail training SOP by using a simple VOC 2012 dataset.

You could download **'VOC2012_Person'** dataset and decompress it as below
```Bash
$ wget --no-check-certificate   \
        "https://drive.google.com/uc?export=download&id=1kRat4AROmm8St8956Os4kAlbJnH9_UaJ" \
        -O VOC2012_Person.tar.bz2

$ tar -jx -f VOC2012_Person.tar.bz2
```
And follow the <a href='../README.md'>Segmentation Model Training</a> to generate the **tfrecord**.

***

## Prerequisite
To training segmentation model **(DeeplabV3+)** model, a few steps need to prepare
1. Activate **'Segmentation'** working environment

    ```Bash
    $ source Segmentation/bin/activate.sh
    ```
2. Prepare training data, the tfrecords
3. Configure the training setting in a **JSON** configure file

    To train a model, it has many configuration, for example, **learning rate**, **number of steps** and so on. We use a single configure file in **JSON** format to facilitate the training flow.

    You could look into the sample configure file under folder **Segmentation/sampleTrain**, the **voc2012_Person.json** file.

    Below gives the configure example
    ```JSON
    {
        "train_split" : "train",
        "model_variant" : "mobilenet_v2",
        "train_logdir" : "eval",
        "dataset_dir" : "VOC2012_Person/tfrecord",
        "output_stride" : 16,
        "train_crop_size" : [513, 513],
        "train_batch_size" : 16,
        "depth_multiplier" : 0.5,
        "min_scale_factor" : 1.0,
        "max_scale_factor" : 1.0,
        "fine_tune_batch_norm" : true,
        "training_number_of_steps" : 100,

        "DatasetDescriptor" : {
            "name" : "voc12_person",
            "splits_to_sizes" : {
                "trainval" : 887,
                "train" : 800,
                "val" : 87
            },

            "num_classes" : 2,
            "ignore_label" : 255
        }
    } 
    ```

## How to train?
Once the previous prerequisites are all satisfied, you could start to train.
The training script, **train.py**, is under folder **Segmentation/bin** after you activate the **'Segmentation'** working environment, you could start to train just launch command as below example
```Bash
$ python ${TOP}/${WORKENV}/bin/train.py --logtostderr \
    --config=voc2012_Person.json 2>&1 | tee train.log
```
The argument **--config=voc2012_Person.json** gives the configuration file, **voc2012_Person.json**, for training. And the reminder command ```| tee train.log``` just means to save all the output into train.log file while under training.


## How to evaluate training result
To evaluate the training result **(The mIOU)**, you also need a **evaluation JSON configuration file**, for example, **voc2012_Person_eval.json**, as below
```JSON
{
    "eval_split" : "val",
    "model_variant" : "mobilenet_v2",
    "checkpoint_dir" : "./eval",
    "dataset_dir" : "VOC2012_Person/tfrecord",
    "eval_crop_size" : [513, 513],
    "depth_multiplier" : 0.5,
    "max_number_of_evaluations": 1,
    "eval_logdir" : ".",

    "DatasetDescriptor" : {
        "name" : "voc12_person",
        "splits_to_sizes" : {
            "trainval" : 887,
            "train" : 800,
            "val" : 87
        },

        "num_classes" : 2,
        "ignore_label" : 255
    }
} 
```
Where **checkpoint_dir** specified the checkpoint folder for the training result, which is the **train_logdir** in training configuration.

Instead of **train** set, it uses **val** set, specified at **eval_split**, to run evaluation as below
```Bash
$ python ${TOP}/${WORKENV}/bin/eval.py --config=voc2012_Person_eval.json
```


## The training configuration
The training configuration could be set in a **JSON** file and feed into training.
Here list few important settings.
* dataset_dir

    This set the dataset folder, which contains the tfrecords.
* train_split

    This specify which set of dataset you would like to feed into train.
    The dataset folder may contain different set of tfrecords as below
    ```
    train-00000-of-00010.tfrecord
    ...

    val-00000-of-00010.tfrecord
    ...
    ```
    There are 2 set of tfrecords, **train** and **val**. You could specify one of them into training, for example, the **train** set as below
    ```
    "train_split" : "train"
    ```
* train_logdir

    The output folder of training log file. When training, there are a number of file will be logged, for example, the **checkpoints**.

* DatasetDescriptor

    This is mandatory to give a description of your dataset, provided at **dataset_dir** folder
    ```JSON
    "DatasetDescriptor" : {
        "name" : "voc12_person",
        "splits_to_sizes" : {
            "trainval" : 887,
            "train" : 800,
            "val" : 87
        },

        "num_classes" : 2,
        "ignore_label" : 255
    }
    ```
    This descriptor contains
    * name

        Gives a name of our dataset. It’s not matter.
    * splits_to_sizes

        This lists the number of training data for each set.
    * num_classes

        Gives the number of classes in this dataset.
    * ignore_label

        This specifies which class number you would like to ignore.



## Export to inference graph
After training, we have the result checkpoints saved at **train_logdir**. We could export to frozen graph for inferencing.

### Export to inference PB
```Bash
 $ python ${TOP}/Segmentation/bin/export_model.py   \
        --logtostderr                               \
        --model_variant="mobilenet_v2"              \
        --export_path=./frozen_inference_graph.pb   \
        --num_classes=2                             \
        --crop_size=513                             \
        --crop_size=513                             \
        --inference_scales=1.0                      \
        --depth_multiplier=0.5                      \
        --checkpoint_path=eval/model.ckpt-100
```
Where,
* --checkpoint_path=eval/model.ckpt-100

    Frozen from the checkpoint **eval/model.ckpt-100** to inference graph.
* --export_path=./frozen_inference_graph.pb

    The output PB file is **frozen_inference_graph.pb**.
