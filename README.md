# Installation

Clone repo and install [requirements.txt](requirements.txt) in a Python=3.7 environment.

```python
conda create --name mobilenet python=3.7
conda activate mobilenet
git clone https://github.com/FITI-HCITA/mobilenetv2.git
cd mobilenetv2
pip install -r requirements.txt
```

# Run The Training Program

This training program will automatically export TFLite model after training and perform an inference with the TFLite model.

- Please check your opensource dataset name **--dataset "opensource dataset name"**

    Example of opensource dataset name
    ``--dataset 'horses_or_humans'``

- **Opensource datatset name** can be obtained from [TensorFlow Website](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).

```python
python main.py --dataset 'opensource_dataset_name' --img_size 224 --epochs 500 --batch_size 32
                         
```

- After training, results saved to ``output/mobilenetv2_w05_dataset_name_YYYY-MM-DD``
