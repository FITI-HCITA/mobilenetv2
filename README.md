# Installation

Clone repo and install [requirements.txt](requirements.txt) in a Python=3.7 environment.

```python
conda create --name mobilenet python=3.7
conda activate mobilenet
pip install -r requirements.txt
```

# Run The Training Program

This training program will automatically export TFLite model after training and perform an inference with the TFLite model.

Opensource “datatset name” can be obtained from [TensorFlow Website](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).

```python
python main.py --dataset <opensource dataset name> --img_size 224 --epochs 500 --batch_size 32
                         'horses_or_humans'
```
