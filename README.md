# Sentiment Analysis with BERT

## Basic structure
- `model/` folder: all the models and its components.
- `setup/` folder: the python and cuda requirements.
- `utils/` folder: the data-handling and evaluation parts.

## Datasets
The datasets will be automatically downloaded when running the code.
There are 3 datasets available:
- Yelp13
- Yelp14
- IMDB

## Setup
```
> sh setup/install_cuda.sh
> pip3 install -r setup/requirements.txt
```

## How to run the models
To run the models do:
```
> python3 -m model.train.<train_model_script> <parameters>
```
Example:
```
> python3 -m model.train.train_vanilla_bert --epochs 12 --learning_rate 0.001 --gradient_accumulation 8 --output_dir ./training_output
```
Available paramaters can be looked up in `model/train/train.py`.

## Outputs
Outputs are saved under the followinf file structure:
`<model_name>/<dataset_name>/`

The followinf files are saved:
- `pytorch_model.bin`: the model
- `args.json`: arguments of the model
- `results.json`: training results

