# FreeShap
Efficient fine-tuning-free approximation of the Shapley value (FreeShap) for instance attribution based on the  neural tangent kernel. 

# Prepare the conda environment

```
conda create --name freeshap python=3.10
conda activate freeshap
pip install -r requirements.txt
```

# Quick start: Test Prediction Explanation
A beginner-friendly Jupyter notebook titled `vinfo/SST2_explanation.ipynb` illustrates the computation of FreeShap and the explanation of test predictions using training examples.

# Code structure
- /configs
The example configuration file is put under `configs/dshap/sst2/ntk_prompt.yaml`
yaml files start with ntk in the name are used to specify the hyperparameters for the ntk. 
yaml files start with finetune in the name are used to specify the hyperparameters for the prompt-based fine-tuning.
The code structure is based on public repo [cords](https://github.com/decile-team/cords). 
- /vinfo
  - /dvutils: Data Shapley code. Core file is `Data_Shaley.py`.
  - /entks: NTK kernel building and regression.  The ntk module it built on the public [empirical-ntks](https://github.com/aw31/empirical-ntks) repo.
    - `nlpmodels.py`: NTK model classes.
    - `ntk.py`: NTK kernel building.
    - `ntk_regression.py`: Kernel regression.
  - `dataset.py`: construction of dataset classes.
  - `probe.py`: computation of utility function and prompt-based fine-tuning model classes. 

# How to run the code

```
python vinfo/ntk.py --yaml_path={YAML_PATH} --dataset_name={DATASET} --file_path={PATH}

```
For instance, to run on SST-2 dataset with 5000 points: 
```
python vinfo/ntk.py --yaml_path="configs/dshap/sst2/ntk_prompt.yaml" --dataset_name=sst2 --file_path={PATH}

```
You may also run `ntk_shapley.sh` file to run the code.
```
bash ntk_shapley.sh "sst2" "0" "5000" "True" "{PATH}"
```

The running result will be stored at `{PATH}` folder.

We have also provided a kernel in [Drive](https://drive.google.com/drive/folders/1BCqdtBO_jderYfjCEYVQOo_ADaKxvzlo?usp=sharing). Can save the kernel in the `{PATH}` to play with explanations. 