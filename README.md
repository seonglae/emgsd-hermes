# EMGSD Hermes
This project explores bias mitigation in GPT2-EMGSD, leveraging correlation analysis for stereotype deduction and activation manipulation, highlighting the potential of an alternative to traditional fine-tuning. Additionally, it demonstrates the feasibility of inducing bias in vanilla GPT2 through activation engineering.

<img width="750" alt="image" src="image/emgsd.png">

## Fast Demo
```bash
# Install python 3.10 which is required by SAE-Lens
⁠⁠⁠git clone ⁠ https://github.com/seonglae/emgsd-hermes && cd emgsd-hermes
p⁠ip install torch colorama sae-lens transformers
python compare.py
```

## Main Pipeline
TBA
### 1. Fine-tuning SAE with EMGSD dataset 
```bash
python empsd.py
```
### 2. Extract features using correlation
```bash
python search_category.py
python search_stereo.py
# replace emgsd/*.json files
python draw_corr.py
```
<img width="750" alt="image" src="image/stereotype_corr.png">

or if you want to calculate mutual information
```
python mi_stereo.py
```

### 3. Compute ratio of stereotyped text in generation
```bash
python compare_all.py
```


<img width="750" alt="image" src="image/stereotype_ratios_per_stereotype.png">
<img width="500" alt="image" src="image/overall_stereotype_ratios_table.png">



## Loss Graph of fine-tuning SAE
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/20ba51ae-7f58-4f11-af5c-5a9eaa2cd0da">
