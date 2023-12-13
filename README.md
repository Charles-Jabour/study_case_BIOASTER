# study_case_BIOASTER
Study case on predicting class related to guts microbiome vibrational spectra

1) In a virtual environment, install required libraries
```
pip install -r requirements.txt
```
2) For statistical analysis, run:

```
python main.py --exp_type stats
```

3) For nested cross-validation analysis, run:

```
python main.py --exp_type ncv --exp_name ncv_exp
```

4) For feature selection experiment, run:

```
python main.py --exp_type fs --exp_name fs_exp
```

Check `options.py` for additional inputs
