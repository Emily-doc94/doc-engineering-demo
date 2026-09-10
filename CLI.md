
```bash
conda activate docs-demo
conda env export > environment.yml
conda env create -f environment.yml
```


```
pip freeze > requirements.txt
pip install -r requirements.txt
```