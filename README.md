# Language ICL models

### Training

```
python main.py --train [config name]
```

### Evaluation

```
python main.py --eval [config name]
```


### Generation

```
python main.py --eval [config name] --generate
```
Note that this uses nucleus (top-p) sampling