# Language ICL models

### Training

```
python main.py --train [config name]
```

### Evaluation

```
python main.py --eval [config name]
```

To load a specific checkpoint, use:

```
python main.py --eval [config name] --checkpoint [checkpoint name]
```

where [checkpoint name] is either "epoch_x" or "best"

### Generation

```
python main.py --eval [config name] --generate
```

Note that this uses nucleus (top-p) sampling

### Survey

To quickly see what models have been trained (and for how many epochs), simply use:

```
python survey.py
```

TEST 1 - Remove diagonal masking
Test 2 - Remove diagonal + [:, :-1, :]
Test 3 - No rotary
