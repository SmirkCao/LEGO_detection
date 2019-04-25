# LEGO Detection

Just another LEGO detection, but it's fun.

## Pipeline

## Demo

### Segmentation

```bash
python solution/demo.py -t seg_ada -c 1
```

### Color Feature

```bash
python solution/demo.py -t color -c 1
```

### Feature Engineering

```bash
cd solution
python demo.py -t feas
```
## Dataset

```bash
# 1. Run following code for different part
python solution/utils.py -l 6091 -c red
# 2. Change the pose of part to collect more data 
# and press any key to quit collect process.
```

## How to use

```bash
cd solution
source run.sh
```
