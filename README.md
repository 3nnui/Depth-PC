# Depth-PC: A Visual Servo Framework Integrated with Cross-Modality Fusion for Sim2Real Transfer

### Getting Started
``` bash
git clone https://github.com/3nnui/Depth-PC.git
```

### Dependencies

``` bash
pip install -r requirements.txt
```

### Training

``` bash
python train.py --batch-size 64 --epochs 50 --init-lr 1e-3 --weight-decay 1e-4 --device cuda:0 --save --load path/to/checkpoint.pth
```

### Tests
``` bash
python -m depth_pc.benchmark.tests
```


## Acknowledgments

Thanks for their brilliant contributions! Here are the codebases we built upon.
 * [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric)
 * [CNS](https://github.com/hhcaz/CNS)
 * [DepthAnythingv2](https://github.com/DepthAnything/Depth-Anything-V2/tree/c86af37112102f128f3db5ff190659d56b4305e2)