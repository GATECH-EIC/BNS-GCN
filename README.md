# BNS-GCN: Efficient Full-Graph Training of Graph Convolutional Networks with Partition-Parallelism and Random Boundary Node Sampling

Cheng Wan<sup>\*</sup> (Rice University), Youjie Li<sup>\*</sup> (UIUC), Ang Li (PNNL), Nam Sung Kim (UIUC), Yingyan Lin (Rice University)

(<sup>\*</sup>Equal contribution)

Accepted at MLSys 2022 [[Paper](https://arxiv.org/abs/2203.10983) | [Video](https://youtu.be/kzI0ksASFQY) | [Slide](https://mlsys.org/media/mlsys-2022/Slides/2178.pdf) | [Docker](https://hub.docker.com/r/cheng1016/bns-gcn) | [Sibling](https://github.com/RICE-EIC/PipeGCN)]



## Directory Structure

```
|-- checkpoint   # model checkpoints
|-- dataset
|-- helper       # auxiliary codes
|   `-- timer
|-- module       # PyTorch modules
|-- partitions   # partitions of input graphs
|-- results      # experiment outputs
`-- scripts      # example scripts
```

Note that `./checkpoint/`, `./dataset/`, `./partitions/` and `./results/` are empty folders at the beginning and will be created when BNS-GCN is launched.

## Setup

### Environment

#### Hardware Dependencies

- A X86-CPU machine with at least 120 GB host memory 
- At least five Nvidia GPUs (at least 11 GB each)

#### Software Dependencies

- Ubuntu 18.04
- Python 3.8
- CUDA 11.3
- [PyTorch 1.12.0](https://github.com/pytorch/pytorch)
- [DGL 0.9.1](https://github.com/dmlc/dgl)
- [OGB 1.3.5](https://ogb.stanford.edu/docs/home/)

### Installation

#### Option 1: Run with Docker

We have prepared a [Docker package](https://hub.docker.com/r/cheng1016/bns-gcn) for BNS-GCN.

```bash
docker pull cheng1016/bns-gcn
docker run --gpus all -it cheng1016/bns-gcn
```

#### Option 2: Install with Pip

Running the following command will install all prerequisites from pip.

```bash
pip install -r requirements.txt
```

#### Option 3: Do it Yourself

If the above options fail to run BNS-GCN, please follow the official guides ([[1]](https://pytorch.org/get-started/locally/), [[2]](https://www.dgl.ai/pages/start.html), [[3]](https://ogb.stanford.edu/docs/home/)) to install PyTorch, DGL and OGB.

### Datasets

We use Reddit, ogbn-products, Yelp and ogbn-papers100M for evaluating BNS-GCN. All datasets are supposed to be stored in `./dataset/` by default.


## Basic Usage

### Core Training Options

- `--dataset`: the dataset you want to use
- `--model`: the GCN model (only GCN, GraphSAGE and GAT are supported at this moment)
- `--lr`: learning rate
- `--sampling-rate`: the sampling rate of BNS-GCN
- `--n-epochs`: the number of training epochs
- `--n-partitions`: the number of partitions
- `--n-hidden`: the number of hidden units
- `--n-layers`: the number of GCN layers
- `--partition-method`: the method for graph partition ('metis' or 'random')
- `--port`: the network port for communication
- `--no-eval`: disable evaluation process

### Run Example Scripts

Simply running `scripts/reddit.sh`, `scripts/ogbn-products.sh` and `scripts/yelp.sh` can reproduce BNS-GCN under the default settings. For example, after running `bash scripts/reddit.sh`, you will get the output like this

```
...
Process 000 | Epoch 02999 | Time(s) 0.3578 | Comm(s) 0.2267 | Reduce(s) 0.0108 | Loss 0.0716
Process 001 | Epoch 02999 | Time(s) 0.3600 | Comm(s) 0.2314 | Reduce(s) 0.0136 | Loss 0.0867
(rank 1) memory stats: current 562.96MB, peak 1997.89MB, reserved 2320.00MB
(rank 0) memory stats: current 557.01MB, peak 2087.31MB, reserved 2296.00MB
Epoch 02999 | Accuracy 96.55%
model saved
Max Validation Accuracy 96.68%
Test Result | Accuracy 97.21%
```

### Run Full Experiments

If you want to reproduce core experiments of our paper (e.g., accuracy in Table 4, throughput in Figure 4, time breakdown in Figure 5, peak memory in Figure 6), please run `scripts/reddit_full.sh`,  `scripts/ogbn-products_full.sh` or  `scripts/yelp_full.sh`, and the outputs will be saved to `./results/` directory. Note that the throughput of these experiments will be significantly slower than the results in our paper because the training is performed along with validation.

### Run Customized Settings

You may adjust `--n-partitions` and `--sampling-rate` to reproduce the results of BNS-GCN under other settings. To verify the exact throughput or time breakdown of BNS-GCN, please add `--no-eval` argument to skip the evaluation step. You may also use the argument `--partition-method=random` to explore the performance of BNS-GCN with random partition.

### Run with Multiple Compute Nodes

Our code base also supports distributed GCN training with multiple compute nodes. To achieve this, you should specify `--master-addr`, `--node-rank` and `--parts-per-node` for each compute node. An example is provided in `scripts/reddit_multi_node.sh` where we train the Reddit graph over 4 compute nodes, each of which contains 10 GPUs, with 40 partitions in total. You should run the command on each node and specify the corresponding node rank. **Please turn on `--fix-seed` argument** so that all nodes initialize the same model weights.

If the compute nodes do not share storage, you should partition the graph in a single device first and manually distribute the partitions to other compute nodes. When run the training script, please enable `--skip-partition` argument.



## Citation

```
@article{wan2022bns,
  title={{BNS-GCN}: Efficient Full-graph Training of Graph Convolutional Networks with Partition-parallelism and Random Boundary Node Sampling},
  author={Wan, Cheng and Li, Youjie and Li, Ang and Kim, Nam Sung and Lin, Yingyan},
  journal={Proceedings of Machine Learning and Systems},
  volume={4},
  pages={673--693},
  year={2022}
}
```



## License

Copyright (c) 2022 GaTech-EIC. All rights reserved.

Licensed under the [MIT](https://github.com/RICE-EIC/BNS-GCN/blob/master/LICENSE) license.
