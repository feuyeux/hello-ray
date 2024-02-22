# Hello Ray

https://docs.ray.io/en/latest/index.html

[Ray](https://github.com/ray-project/ray) is an open-source unified framework for scaling AI and Python applications. It provides the compute layer for parallel processing so that you don’t need to be a distributed systems expert.

## Ray framework

Ray’s unified compute framework consists of three layers:

1. **Ray AI Libraries**–An open-source, Python, domain-specific set of libraries that equip **ML engineers**, **data scientists**, and **researchers** with a scalable and unified toolkit for ML applications.
   1. [Data](https://docs.ray.io/en/latest/data/dataset.html): Scalable Datasets for ML
   2. [Train](https://docs.ray.io/en/latest/train/train.html): Distributed Training
   3. [Tune](https://docs.ray.io/en/latest/tune/index.html): Scalable Hyperparameter Tuning
   4. [RLlib](https://docs.ray.io/en/latest/rllib/index.html): Scalable Reinforcement Learning
   5. [Serve](https://docs.ray.io/en/latest/serve/index.html): Scalable and Programmable Serving

2. **Ray Core**–An open-source, Python, general purpose, distributed computing library that enables ML engineers and Python developers to scale Python applications and accelerate machine learning workloads.
   1. [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html): Stateless functions executed in the cluster.
   2. [Actors](https://docs.ray.io/en/latest/ray-core/actors.html): Stateful worker processes created in the cluster.
   3. [Objects](https://docs.ray.io/en/latest/ray-core/objects.html): Immutable values accessible across the cluster.

3. **Ray Clusters**–A set of worker nodes connected to a common Ray head node. Ray clusters can be fixed-size, or they can autoscale up and down according to the resources requested by applications running on the cluster.

![ray-air](https://docs.ray.io/en/latest/_images/ray-air.svg) **Ray AI Runtime(AIR)** 

## Setup

<https://docs.ray.io/en/latest/ray-overview/getting-started.html>

```sh
python3 -m venv ray_env
```

```sh
export http_proxy=http://127.0.0.1:59503
# On Linux and macOS
source ray_env/bin/activate
# On Windows
source ray_env/Scripts/activate
```

## 0 Ray Core

```sh
pip install --upgrade pip
pip install -U "ray"
```

## 1 Ray Data

```sh
pip install -U "ray[data]"

# workaround for https://github.com/ray-project/ray/issues/42842
pip uninstall pandas
pip install pandas==
pip install -Iv pandas==2.1.4
```

## 2 Ray Train

```sh
pip install -U "ray[train]" torch torchvision
```

## 3 Ray Tune

```sh
pip install -U "ray[tune]"
```

## 4 Ray Serve

```sh
pip install -U "ray[serve]" scikit-learn
```

## 5 Ray RLlib

```sh
pip install -U "ray[rllib]"
```

## RUN hello

```sh
# 0
python hell_ray_core.py
# 1- 5
python hello_ray_ai_libraries.py
```
## With langchain

<https://python.langchain.com/docs/integrations/providers/ray_serve>
