<!-- markdownlint-disable MD033 MD045 -->

# Hello Ray

[Ray](https://github.com/ray-project/ray) is an open-source unified framework for scaling AI and Python applications. It provides the compute layer for parallel processing so that you don’t need to be a distributed systems expert.

- Youtube channel: [@anyscale](https://www.youtube.com/@anyscale)
- Twitter: [@raydistributed](https://twitter.com/raydistributed)
- Github: <https://github.com/ray-project/ray>

## Ray framework

<https://docs.ray.io/en/latest/index.html>

Ray’s unified compute framework consists of three layers:

1. <img src="https://docs.ray.io/en/latest/_static/img/AIR.png" style="width:10px" />**Ray AI Libraries**–An open-source, Python, domain-specific set of libraries that equip **ML engineers**, **data scientists**, and **researchers** with a scalable and unified toolkit for ML applications.
   1. [Data](https://docs.ray.io/en/latest/data/dataset.html): Scalable Datasets for ML
      1. Loading data: create datasets from on-disk files, Python objects, and cloud storage services like S3.
      2. Transforming data: apply user-defined functions (UDFs) to transform(in parallel) datasets.
      3. Consuming data: pass datasets to Ray Tasks or Actors, and access records.
      4. Saving data: save dataset contents to local or remote filesystems(parquet).
   2. [Train](https://docs.ray.io/en/latest/train/train.html): Distributed Training
      1. Training function: A Python function that contains your model training logic.
      2. Worker: A process that runs the training function.
      3. Scaling configuration: A configuration of the number of workers and compute resources (for example, CPUs or GPUs).
      4. Trainer: A Python class that ties together the training function, workers, and scaling configuration to execute a distributed training job.
         ![ray train overview](https://docs.ray.io/en/latest/_images/overview.png)
   3. [Tune](https://docs.ray.io/en/latest/tune/index.html): Scalable Hyperparameter Tuning
   4. [RLlib](https://docs.ray.io/en/latest/rllib/index.html): Scalable Reinforcement Learning
   5. [Serve](https://docs.ray.io/en/latest/serve/index.html): Scalable and Programmable Serving
2. <img src="https://docs.ray.io/en/latest/_static/img/Core.png" style="width:10px" />**Ray Core**–An open-source, Python, general purpose, distributed computing library that enables ML engineers and Python developers to scale Python applications and accelerate machine learning workloads.
   1. [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html): Stateless functions executed in the cluster.
   2. [Actors](https://docs.ray.io/en/latest/ray-core/actors.html): Stateful worker processes created in the cluster.
   3. [Objects](https://docs.ray.io/en/latest/ray-core/objects.html): Immutable values accessible across the cluster.
      ![application-logging](https://docs.ray.io/en/latest/_images/application-logging.png)
3. <img src="https://docs.ray.io/en/latest/_static/img/rayclusters.png" style="width:10px" />**Ray Clusters**–A set of worker nodes connected to a common Ray head node. Ray clusters can be fixed-size, or they can autoscale up and down according to the resources requested by applications running on the cluster.

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
