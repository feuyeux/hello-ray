# Hello Ray

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
