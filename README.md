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

## Ray Core

```sh
pip install --upgrade pip
pip install -U "ray"
```

## Ray Data

```sh
pip install -U "ray[data]"

# workaround for https://github.com/ray-project/ray/issues/42842
pip uninstall pandas
pip install pandas==
pip install -Iv pandas==2.1.4
```

## Ray Train

```sh
pip install -U "ray[train]" torch torchvision
```
