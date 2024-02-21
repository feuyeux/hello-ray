from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym
from ray import serve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from starlette.requests import Request
import requests
from ray import train, tune
from ray import train
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Dict
import numpy as np
import ray

print("#### 1 Ray Data ####")

# Create datasets from on-disk files, Python objects, and cloud storage like S3.
# ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")
# https://github.com/ray-project/air-sample-data/blob/main/iris.csv
ds = ray.data.read_csv("local:///tmp/iris/original.csv")

# Apply functions to transform data. Ray Data executes transformations in parallel.


def compute_area(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    length = batch["petal length (cm)"]
    width = batch["petal width (cm)"]
    batch["petal area (cm^2)"] = length * width
    return batch


transformed_ds = ds.map_batches(compute_area)

print("Iterate over batches of data.")
for batch in transformed_ds.iter_batches(batch_size=4):
    print(batch)

# Save dataset contents to on-disk files or cloud storage.
transformed_ds.write_parquet("local:///tmp/iris/")

print("#### 2 Ray Train ####")


def get_dataset():
    return datasets.FashionMNIST(
        root="/tmp/data",
        train=True,
        download=True,
        transform=ToTensor(),
    )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, inputs):
        inputs = self.flatten(inputs)
        logits = self.linear_relu_stack(inputs)
        return logits


print("single-worker PyTorch training function.")


def train_func():
    num_epochs = 3
    batch_size = 64

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = NeuralNetwork()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")


train_func()

print("a distributed multi-worker training function.")


def train_func_distributed():
    num_epochs = 3
    batch_size = 64

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader = train.torch.prepare_data_loader(dataloader)

    model = NeuralNetwork()
    model = train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")


# For GPU Training, set `use_gpu` to True.
use_gpu = False

trainer = TorchTrainer(
    train_func_distributed,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=use_gpu)
)

results = trainer.fit()
print(results)


print("#### 3 Ray Tune ####")


def objective(config):  # ①
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


search_space = {  # ②
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

tuner = tune.Tuner(objective, param_space=search_space)  # ③

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)

print("#### 4 Ray Serve ####")


# Train model.
iris_dataset = load_iris()
model = GradientBoostingClassifier()
model.fit(iris_dataset["data"], iris_dataset["target"])


@serve.deployment
class BoostingModel:
    def __init__(self, model):
        self.model = model
        self.label_list = iris_dataset["target_names"].tolist()

    async def __call__(self, request: Request) -> Dict:
        payload = (await request.json())["vector"]
        print(f"Received http request with data {payload}")

        prediction = self.model.predict([payload])[0]
        human_name = self.label_list[prediction]
        return {"result": human_name}


# Deploy model.
serve.run(BoostingModel.bind(model), route_prefix="/iris")

# Query it!
sample_request_input = {"vector": [1.2, 1.0, 1.1, 0.9]}
response = requests.get(
    "http://localhost:8000/iris", json=sample_request_input)
print("response", response.text)

print("#### 5 Ray RLlib ####")


# Define your problem using python and Farama-Foundation's gymnasium API:
class SimpleCorridor(gym.Env):
    """Corridor in which an agent must learn to move right to reach the exit.

    ---------------------
    | S | 1 | 2 | 3 | G |   S=start; G=goal; corridor_length=5
    ---------------------

    Possible actions to chose from are: 0=left; 1=right
    Observations are floats indicating the current field index, e.g. 0.0 for
    starting position, 1.0 for the field next to the starting position, etc..
    Rewards are -0.1 for all steps, except when reaching the goal (+1.0).
    """

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)  # left and right
        self.observation_space = gym.spaces.Box(0.0, self.end_pos, shape=(1,))

    def reset(self, *, seed=None, options=None):
        """Resets the episode.

        Returns:
           Initial observation of the new episode and an info dict.
        """
        self.cur_pos = 0
        # Return initial observation.
        return [self.cur_pos], {}

    def step(self, action):
        """Takes a single step in the episode given `action`.

        Returns:
            New observation, reward, terminated-flag, truncated-flag, info-dict (empty).
        """
        # Walk left.
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        # Walk right.
        elif action == 1:
            self.cur_pos += 1
        # Set `terminated` flag when end of corridor (goal) reached.
        terminated = self.cur_pos >= self.end_pos
        truncated = False
        # +1 when goal reached, otherwise -1.
        reward = 1.0 if terminated else -0.1
        return [self.cur_pos], reward, terminated, truncated, {}


# Create an RLlib Algorithm instance from a PPOConfig object.
config = (
    PPOConfig().environment(
        # Env class to use (here: our gym.Env sub-class from above).
        env=SimpleCorridor,
        # Config dict to be passed to our custom env's constructor.
        # Use corridor with 20 fields (including S and G).
        env_config={"corridor_length": 28},
    )
    # Parallelize environment rollouts.
    .rollouts(num_rollout_workers=3)
)
# Construct the actual (PPO) algorithm object from the config.
algo = config.build()

# Train for n iterations and report results (mean episode rewards).
# Since we have to move at least 19 times in the env to reach the goal and
# each move gives us -0.1 reward (except the last move at the end: +1.0),
# we can expect to reach an optimal episode reward of -0.1*18 + 1.0 = -0.8
for i in range(5):
    results = algo.train()
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")

# Perform inference (action computations) based on given env observations.
# Note that we are using a slightly different env here (len 10 instead of 20),
# however, this should still work as the agent has (hopefully) learned
# to "just always walk right!"
env = SimpleCorridor({"corridor_length": 10})
# Get the initial observation (should be: [0.0] for the starting position).
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0
# Play one episode.
while not terminated and not truncated:
    # Compute a single action, given the current observation
    # from the environment.
    action = algo.compute_single_action(obs)
    # Apply the computed action in the environment.
    obs, reward, terminated, truncated, info = env.step(action)
    # Sum up rewards for reporting purposes.
    total_reward += reward
# Report results.
print(f"Played 1 episode; total-reward={total_reward}")
