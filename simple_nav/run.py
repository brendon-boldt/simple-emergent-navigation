from typing import Any, Tuple, List, Iterator, Union, Dict, Optional
import argparse
from argparse import Namespace
from pathlib import Path
import pickle as pkl
import shutil
import uuid
import re
import warnings
import json

from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore

from . import env
from .callback import EvalCallback
from . import util
from . import experiment_configs

_cfg = experiment_configs.default_config


def execute_run(base_dir: Path, cfg: argparse.Namespace, idx: int) -> None:
    # Ignore warnings about non-matching batch size
    if "buffer_size" in cfg.cfg_name:
        warnings.filterwarnings("ignore", module="stable_baselines3")

    log_dir = base_dir / f"run-{idx}"
    if (log_dir / "completed").exists():
        return
    elif log_dir.exists():
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    with (log_dir / "config.txt").open("w") as text_fo:
        text_fo.write(str(cfg))
    with (log_dir / "config.pkl").open("wb") as binary_fo:
        pkl.dump(cfg, binary_fo)
    env_kwargs = util.make_env_kwargs(cfg)
    env_eval = DummyVecEnv([lambda: cfg.environment(is_eval=True, **env_kwargs)])
    logging_callback = EvalCallback(
        eval_env=env_eval,
        n_eval_episodes=cfg.eval_episodes_logging,
        eval_freq=cfg.eval_freq,
        writer=writer,
        cfg=cfg,
    )
    model = util.make_model(cfg)
    if model is None:
        raise ValueError("Could not restore model.")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[logging_callback],
    )
    # Create empty file to show the run is completed in case it gets interrupted
    # halfway through.
    (log_dir / "completed").open("w")


def execute_configuration(
    base_dir: Path, cfg: Namespace, name_props: List[str], num_trials: int
) -> Iterator[Any]:
    # name = "_".join(str(getattr(cfg, prop)).replace("/", ",") for prop in name_props)
    formatter = lambda x: f"{x:.3g}" if isinstance(x, float) else str(x)
    name = "_".join(formatter(getattr(cfg, prop)).replace("/", ",") for prop in name_props)
    log_dir = base_dir / name
    return (delayed(execute_run)(log_dir, cfg, i) for i in range(num_trials))


def run_experiments(
    exp_name: str, num_trials: int, n_jobs: int, out_dir=Path("log")
) -> None:
    jobs: List[Tuple] = []
    out_path = out_dir / exp_name
    for config in experiment_configs.CONFIGS[exp_name]:
        final_config = {**vars(_cfg), **config}
        cfg = Namespace(**final_config)
        cfg.cfg_name = exp_name
        jobs.extend(
            execute_configuration(out_path, cfg, list(config.keys()), num_trials)
        )

    if len(jobs) == 1 or n_jobs == 1:
        for j in jobs:
            j[0](*j[1], **j[2])
    else:
        Parallel(n_jobs=n_jobs)(j for j in tqdm(jobs))


def patch_old_configs(cfg: Namespace) -> Namespace:
    """Add recently added parameters to old config objects.

    Example
    ```
    if not hasattr(cfg, "gamma"):
        cfg.gamma = 0.9
    ```

    """
    if not hasattr(cfg, "biased_reward_shaping"):
        cfg.biased_reward_shaping = False
    return cfg


def get_one_hot_vectors(policy: Any) -> np.ndarray:
    _data = []
    bn_size = next(policy.mlp_extractor.post_net.modules())[0].in_features
    for i in range(bn_size):
        x = torch.zeros(bn_size)
        x[i] = 1.0
        with torch.no_grad():
            x = policy.mlp_extractor.post_net(x)
            x = policy.action_net(x)
        _data.append(x.numpy())
    return np.array(_data)


def collect_metrics(
    model_path: Path,
    out_path: Path,
    eval_episodes: int,
) -> Optional[pd.DataFrame]:
    with (model_path.parent / "config.pkl").open("rb") as fo:
        cfg = pkl.load(fo)
    # Ignore warnings about non-matching batch size
    if "buffer_size" in cfg.cfg_name:
        warnings.filterwarnings("ignore", module="stable_baselines3")
    cfg = patch_old_configs(cfg)
    env_kwargs: Dict[str, Any] = {
        **util.make_env_kwargs(cfg),
        "is_eval": True,
    }
    if cfg.environment == env.NavToCenter:
        env_kwargs["world_radius"] = _cfg.eval_world_radius
    environment = DummyVecEnv([lambda: cfg.environment(**env_kwargs)])
    model = util.make_model(cfg)
    if model is None:
        print(f'Could not restore model "{cfg.init_model_path}"')
        return None
    policy = model.policy
    try:
        policy.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(e)
        return None
    vectors = get_one_hot_vectors(model.policy)
    mlp_extractor = policy.mlp_extractor.cpu()
    bottleneck_values = []
    steps_values = []
    successes = 0.0
    n_episodes = 0
    discretize = True

    # The index is passed to the initialization. We start with lo so that that
    # the agent is not intialized in the center.
    # TODO Fix initialization so we don't have to compute lo manually.
    for i in range(eval_episodes):
        n_episodes += 1
        environment.reset()
        if isinstance(environment, env.Navigation):
            environment.fib_disc_init(i, eval_episodes)
        results = util.eval_episode(policy, mlp_extractor, environment, discretize)
        successes += results["total_reward"]
        steps_values.append(results["steps"])
        # bottleneck_values.extend(results["bn_activations"])
        bottleneck_values.append(results["bn_activations"])
    messages = [[x.argmax() for x in y] for y in bottleneck_values]
    # Flatten them
    bottleneck_values = [x for y in bottleneck_values for x in y]
    np_bn_values = np.stack(bottleneck_values)
    entropy = util.get_entropy(np_bn_values)
    sample_id = str(uuid.uuid4())

    contents = {
        "path": str(model_path),
        "uuid": sample_id,
        "steps": np.mean(steps_values),
        "success_rate": successes / n_episodes,
        "entropy": entropy,
        "discretize": discretize,
        "usages": np_bn_values.mean(0).tolist(),
        "vectors": vectors.tolist(),
        "messages":  messages,
        **vars(cfg),
    }
    return pd.DataFrame({k: [v] for k, v in contents.items()})


def expand_paths(
    path_like: Union[str, Path], progression: bool, target_ts: Optional[int]
) -> List[Path]:
    root = Path(path_like)
    if not root.is_dir():
        return []
    contents = {x for x in root.iterdir()}
    names = {x.name for x in contents}
    paths = []
    if progression:
        new_paths = sorted(
            root.glob("model*.pt"),
            key=lambda x: int(str(x).split("-")[-1].split(".")[0]),
        )
        paths.extend(new_paths)
    elif len({"completed", "config.pkl"} & names) == 2:
        if target_ts is None:
            max_ts = 0
            for name in names:
                mo = re.match(r"model-([0-9]+)\.pt", name)
                if mo:
                    ts = int(mo.group(1))
                    if ts > max_ts:
                        max_ts = ts
            target_ts = max_ts
        target_fn = f"model-{target_ts}.pt"
        paths.append(root / target_fn)
    paths.extend(x for c in contents for x in expand_paths(c, progression, target_ts))
    return paths


def aggregate_results(
    path_strs: List[str],
    n_jobs: int,
    progression: bool,
    target_ts: Optional[int],
    df_concat_paths: List[Path],
) -> None:
    dfs_to_concat = [pd.read_csv(p) for p in df_concat_paths]
    out_dir = Path("results") / Path(path_strs[0]).name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    paths = [x for p in path_strs for x in expand_paths(p, progression, target_ts)]
    jobs = [delayed(collect_metrics)(p, out_dir, _cfg.eval_episodes) for p in paths]
    results = [
        r for r in Parallel(n_jobs=n_jobs)(x for x in tqdm(jobs)) if r is not None
    ]
    df_contents = results
    df = pd.concat(df_contents + dfs_to_concat, ignore_index=True)

    # Write messages
    for key in df.index:
        messages = df.loc[key, "messages"]
        prop = "_".join(df.loc[key, "cfg_name"].split("_")[2:])
        value = df.loc[key, "path"].split("/")[-3].split("_")[-1]
        name = f"{prop}_{value}"
        with (out_dir / f"{name}.messages.jsonl").open("w") as fo:
            for line in messages:
                # Convert np.int64 to int
                line = np.array(line).tolist()
                fo.write(json.dumps(line, indent=None))
                fo.write("\n")
    df.drop("messages", axis=1, inplace=True)
    df.to_csv(out_dir / "data.csv", index=False)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("targets", type=str, nargs="+")
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--progression", action="store_true")
    parser.add_argument("--target_timestep", type=int, default=None)
    parser.add_argument("--include_csv", type=str, action="append")
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if args.include_csv is None:
        args.include_csv = []

    if args.command == "eval":
        aggregate_results(
            args.targets,
            args.j,
            args.progression,
            args.target_timestep,
            args.include_csv,
        )
    elif args.command == "run":
        run_experiments(args.targets[0], args.num_trials, args.j)
    else:
        raise ValueError(f"Command '{args.command}' not recognized.")
