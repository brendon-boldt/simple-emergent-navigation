from typing import Any, Tuple, List, Iterator, Union, Dict, Optional
import argparse
from argparse import Namespace
from pathlib import Path
import pickle as pkl
import importlib
import shutil
import uuid

from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
import torch
import numpy as np  # type: ignore
from torch.utils.tensorboard import SummaryWriter
from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore

from . import env
from .callback import LoggingCallback
from . import util
from .default_config import cfg as _cfg


def execute_run(base_dir: Path, cfg: argparse.Namespace, idx: int) -> None:
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
    env_eval = DummyVecEnv([lambda: env.NavToCenter(is_eval=True, **env_kwargs)])
    logging_callback = LoggingCallback(
        eval_env=env_eval,
        n_eval_episodes=cfg.eval_steps,
        eval_freq=cfg.eval_freq,
        writer=writer,
        verbose=0,
        save_all_checkpoints=cfg.save_all_checkpoints,
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
    name = "_".join(str(getattr(cfg, prop)).replace("/", ",") for prop in name_props)
    log_dir = base_dir / name
    return (delayed(execute_run)(log_dir, cfg, i) for i in range(num_trials))


def run_experiments(
    config_paths: List[str], num_trials: int, n_jobs: int, out_dir=Path("log")
) -> None:
    jobs: List[Tuple] = []
    for config_path in config_paths:
        config_name = config_path.split("/")[-1][:-3]
        out_path = out_dir / config_name
        module_name = config_path.rstrip("/").replace("/", ".")[:-3]
        mod: Any = importlib.import_module(module_name)
        for config in mod.generate_configs():
            final_config = {**vars(_cfg), **config}
            cfg = Namespace(**final_config)
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
    eval_steps: int,
) -> Optional[pd.DataFrame]:
    with (model_path.parent / "config.pkl").open("rb") as fo:
        cfg = pkl.load(fo)
    cfg = patch_old_configs(cfg)
    env_kwargs: Dict[str, Any] = {
        **util.make_env_kwargs(cfg),
        "is_eval": True,
        "world_radius": 9.0,
    }
    environment = env.NavToCenter(**env_kwargs)
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
    trajs: List[List] = []
    n_episodes = 0
    n_steps = 0
    discretize = True

    g_rad = environment.goal_radius / environment.world_radius
    lo = int(np.ceil(eval_steps * g_rad ** 2))
    hi = int(np.ceil(eval_steps / (1 - g_rad ** 2)))

    # The index is passed to the initialization. We start with lo so that that
    # the agent is not intialized in the center.
    # TODO Fix initialization so we don't have to compute lo manually.
    for i in range(lo, hi):
        n_episodes += 1
        environment.reset()
        environment.fib_disc_init(i, hi)
        ep_len, bns, success, traj = util.eval_episode(
            policy, mlp_extractor, environment, discretize
        )
        n_steps += ep_len
        successes += success
        steps_values.append(ep_len)
        bottleneck_values.extend(bns)
        trajs.extend(traj)
    np_bn_values = np.stack(bottleneck_values)
    entropy = util.get_entropy(np_bn_values)
    sample_id = str(uuid.uuid4())

    trajectories_dir = out_path / "trajectories"
    if not trajectories_dir.exists():
        trajectories_dir.mkdir()
    select = lambda x: np.array([t[x] for t in trajs])
    np.savez(
        trajectories_dir / (sample_id + ".npz"),
        t=select(0),
        s=select(1),
        a=select(2),
        r=select(3),
        s_next=select(4),
        done=select(5),
    )

    contents = {
        "path": str(model_path),
        "uuid": sample_id,
        "steps": np.mean(steps_values),
        "success_rate": successes / n_episodes,
        "entropy": entropy,
        "discretize": discretize,
        "usages": np_bn_values.mean(0).tolist(),
        "vectors": vectors.tolist(),
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
    target_fn = "best.pt" if target_ts is None else f"model-{target_ts}.pt"
    if progression:
        new_paths = sorted(
            root.glob("model*.pt"),
            key=lambda x: int(str(x).split("-")[-1].split(".")[0]),
        )
        paths.extend(new_paths)
    elif len({target_fn, "config.pkl"} & names) == 2:
        paths.append(root / target_fn)
    paths.extend(x for c in contents for x in expand_paths(c, progression, target_ts))
    return paths


def aggregate_results(
    path_strs: List[str],
    out_dir: Path,
    n_jobs: int,
    progression: bool,
    target_ts: Optional[int],
    eval_steps: int,
    df_concat_paths: List[Path],
) -> None:
    dfs_to_concat = [pd.read_csv(p) for p in df_concat_paths]
    if not out_dir.exists():
        out_dir.mkdir()
    paths = [x for p in path_strs for x in expand_paths(p, progression, target_ts)]
    jobs = [delayed(collect_metrics)(p, out_dir, eval_steps) for p in paths]
    results = [
        r for r in Parallel(n_jobs=n_jobs)(x for x in tqdm(jobs)) if r is not None
    ]
    df_contents = results
    df = pd.concat(df_contents + dfs_to_concat, ignore_index=True)
    df.to_csv(out_dir / "data.csv", index=False)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("targets", type=str, nargs="*")
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--out_dir", "-o", type=str, default=".")
    parser.add_argument("--progression", action="store_true")
    parser.add_argument("--target_timestep", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=10_000)
    parser.add_argument("--include_csv", type=str, action="append")
    parser.add_argument("-j", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    args.out_dir = Path(args.out_dir)
    if args.include_csv is None:
        args.include_csv = []

    if args.command == "eval":
        if args.target_timestep is None:
            raise ValueError("Please set --target_timestep")
        aggregate_results(
            args.targets,
            args.out_dir,
            args.j,
            args.progression,
            args.target_timestep,
            args.eval_steps,
            args.include_csv,
        )
    elif args.command == "run":
        run_experiments(args.targets, args.num_trials, args.j)
    else:
        raise ValueError(f"Command '{args.command}' not recognized.")


if __name__ == "__main__":
    main()
