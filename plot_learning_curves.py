import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    seeds = args.seeds
    runs_dir = args.runs_dir

    train_arrays = []
    eval_arrays = []
    ref_train_steps = None
    ref_eval_steps = None

    for seed in seeds:
        run_name = f"sac_cartpole_swingup_seed{seed}"
        path = os.path.join(runs_dir, run_name, "logs.npz")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue

        data = np.load(path, allow_pickle=True)
        train_returns = data["train_returns"]
        train_steps = data["train_steps"]
        eval_returns = data["eval_returns"]
        eval_steps = data["eval_steps"]

        if ref_train_steps is None:
            ref_train_steps = train_steps
        if ref_eval_steps is None:
            ref_eval_steps = eval_steps

        train_arrays.append(train_returns)
        eval_arrays.append(eval_returns)

    if len(train_arrays) == 0:
        print("No logs found")
        return

    train_arrays = np.array(train_arrays) 
    eval_arrays = np.array(eval_arrays)   

    train_mean = train_arrays.mean(axis=0)
    train_std = train_arrays.std(axis=0)
    eval_mean = eval_arrays.mean(axis=0)
    eval_std = eval_arrays.std(axis=0)

    plt.figure()
    # training curve
    plt.fill_between(ref_train_steps, train_mean - train_std, train_mean + train_std, alpha=0.3)
    plt.plot(ref_train_steps, train_mean, label="Train return (mean)")

    # evaluation curve
    plt.fill_between(ref_eval_steps, eval_mean - eval_std, eval_mean + eval_std, alpha=0.3)
    plt.plot(ref_eval_steps, eval_mean, linestyle="--", label="Eval return (mean)")

    plt.xlabel("Environment steps")
    plt.ylabel("Episode return")
    plt.title("SAC on DMControl cartpole/swingup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--outfile", type=str, default="cartpole_sac_learning_curve.png")
    args = parser.parse_args()
    main(args)
