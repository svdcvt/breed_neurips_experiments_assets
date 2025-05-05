# i have path to full validation
# i have path to models for different studies
# i want
# 1.1) plot of validation quality (nRMSE, valid rollout full) for one study and its different models
# 1.2) plot of prediction for few samples for one study and its different models
# 2.1) plot of validation quality (nRMSE, valid rollout full) for all studies and its best models
# 2.2) plot of prediction for few samples for all studies and its best models

# choose 3 validation samples based on diff_std

# arguments to pass
# 1. path to validation data
# 2. path(s) to study(ies) -> if list is given, then plot for all studies' best model; if one is given, then plot for that study several models

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import glob
from apebench_server import APEBenchServerValidation

import rapidjson
CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


# valid_ids_to_predict = [317, 1026, 689, 800, 927]
# valid_diffs = [0.003, 0.019, 0.022, 0.025, 0.036]
# valid_ids_to_predict = [317, 954, 167, 330, 1243, 81, 239, 884, 512, 125, 1022, 927]
# valid_diffs = [0.004, 0.010, 0.024, 0.027, 0.029, 0.030, 0.032, 0.033, 0.035, 0.038, 0.045, 0.051]

valid_ids_to_predict = [317, 439, 311, 154, 738, 1138, 87, 488, 595, 345, 1022, 927]
valid_diffs = [0.003, 0.007, 0.017, 0.019, 0.020, 0.022, 0.023, 0.024, 0.025, 0.027, 0.033, 0.036]

# valid_ids_to_predict = [317, 954, 1025, 1, 459, 274, 551, 77, 1022, 927]
# valid_diffs = [0.004, 0.010, 0.025, 0.028, 0.030, 0.032, 0.034, 0.037, 0.045, 0.051]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot model predictions and validation quality.")
    parser.add_argument("--study-paths", type=str, nargs='+', default=".", help="Path(s) to study(ies) with model(s).")
    parser.add_argument("--output-dir", type=str, default="./validation_results/", help="Directory to save plots.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.study_paths = [os.path.abspath(path) for path in args.study_paths]
    study_names = [os.path.split(path)[1] for path in args.study_paths]
    configs = []
    for study_path in args.study_paths:
        config_path = glob.glob(os.path.join(study_path, "config*.json"))[0]
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config = rapidjson.load(f, parse_mode=CONFIG_PARSE_MODE)
                configs.append(config)
        else:
            print(f"Config file not found in {study_path}. Skipping this study.")
            continue

    if len(configs) > 1:
        # we take model_best from each directory, so APEBenchServerValidation().load_model_from_checkpoint(-1) from each
        # gather list of configs, model_index = [-1]
        model_index = [-1]
    elif len(configs) == 1:
        # we take all models from the directory, so APEBenchServerValidation().load_model_from_checkpoint(i) for i in range  the directory
        # list of configs is [config], model_index is range(server.num_models - 1) + [-1]
        model_index = [None, -1]
    elif len(configs) == 0:
        # it is an error, we should not be here
        print("No config files found in the provided study paths. Exiting.")
        sys.exit(1)

    stats_names = [
        "max", "min", "std", "mean_pstd", "mean_mean", "mean_mstd",
        "p90", "p75", "p50", "p25", "p10"
    ]

    if os.path.exists(os.path.join(args.output_dir, "validation_results.csv")) and \
            os.path.exists(os.path.join(args.output_dir, "loss_per_sample.npy")) and \
            os.path.exists(os.path.join(args.output_dir, "rollout_losses.npy")) and \
            os.path.exists(os.path.join(args.output_dir, "predictions.npy")) and \
            os.path.exists(os.path.join(args.output_dir, "val_samples_to_predict.npy")):
        dataframe = pd.read_csv(os.path.join(args.output_dir, "validation_results.csv"))
        if set(dataframe["study"].values) != set(study_names):
            print("Warning: study names in the dataframe do not match the provided study paths. Exiting.")
            sys.exit(1)
        else:
            print("Study names in the dataframe match the provided study paths.")
        print("Validation results already exist. Loading from files.")
        all_loss_per_sample = np.load(os.path.join(args.output_dir, "loss_per_sample.npy"))
        all_rollout_losses = np.load(os.path.join(args.output_dir, "rollout_losses.npy"))
        all_predictions = np.load(os.path.join(args.output_dir, "predictions.npy"))
        all_val_samples_to_predict = np.load(os.path.join(args.output_dir, "val_samples_to_predict.npy"))
        if all_val_samples_to_predict.shape[0] != len(valid_ids_to_predict):
            print("Warning: number of validation samples to predict does not match the provided valid_ids_to_predict. Exiting.")
            sys.exit(1)
        validation_input_params = np.load(os.path.join(config["dl_config"]["validation_directory"], "input_parameters.npy"))
    else:
        dataframe = pd.DataFrame(columns=["study", "model_index", "avg_val_loss", "rollout_loss"] + stats_names)
        all_loss_per_sample = []
        all_rollout_losses = []
        all_predictions = []
        all_val_samples_to_predict = []

        for i, (config, study_path) in enumerate(zip(configs, args.study_paths)):
            print(f"Processing study {i + 1}/{len(configs)}: {config['output_dir']}")
            all_loss_per_sample.append([])
            all_rollout_losses.append([])
            all_predictions.append([])

            # we should read json config*.json and pass to the APEBenchServerValidation
            # server = APEBenchServerValidation(config_dict)
            # at that moment server has loaded validation path as in the config, network architecture, and other parameters
            # for the first iteration we load validation through server object, but further we wil reuse these as to not reload them again
            # server.load_validation_data()
            # previous_validation_data = server.valid_dataset, server.valid_parameters, server.valid_dataloader, server.valid_dataloader_rollout
            # or
            # server.load_validation_data(*previous_validation_data)
            # then either we iterate over the models [0,1,...,len(server.models_paths)-1] or we take the best model [-1]
            # if None we do for i in range(server.num_models - 1)
            # if -1 we use best only
            # server.load_model_from_checkpoint(-1)
            # at that moment server.model is the model we want to use
            # avg_val_loss, loss_per_sample, stats, rollout_loss, rollout_losses = server.run_validation()
            # this can take very long but tqdm helps to monitor that
            # to do prediction, we have validation sample index i, then
            # sample, prediction = server.rollout_sample(i)

            server = APEBenchServerValidation(config, study_path)
            if i == 0:
                server.load_validation_data()
                validation_data = (server.valid_dataset, server.valid_parameters, server.valid_dataloader, server.valid_dataloader_rollout)
            else:
                server.load_validation_data(validation_data)

            for model_idx in model_index:
                if model_idx is None:
                    for model_idx in range(server.num_models - 2):
                        print(f"Loading model {model_idx} from {server.models_paths[model_idx]}")
                        server.load_model_from_checkpoint(model_idx)
                        avg_val_loss, loss_per_sample, stats, rollout_loss, rollout_losses, rollout_losses_all = server.run_validation()
                        all_loss_per_sample[-1].append(loss_per_sample)
                        all_rollout_losses[-1].append(rollout_losses_all)
                        dataframe = pd.concat([
                            dataframe,
                            pd.Series({
                                "study": os.path.split(server.study_path)[1],
                                "model_index": model_idx,
                                "avg_val_loss": avg_val_loss,
                                "rollout_loss": rollout_loss,
                                **{f"{name}": stats[name] for name in stats_names}
                            }).to_frame().T
                        ], ignore_index=True)
                else:
                    print(f"Loading best model from {server.models_paths[-1]}")
                    server.load_model_from_checkpoint(-1)
                    avg_val_loss, loss_per_sample, stats, rollout_loss, rollout_losses, rollout_losses_all = server.run_validation()
                    all_loss_per_sample[-1].append(loss_per_sample)
                    all_rollout_losses[-1].append(rollout_losses_all)
                    dataframe = pd.concat([
                        dataframe,
                        pd.Series({
                            "study": os.path.split(server.study_path)[1],
                            "model_index": f"best_batch_{server.model_at_batch}",
                            "avg_val_loss": avg_val_loss,
                            "rollout_loss": rollout_loss,
                            **{f"{name}": stats[name] for name in stats_names}
                        }).to_frame().T
                    ], ignore_index=True)
                    for val_id in valid_ids_to_predict:
                        sample, prediction = server.rollout_sample(val_id)
                        if len(all_val_samples_to_predict) < len(valid_ids_to_predict):
                            all_val_samples_to_predict.append(sample)
                        all_predictions[-1].append(prediction)

        # save dataframe to csv
        dataframe.to_csv(os.path.join(args.output_dir, "validation_results.csv"), index=False)
        print(dataframe)

        all_rollout_losses = np.array(all_rollout_losses).reshape(-1, server.valid_dataset.num_samples, server.valid_rollout)
        print(f"all_rollout_losses shape is {all_rollout_losses.shape}")
        # all_rollout_losses shape is (num_configs, num_samples_val, 100)

        all_loss_per_sample = np.array(all_loss_per_sample).reshape(-1, server.valid_dataset.num_samples, server.valid_dataset.nb_time_steps - 1)
        print(f"all_loss_per_sample shape is {all_loss_per_sample.shape}")
        # all_loss_per_sample shape is (num_configs, num_samples_val, 100)

        all_predictions = np.array(all_predictions)
        print(f"all_predictions shape is {all_predictions.shape}")
        # all_predictions shape is (num_configs, num_val_samples=5, 101, 1, 800)

        all_val_samples_to_predict = np.array(all_val_samples_to_predict)
        print(f"all_val_samples_to_predict shape is {all_val_samples_to_predict.shape}")
        # all_val_samples_to_predict shape is (num_val_samples=5, 101, 1, 800)

        np.save(os.path.join(args.output_dir, "loss_per_sample.npy"), all_loss_per_sample)
        np.save(os.path.join(args.output_dir, "rollout_losses.npy"), all_rollout_losses)
        np.save(os.path.join(args.output_dir, "predictions.npy"), all_predictions)
        np.save(os.path.join(args.output_dir, "val_samples_to_predict.npy"), all_val_samples_to_predict)
        print("Validation results saved to files.")
        validation_input_params = validation_data[1]

    all_names = dataframe.apply(lambda x: f"{x['study']}_{x['model_index']}", axis=1).values
    # plot rollout loss
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), layout='constrained')
    for i, (rollout_loss, name) in enumerate(zip(all_rollout_losses, all_names)):
        mean_rollout_loss = rollout_loss.mean(0)
        std_rollout_loss = rollout_loss.std(0)
        pl = ax.plot(mean_rollout_loss, label=name)
        ax.fill_between(
            np.arange(len(mean_rollout_loss)),
            mean_rollout_loss - std_rollout_loss,
            mean_rollout_loss + std_rollout_loss,
            alpha=0.1,
            color=pl[0].get_color()
        )
    ax.vlines(
        75,  # TODO: this is hardcoded, should be from config
        ymin=0,
        ymax=ax.get_ylim()[1],
        color="red",
        linestyle="--",
        label="Known horizon"
    )
    fig.legend(loc="outside lower left", fontsize=6)
    ax.grid(which="both")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("nRMSE")
    ax.set_title("Validation rollout loss (non-cumulative)")
    plt.savefig(os.path.join(args.output_dir, "validation_rollout_loss.png"), dpi=300)
    plt.close(fig)

    # plot loss per sample
    all_loss_per_sample_mean = all_loss_per_sample.mean(-1)  # (num_configs, num_samples_val)
    min_loss, max_loss = all_loss_per_sample.min(), all_loss_per_sample.max()
    bins = np.linspace(min_loss, max_loss, 100)
    max_loss_mean = np.max(all_loss_per_sample_mean)

    ncols = len(all_names)
    width = ncols * 5 + ncols - 1
    height = 5 * 2 + 2

    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(2, ncols, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = [fig.add_subplot(gs[1, i]) for i in range(ncols)]

    ax1.set_title("Empirical CDF")
    ax1.set_xlabel("Loss")
    ax1.set_ylabel("Probability of occurrence")
    ax1.grid(which="both")
    ax1.set_xscale("log")
    ax1.set_xlim(min_loss * 0.98, max_loss * 1.02)
    ax1.set_ylim(0, 1.05)

    for i, (loss_per_sample_mean, name) in enumerate(zip(all_loss_per_sample_mean, all_names)):
        ax1.ecdf(all_loss_per_sample[i].ravel(), label=name, linewidth=1.5)
        ax2[i].set_title(name + "\nLoss per sample")
        sc = ax2[i].scatter(
            *validation_input_params[:, [0, 2]].T,  # input parameters of samples
            c=loss_per_sample_mean,
            s=20, cmap="Reds", vmin=0, vmax=max_loss_mean, alpha=0.5
        )
        ax2[i].set_aspect("equal")
        ax2[i].set_xlabel("Amp0")
        ax2[i].set_ylabel("Amp1")
        ax2[i].set_xlim(-1, 1)
        ax2[i].set_ylim(-1, 1)
    ax1.legend(loc="lower right", fontsize=6)
    fig.colorbar(sc, ax=ax2[-1], label="Mean Loss")
    plt.savefig(os.path.join(args.output_dir, "validation_loss_per_sample.png"), dpi=300)
    plt.close(fig)

    # plot predictions
    min_val_per_sample, max_val_per_sample = all_val_samples_to_predict.min((1, 2, 3)), all_val_samples_to_predict.max((1, 2, 3))
    ncols = len(valid_ids_to_predict)
    nrows = len(all_predictions) + 1
    width = ncols * 5 + ncols - 1
    height = nrows * 4
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        sharey=True, sharex=True
    )
    for j in range(len(valid_ids_to_predict)):
        sample = all_val_samples_to_predict[j]
        if j == 0:
            ax[0][j].set_title(f"True Simulation\t diff={valid_diffs[j]}", fontsize=10, loc="left")
        else:
            ax[0][j].set_title(f"diff={valid_diffs[j]}, simid={valid_ids_to_predict[j]}", fontsize=10, loc="left")
        im = ax[0][j].imshow(sample.squeeze(1).T, cmap="coolwarm", vmin=min_val_per_sample[j], vmax=max_val_per_sample[j], aspect="auto")
        cb = fig.colorbar(im, ax=ax[:, j], fraction=0.05, pad=0.05, aspect=40, extend="both")
        cb.ax.tick_params(labelsize=8)

    for i, (predictions, name) in enumerate(zip(all_predictions, all_names), start=1):
        for j, pred in enumerate(predictions):
            im = ax[i][j].imshow(pred.squeeze(1).T, cmap="coolwarm", vmin=min_val_per_sample[j], vmax=max_val_per_sample[j], aspect="auto")
        ax[i][0].set_title(name, fontsize=10, loc="left")

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].set_yticks([])
            if i == nrows - 1:
                ax[i][j].set_xlabel("Time ->")

    # fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "validation_predictions.png"), dpi=300)
    plt.close(fig)

    # plot predictions
    max_abs_error = (np.maximum(abs(max_val_per_sample), abs(min_val_per_sample)) * 2) ** 2

    ncols = len(valid_ids_to_predict)
    nrows = len(all_predictions) + 1
    width = ncols * 5 + ncols - 1
    height = nrows * 4
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        sharey=True, sharex=True
    )
    for j in range(len(valid_ids_to_predict)):
        sample = all_val_samples_to_predict[j]
        if j == 0:
            ax[0][j].set_title(f"True Simulation | diff={valid_diffs[j]}, simid={valid_ids_to_predict[j]}", fontsize=16, loc="left")
        else:
            ax[0][j].set_title(f"diff={valid_diffs[j]}, simid={valid_ids_to_predict[j]}", fontsize=16, loc="left")
        im = ax[0][j].imshow(sample.squeeze(1).T, cmap="coolwarm", vmin=min_val_per_sample[j], vmax=max_val_per_sample[j], aspect="auto")
    cb = fig.colorbar(im, ax=ax[0][j], fraction=0.05, pad=0.05, aspect=40, extend="both")
    cb.ax.tick_params(labelsize=8)

    for i, (predictions, name) in enumerate(zip(all_predictions, all_names), start=1):
        for j, pred in enumerate(predictions):
            im = ax[i][j].imshow(((pred - sample) ** 2).squeeze(1).T, cmap="Reds", vmin=0, vmax=max_abs_error[j], aspect="auto")
            if i == 0:
                cb = fig.colorbar(im, ax=ax[1:, j], fraction=0.05, pad=0.05, aspect=40, extend="both")
                cb.ax.tick_params(labelsize=10)
        ax[i][0].set_title(name, fontsize=16, loc="left")

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].set_yticks([])
            if i == nrows - 1:
                ax[i][j].set_xlabel("Time ->")

    # fig.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "validation_predictions_error.png"), dpi=300)
    plt.close(fig)

    print(f"Plots saved to {args.output_dir}")
    print("Done.")
