import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import pandas as pd
import seaborn as sns
import glob

import rapidjson
CONFIG_PARSE_MODE = rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS


NUM = 5
DEPTH_PDE_FOLDER = 6
valid_ids_path = "$DATASET_ROOT/validation_ids.csv"
MODEL_ID_FOR_PREDICTION = 4000

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot model predictions and validation quality.")
    parser.add_argument("--study-paths", type=str, nargs='+', default=".", help="Path(s) to study(ies) with model(s).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save plots.")
    parser.add_argument("--model-id", type=int, default=None, help="Model ID for prediction.")

    parser.add_argument("--rollout-plot", action="store_true", help="Plot rollout loss.")
    parser.add_argument("--ecdf-plot", action="store_true", help="Plot validation distribution loss.")
    parser.add_argument("--scatter-plot", action="store_true", help="Plot validation loss scatter against input parameters")
    parser.add_argument("--predictions-plot", action="store_true", help="Plot predictions.")
    parser.add_argument("--predictions-error-plot", action="store_true", help="Plot prediction error.")
    parser.add_argument("--all-plots", action="store_true", help="Plot all.")

    args = parser.parse_args()
    if args.all_plots:
        args.rollout_plot = True
        args.ecdf_plot = True
        args.scatter_plot = True
        args.predictions_plot = True
        args.predictions_error_plot = True
    if not (args.rollout_plot or args.ecdf_plot or args.scatter_plot or args.predictions_plot or args.predictions_error_plot):
        print("No plots selected. Exiting.")
        sys.exit(1)
    
    # if args.study_paths is not None:
    args.study_paths = [os.path.abspath(path) for path in args.study_paths]
    args.study_paths = [path for path in args.study_paths if os.path.isdir(path)]
    study_names = [os.path.split(path)[1] for path in args.study_paths]
    study_common_pde = set([path.split('/')[DEPTH_PDE_FOLDER] for path in args.study_paths])
    if len(study_common_pde) > 1:
        print("Error: study paths do not share the same common PDE folder. Exiting.")
        sys.exit(1)
    else:
        study_common_pde = study_common_pde.pop()
    print(f"Common PDE folder: {study_common_pde}")
    configs = []
    for study_path in args.study_paths:
        print(f"Processing study: {study_path}")
        config_path = glob.glob(os.path.join(study_path, "config*.json"))[0]
        if os.path.exists(config_path):
            # print(f"Loading config from {config_path}")
            with open(config_path, 'r') as f:
                config = rapidjson.load(f, parse_mode=CONFIG_PARSE_MODE)
                configs.append(config)
        else:
            print(f"Config file not found in {study_path}. Skipping this study.")
            continue

    # read validation_ids.csv
    if os.path.exists(valid_ids_path):
        validation_ids_df = pd.read_csv(valid_ids_path, index_col=0, header=0)
        validation_ids_df = validation_ids_df.loc[study_common_pde]
        # print(validation_ids_df)
        valid_ids_to_predict = validation_ids_df.iloc[:NUM].values.astype(int).flatten()
        valid_diffs = validation_ids_df.iloc[NUM:].values.astype(float).flatten()
    else:
        if args.plot_predictions or args.predictions_error_plot:
            print(f"Validation ids file not found in {valid_ids_path}. Exiting. Run `find_validation_ids.py` to generate it.")
            sys.exit(1)

    if len(configs) > 1:
        # we take model_best from each directory, so APEBenchServerValidation().load_model_from_checkpoint(-1) from each
        # gather list of configs, model_index = [-1]
        if args.model_id is not None:
            model_index = [args.model_id]
        else:
            model_index = [-1]
    elif len(configs) == 1:
        # we take all models from the directory, so APEBenchServerValidation().load_model_from_checkpoint(i) for i in range  the directory
        # list of configs is [config], model_index is range(server.num_models - 1) + [-1]
        if args.model_id is not None:
            model_index = [args.model_id]
        else:
            model_index = [None, -1]
    elif len(configs) == 0:
        # it is an error, we should not be here
        print("No config files found in the provided study paths. Exiting.")
        sys.exit(1)

    stats_names = [
        "max", "min", "std", "mean_mean",
        "p90", "p75", "p50", "p25", "p10"
    ]

    if args.output_dir is None:
        if args.model_id is not None:
            args.output_dir = f"$REPO_ROOT/validation_results/{study_common_pde}/{args.model_id}/"
        else:
            args.output_dir = f"$REPO_ROOT/validation_results/{study_common_pde}/best/"
    else:
        if study_common_pde not in args.output_dir:
            args.output_dir = os.path.join(args.output_dir, study_common_pde)
            if args.model_id is not None:
                args.output_dir = os.path.join(args.output_dir, str(args.model_id))
            else:
                args.output_dir = os.path.join(args.output_dir, "best")
        else:
            if args.model_id is not None:
                if args.model_id not in args.output_dir:
                    args.output_dir = os.path.join(args.output_dir, str(args.model_id))
            else:
                if "best" not in args.output_dir:
                    args.output_dir = os.path.join(args.output_dir, "best")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"|||||||| Output directory: {args.output_dir}")

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
        if "ecdf_loss.pdf" in os.listdir(args.output_dir) and args.ecdf_plot:
            print("ECDF plot already exists. Skipping.")

        if "scatter_loss.pdf" in os.listdir(args.output_dir) and args.scatter_plot:
            print("Scatter plot already exists. Skipping.")
        if "rollout_loss_boxplot.pdf" in os.listdir(args.output_dir) and args.rollout_plot:
            print("Rollout loss boxplot already exists. Skipping.")
        if "predictions.pdf" in os.listdir(args.output_dir) and args.predictions_plot:
            print("Predictions plot already exists. Skipping.")
            exit(0)
            
        all_loss_per_sample = np.load(os.path.join(args.output_dir, "loss_per_sample.npy"))
        all_rollout_losses = np.load(os.path.join(args.output_dir, "rollout_losses.npy"))
        all_predictions = np.load(os.path.join(args.output_dir, "predictions.npy"))
        all_val_samples_to_predict = np.load(os.path.join(args.output_dir, "val_samples_to_predict.npy"))
        if all_val_samples_to_predict.shape[0] != NUM:
            print("Warning: number of validation samples to predict does not match the provided valid_ids_to_predict. Exiting.")
            sys.exit(1)
        validation_input_params = np.load(os.path.join(config["dl_config"]["validation_directory"], "input_parameters.npy"))
    else:
        from apebench_server import APEBenchServerValidation

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
                elif model_idx == -1:
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
                        if len(all_val_samples_to_predict) < NUM:
                            all_val_samples_to_predict.append(sample)
                        all_predictions[-1].append(prediction)
                else:
                    print(f"Loading model {model_idx}")
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
                    for val_id in valid_ids_to_predict:
                        sample, prediction = server.rollout_sample(val_id)
                        if len(all_val_samples_to_predict) < NUM:
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

    all_names_unsorted = dataframe.apply(lambda x: f"{x['study']}_{x['model_index']}", axis=1).values
    algo_names = ["soft", "broad", "precise", "mixed", "uniform", "no_resampling"]
    colors = ["#a92be2", "#004080", "#008000", "#15b2a8", "#FA0053", "#FF512E"]
    markers = ["^", "<", ">", "v", "o", "o"]
    properties_i_n_c_m = [((i, an.replace("no_resampling", "static"), name), colors[j], markers[j]) for j, an in enumerate(algo_names) for i, name in enumerate(all_names_unsorted) if an in name]

    model_state_id = [x[0][2].split("_")[0] for x in properties_i_n_c_m]

    plt.style.use("$REPO_ROOT/common/science.mplstyle")
    fig_width = 3.5
    fig_height = 2.5

    ##### ROLLOUT LOSS
    if args.rollout_plot:
        if all_rollout_losses.shape[2] > 100:
            rollout_ids = [9, 24, 39, 59, 79, 99, 119, 139, 159, 179, 199]
            width = fig_width * 1.5
        else:
            rollout_ids = [9, 29, 49, 74, 99]
            width = fig_width * 1.2
        all_rollout_subset = all_rollout_losses[:,:,rollout_ids]
        # mean/std across samples -> (num_conf, num_id)
        # all_rollout_subset_mean = all_rollout_subset.mean(1)
        # all_rollout_subset_std = all_rollout_subset.std(1)
        # convert to data frame to use snsplot
        all_rollout_subset = all_rollout_subset.ravel() # numconf X num_samples X 6 

        get_name = lambda x: x[0][1] if args.model_id is not None else f"{x[0][1].ljust(8, ' ')}(i={x[0][2].split('_')[-1]})"
        nnames = [get_name(x) for x in sorted(properties_i_n_c_m, key=lambda x: x[0][0])]
        nnnames = [get_name(x) for x in properties_i_n_c_m]
        cccolors = [x[1] for x in properties_i_n_c_m]
        mmarkers = [x[2] for x in properties_i_n_c_m]

        df = pd.DataFrame(all_rollout_subset, columns=["nRMSE"])
        df["rollout_step"] = np.tile(rollout_ids, len(all_rollout_subset)//len(rollout_ids)) + 1
        df["algorithm"] = np.repeat(nnames, len(rollout_ids) * all_rollout_losses.shape[1])
        

        fig, ax = plt.subplots(1, 1, figsize=(width, fig_height), layout='constrained')
        fig.suptitle("Validation rollout loss (non-cumulative)")
        axx = sns.boxplot(df, x="rollout_step", y="nRMSE", 
                          hue="algorithm",
                          fill=False,
                          palette=cccolors, ax=ax, 
                          native_scale=True, log_scale=(False, True),
                          hue_order=nnnames,
                          gap=0.1,
                          width=0.7,
                          linewidth=1.0,
                          whis=(0.05, 99.95),
                          flierprops={"marker": "x", "markersize": 2.5, "linewidth": 0.05},
                          )
        
        # axx.axvline(75 if len(rollout_ids) < 10 else 150, color="red", linestyle="dotted", label="Known horizon")
        sns.move_legend(axx, loc="lower left", bbox_to_anchor=(1.01, 0),
                        title=None, prop={"family":"monospace", "size": 6},
                        handlelength=0.5, handleheight=0.5)
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("nRMSE (logarithmic scale)")
        ax.set_xticks(rollout_ids)
        ax.set_xticklabels([x+1 for x in rollout_ids])
        ax.xaxis.minorticks_off()
        ax.yaxis.grid(which="major")

        plt.savefig(os.path.join(args.output_dir, "rollout_loss_boxplot.pdf"))
        plt.close(fig)


    #### ECDF OF VALIDATION LOSS
    if args.ecdf_plot:
        min_loss, max_loss = all_loss_per_sample.min(), all_loss_per_sample.max()
        all_loss_per_sample_percentile = np.percentile(all_loss_per_sample, [0, 100], axis=(1, 2)).T

        fig, ax1 = plt.subplots(1, 1, figsize=(fig_width * 1.25, fig_height* 1.25), layout='constrained')
        title = f"Empirical CDF of validation 1-to-1 loss"
        if args.model_id is not None:
            title += f" (i={args.model_id})"
        ax1.set_title(title)
        ax1.set_xlabel("Loss value")
        ax1.set_ylabel("Probability of occurrence")
        ax1.set_yticks([0, 0.05, 0.2, 0.50, 0.8, 0.95, 1.00])
        ax1.grid()
        ax1.set_xscale("log")
        ax1.set_xlim(min_loss * 0.9, max_loss * 1.1)
        ax1.set_ylim(-0.05, 1.05)

        for i, ((ii, alg, name), color, marker) in enumerate(properties_i_n_c_m):
            label = alg if args.model_id is not None else f"{alg} (i={name.split('_')[-1]})"
            ax1.ecdf(all_loss_per_sample[ii].ravel(), label=label, 
                     linewidth=1.5, color=color,
                     marker=marker, markersize=5,
                     fillstyle="full" if i !=4 else "none",
                     markeredgewidth=0.5 if i !=4 else 1)
            ll = len(ax1.lines[-1].get_xydata())
            ax1.lines[-1].set_markevery(ll//10)
             
        ax1.legend(loc="lower right")#, fontsize=6)
        plt.savefig(os.path.join(args.output_dir, "ecdf_loss.pdf"))#, dpi=300)
        plt.close(fig)

    #### LOSS SCATTER ONTO PARAMETERS 
    if args.scatter_plot:
        all_loss_per_sample_mean = all_loss_per_sample.mean(-1)  # (num_configs, num_samples_val)
        min_loss_mean = np.min(all_loss_per_sample_mean)
        max_loss_mean = np.max(all_loss_per_sample_mean)
        cmap = plt.get_cmap("gist_heat_r")
        norm = mpc.PowerNorm(gamma=0.5, vmin=min_loss_mean, vmax=max_loss_mean)
        
        ncols = len(properties_i_n_c_m)
        width = ncols * fig_width
        height = fig_width

        fig, ax2 = plt.subplots(1, ncols, figsize=(width, height), sharey=True, sharex=True, layout='constrained')
        fig.suptitle("Average validation loss per trajectory")

        for i, ((ii, alg, name), color, marker) in enumerate(properties_i_n_c_m, start=0):
            label = alg if args.model_id is not None else f"{alg} (i={name.split('_')[-1]})"
            ax2[i].set_title(label)
            sc = ax2[i].scatter(
                *validation_input_params[:, [0, 2]].T,  # input parameters of samples
                c=all_loss_per_sample_mean[ii],
                s=25, cmap=cmap, norm=norm, alpha=0.85
            )
            ax2[i].set_aspect("equal")
            ax2[i].set_xlabel("$A_0$")
            ax2[i].set_ylabel("$A_1$")
            ax2[i].set_xlim(-1.07, 1.07)
            ax2[i].set_ylim(-1.07, 1.07)
        fig.colorbar(sc, ax=ax2[-1], label="Mean MSE")
        plt.savefig(os.path.join(args.output_dir, "scatter_loss.pdf"))#, dpi=300)
        plt.close(fig)

    # plot predictions
    if args.predictions_plot or args.predictions_error_plot:
        min_val_per_sample, max_val_per_sample = all_val_samples_to_predict.min((1, 2, 3)), all_val_samples_to_predict.max((1, 2, 3))
        abs_max_val_per_sample = np.maximum(abs(max_val_per_sample), abs(min_val_per_sample))

        if args.predictions_plot:
            # print(f"min_val_per_sample: {min_val_per_sample}")
            # print(f"max_val_per_sample: {max_val_per_sample}")
            # across all models per val sample
            min_val_per_sample_pred, max_val_per_sample_pred = all_predictions.min((0, 2, 3, 4)), all_predictions.max((0, 2, 3, 4))
            # print(f"min_val_per_sample_pred: {min_val_per_sample_pred}")
            # print(f"max_val_per_sample_pred: {max_val_per_sample_pred}")
            ratio_min = min_val_per_sample / min_val_per_sample_pred # it should be >= 1
            ratio_max = max_val_per_sample / max_val_per_sample_pred # it should be >= 1
            # print(f"ratio_min: {ratio_min}")
            # print(f"ratio_max: {ratio_max}")
            # print((-1.5 * np.log10(ratio_min)))
            # print((-1.5 * np.log10(ratio_max)))

            # scaling factor should be defined based on ratio, 
            # the function then is 

            min_val_per_sample = np.where(ratio_min < 0.5, min_val_per_sample * (-1.5 * np.log10(ratio_min) + 1), min_val_per_sample)
            max_val_per_sample = np.where(ratio_max < 0.5, max_val_per_sample * (-1.5 * np.log10(ratio_max) + 1), max_val_per_sample)
            max_abs_val_per_sample = np.maximum(abs(max_val_per_sample), abs(min_val_per_sample))
            # print(f"min_val_per_sample: {min_val_per_sample}")
            # print(f"max_val_per_sample: {max_val_per_sample}")
            # print(f"max_abs_val_per_sample: {max_abs_val_per_sample}")

            cmap_norms = [
                mpc.CenteredNorm(vcenter=0.0, halfrange=max_abs_val_per_sample[j])
                for j in range(len(valid_ids_to_predict))
            ]
        else:
            # all_pred is num_conf, num_val, val shape
            # all_val_samples_to_predict is num_val, val shape
            all_errors = np.abs(all_predictions - all_val_samples_to_predict[None, ...])
            # all_errors is num_conf, num_val, T, 1, M
            min_error = all_errors.min((0, 2, 3, 4)) # shape is num_val
            max_error = all_errors.max((0, 2, 3, 4)) # shape is num_val
            print(f"min_error: {min_error}")
            print(f"max_error: {max_error}")
            # power norm should be defined based on difference between max error and max val sample
            # if they differ a lot then gamma less than 1
            # abs_max_val / max error -> closer to 1 closer to linear
            # gamma cant be lower than 0.5
            gammas = np.maximum(0.5, abs_max_val_per_sample / max_error)

            cmap_err = plt.get_cmap("gist_heat_r")
            cmap_norms_err = [
                mpc.PowerNorm(gamma=gammas[j], vmin=min_error[j], vmax=max_error[j])
                for j in range(len(valid_ids_to_predict))
            ]
            cmap_pred = plt.get_cmap("coolwarm")
            cmap_norms_pred = [
                mpc.CenteredNorm(vcenter=0.0, halfrange=abs_max_val_per_sample[j])
                for j in range(len(valid_ids_to_predict))
            ]

        ncols = len(valid_ids_to_predict)
        nrows = len(all_predictions) + 1
        width = fig_width * (ncols / nrows) * 3
        height = fig_height * 2.5
        fig, ax = plt.subplots(
            nrows,
            ncols,
            figsize=(width, height),
            sharey=True, sharex=True,
            layout='constrained'
        )
        for j in range(len(valid_ids_to_predict)):
            sample = all_val_samples_to_predict[j]
            if j == 0:
                ax[0][j].set_title(f"True: std={valid_diffs[j]:.1e}", loc="left")
            else:
                ax[0][j].set_title(f"std={valid_diffs[j]:.1e}, j={int(valid_ids_to_predict[j])}", loc="left")
            if args.predictions_plot:
                im = ax[0][j].imshow(sample.squeeze(1).T, cmap="coolwarm", norm=cmap_norms[j], aspect="auto")
                cb = fig.colorbar(im, ax=ax[:, j], fraction=0.05, pad=0.05, aspect=40, extend="both")
            else:
                im = ax[0][j].imshow(sample.squeeze(1).T, cmap=cmap_pred, norm=cmap_norms_pred[j], aspect="auto")
                cb = fig.colorbar(im, ax=ax[0, j], fraction=0.05, pad=0.05, aspect=20, extend="both")
            cb.ax.tick_params(labelsize=6)
        cb.set_label(label='Solution Value', size=('large' if args.predictions_plot else "small"))

        for i, ((ii, alg, name), color, marker) in enumerate(properties_i_n_c_m, start=1):
            for j, pred in enumerate(all_predictions[ii]):
                if args.predictions_plot:
                    im = ax[i][j].imshow(pred.squeeze(1).T, cmap="coolwarm", norm=cmap_norms[j], aspect="auto")
                else:
                    im = ax[i][j].imshow(all_errors[ii][j].squeeze(1).T, cmap=cmap_err, norm=cmap_norms_err[j], aspect="auto")
                    if i == (len(properties_i_n_c_m) - 1):
                        cb = fig.colorbar(im, ax=ax[1:, j], fraction=0.05, pad=0.05, aspect=40, extend="both")
                        cb.ax.tick_params(labelsize=6)
            if args.model_id is None:
                title = f"{alg} (i={name.split('_')[-1]})"
            else:
                title = f"{alg}"
            ax[i][0].set_title(title, loc="left")
        if args.predictions_error_plot:                
            cb.set_label(label='Absolute Error', size='large')
        for i in range(len(ax)):
            for j in range(len(ax[i])):
                ax[i][j].set_yticks([])
                if i == nrows - 1:
                    ax[i][j].set_xlabel("Time")
        if args.predictions_plot:
            plt.savefig(os.path.join(args.output_dir, "predictions.pdf"))
        else:
            plt.savefig(os.path.join(args.output_dir, "predictions_error.pdf"))
        plt.close(fig)

    print(f"Plots saved to {args.output_dir}")
    print("Done.")
