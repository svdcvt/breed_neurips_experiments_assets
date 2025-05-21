### Module:

- `apebench_server.py`
    - Melissa server implementation for the APEBench framework
    -`APEBenchServer` for online training, 
    - `APEBenchOfflineServer` for creating validation dataset, 
    - `APEBenchValidationServer` for post-training inference.
- `ic_generation.py` and `sampler.py`
    - Parameter sampling implementation, child classes of AL method from Melissa fork
    - Manages parameter ranges and configurations
    - Processes sampled parameters for IC generation
    - Initial condition generation for PDEs (Exponax)
- `scenarios.py`: configures Melissa server parts according to APEBench scenario and provided configuration file
- `solver.py`: APEBench (Exponax) solvers (steppers) that produce data and either send it to the Melissa server for online training or save to files for dataset creation. Also creates solution trajectories visualisations, which helped to analyse PDEs behaviours.

---

### Script `plot_model_predictions.py`
- post-training inference and visualisations
- example below will load models weights of iteration 2k for PDE case `kdv__2w_x10_easier_max1` (KdV equation, easier difficulty coefficients, normalise IC, 2 waves IC), will inference the model on the validation dataset (paths are handled through config), and will produce "all plots" (ECDF, rollout box-plot, predictions)

```bash
python3 plot_model_predictions.py --study-paths ../experiments/set/diff_kdv__2w_x10_easier_max1_1d_x5/* --model-id 2000 --all-plots --output-dir ../validation_results/validation_results_decay/
```
