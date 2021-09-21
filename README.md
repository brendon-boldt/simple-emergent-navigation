# Reward Shaping Interfers with Emergent Language


## Running the code

Create an environment (e.g., using `pip` or `conda`) and install the packages specified in `requirements.txt`.
This code has been tested with Python 3.8 and 3.9 on GNU/Linux.
Since the models used in this project are small, we recommend using a CPU and parallelization rather than a GPU.

To train the models run,
```
python -m simple_nav run quick_test -j4
```
where `quick_test` is the config-generating function in `simple_nav/experiment_configs.py` and `-j4` specifies the number of experiments to run in parallel (use `-j$(nproc)` for best performance and `-j1` for debugging).
`quick_test` is a toy configuration that should run quickly; it takes 1.5 minutes with `-j4` on a laptop with an Intel i7-4600U.
You can run TensorBoard in `log/` or `log/quick_test/` to track training progress.

After training the models, you must generate the evaluation data with those trained models.
```
python -m simple_nav eval log/quick_test -j4
```
where `log/quick_test` contains the trained models generated by the previous step.
This takes about 2 minutes on the aforementioned laptop.

Finally run the analysis using,
```
python -m simple_nav analyze quick_test
```
where `quick_test` is the name of analysis given in `simple_nav/analysis_configs.py`.
The results of the linear regression analysis will be printed to the screen; the figures will be saved under `results/quick_test/` as specified in `analysis_config.py`.


## Experiments used in the paper


We use the following experiment and analysis configurations in the paper:
- `nav_to_edges`
- `entropy_histogram`
- `world_radius`
- `buffer_size`
