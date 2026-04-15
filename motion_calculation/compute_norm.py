from orbit_mlp import compute_norm_stats_from_files, flogger

flogger.set_write_to_file(False)

compute_norm_stats_from_files(
    [
        "data/dataset_13S/training_data",
    ],
    "norms/orbit_norm_data_13S.json"
)
