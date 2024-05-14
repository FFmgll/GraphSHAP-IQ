"""This script renames the files in the approximation directories."""

import os
import sys

from approximation_utils import (
    L_SHAPLEY_APPROXIMATION_DIR,
    BASELINES_DIR,
    GRAPHSHAPIQ_APPROXIMATION_DIR,
    EXACT_DIR,
    ALL_DIRECTORIES,
    parse_file_name,
    create_file_name,
)

if __name__ == "__main__":
    # DONT RUN THIS SCRIPT UNLESS YOU KNOW WHAT YOU ARE DOING
    sys.exit(0)
    # rename files in the baselines directory
    for graph_shapiq_file in os.listdir(GRAPHSHAPIQ_APPROXIMATION_DIR):
        file_attributes = parse_file_name(graph_shapiq_file)
        max_interaction_size = str(file_attributes["max_interaction_size"])  # this will be inserted

        first_file_identifier = "_".join(
            (
                str(file_attributes["model_id"]),
                str(file_attributes["dataset_name"]),
                str(file_attributes["n_layers"]),
                str(file_attributes["data_id"]),
                str(file_attributes["n_players"]),
            )
        )

        budget = str(file_attributes["budget"])

        for save_directory in ALL_DIRECTORIES:
            if (
                save_directory == EXACT_DIR
                or save_directory == BASELINES_DIR
                or save_directory == GRAPHSHAPIQ_APPROXIMATION_DIR
                or save_directory == L_SHAPLEY_APPROXIMATION_DIR
            ):
                continue
            # only baseline approximations
            for file in os.listdir(save_directory):
                if first_file_identifier in file and budget in file:
                    # get the iteration number of this file
                    approx_file_attributes = parse_file_name(file)
                    new_file_name = create_file_name(
                        model_id=approx_file_attributes["model_id"],
                        dataset_name=approx_file_attributes["dataset_name"],
                        n_layers=approx_file_attributes["n_layers"],
                        data_id=approx_file_attributes["data_id"],
                        n_players=approx_file_attributes["n_players"],
                        max_interaction_size=int(max_interaction_size),
                        efficiency=approx_file_attributes["efficiency"],
                        budget=approx_file_attributes["budget"],
                        iteration=approx_file_attributes["iteration"],
                        index=approx_file_attributes["index"],
                        order=approx_file_attributes["order"],
                    )
                    print(f"Renaming {file} to {new_file_name}")
                    os.rename(
                        os.path.join(save_directory, file),
                        os.path.join(save_directory, new_file_name),
                    )
