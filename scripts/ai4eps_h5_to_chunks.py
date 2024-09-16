""" 
ai4eps_h5_to_chunks.py: this file splits the AI4EPS hdf5 file into chunks to improve the performance of the dataloader.
"""
import h5py
import argparse
from pathlib import Path
from joblib import Parallel, delayed


def copy_group_to_new_file(input_h5_file_path, group_name, output_folder):
    # Create the output file path
    output_file_path = output_folder / f"{group_name}.h5"

    # Open the input h5py file in read mode
    with h5py.File(input_h5_file_path, "r") as input_h5_file:
        # Open the output h5py file in write mode
        with h5py.File(output_file_path, "w") as output_h5_file:
            # Copy the group from the input h5py file to the output h5py file
            input_h5_file.copy(group_name, output_h5_file)


def parallel_split(input_h5_file_path, output_folder, n_jobs=-1):
    # Create the output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open the input h5py file in read mode
    with h5py.File(input_h5_file_path, "r") as input_h5_file:
        # Get the list of group names in the input h5py file
        group_names = list(input_h5_file.keys())

    # Use joblib to parallelize the copying process
    Parallel(n_jobs=n_jobs)(
        delayed(copy_group_to_new_file)(
            input_h5_file_path, group_name, output_folder)
        for group_name in group_names
    )


def main(args):
    input_h5_file_path = Path(args.input_h5_file)
    output_folder = Path(args.output_folder)
    parallel_split(input_h5_file_path, output_folder, n_jobs=args.n_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split h5py file groups into separate files")
    parser.add_argument("input_h5_file", help="Path to the input h5py file")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("-j", "--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs (default: -1, use all available processors)")

    args = parser.parse_args()
    main(args)
