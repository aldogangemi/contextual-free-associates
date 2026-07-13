import pandas as pd
import numpy as np

def process_and_save_dataset(input_filepath="fullexpandeddataset_updated.tsv", output_filepath="fullexpandeddataset_processed.tsv"):
    """
    Processes the TSV dataset to add a 'removal_status' column,
    remove RT1/RT2 values based on 'inconsistent' rows, and saves
    the modified data to a new TSV file.

    Args:
        input_filepath (str): The path to the input TSV dataset file.
        output_filepath (str): The path where the processed TSV dataset will be saved.
    """
    df = pd.read_csv(input_filepath, sep='\t')

    # 1) Add a new column in the second position for 'removal_status'
    df.insert(1, 'removal_status', '')

    # Group by 'input' to identify series of rows with the same input value
    for input_val, group in df.groupby('input'):
        inconsistent_rows = group[group['inconsistent'] == 'inconsistent']

        if not inconsistent_rows.empty:
            # Get the index of the very first inconsistent row in this group
            first_inconsistent_idx_in_group = inconsistent_rows.index[0]

            # Check if this first inconsistent row is also the first row of the entire 'input' group
            # This is the crucial check for 'remove-plus'
            is_first_row_of_group = (group.index[0] == first_inconsistent_idx_in_group)

            if is_first_row_of_group:
                # 2) The value will be "remove-plus" if the inconsistent row is the first of a series
                df.loc[first_inconsistent_idx_in_group, 'removal_status'] = 'remove-plus'
                # 4) Remove RT1 and RT2 values from this "remove-plus" row
                df.loc[first_inconsistent_idx_in_group, ['RT1', 'RT2']] = np.nan

                # 5) Remove RT1 and RT2 values from ALL rows that follow this "remove-plus" row
                # and have the same input value. This means all rows in the group *after* the 'remove-plus' row.
                rows_after_remove_plus = group[group.index > first_inconsistent_idx_in_group]
                df.loc[rows_after_remove_plus.index, ['RT1', 'RT2']] = np.nan

                # All other inconsistent rows in this same group (if any) will be "remove"
                for idx in inconsistent_rows.index:
                    if idx != first_inconsistent_idx_in_group:
                        # 3) The value will be "remove" if the inconsistent row is not the first
                        df.loc[idx, 'removal_status'] = 'remove'
                        # RT values are already cleared by the previous step (point 5) for these rows.
                        # However, for clarity and robustness, we can explicitly set them to NaN again if needed.
                        # But typically, point 5 handles all subsequent RTs.
                        # For example: df.loc[idx, ['RT1', 'RT2']] = np.nan
            else:
                # If the first inconsistent row is NOT the first row of the overall 'input' group,
                # then all inconsistent rows in this group (including the first one) are just "remove".
                # 3) The value will be "remove" if the inconsistent row is not the first of a series,
                # or if there is only one row for a same input value (already covered if first_inconsistent_idx_in_group
                # is not the group's first row).
                for idx in inconsistent_rows.index:
                    df.loc[idx, 'removal_status'] = 'remove'
                    # 4) Remove RT1 and RT2 values from these inconsistent rows
                    df.loc[idx, ['RT1', 'RT2']] = np.nan

    # Save the processed DataFrame to a new TSV file
    df.to_csv(output_filepath, sep='\t', index=False)
    print(f"Processed data saved to '{output_filepath}'")

if __name__ == "__main__":
    # Call the function to process and save the dataset
    process_and_save_dataset()