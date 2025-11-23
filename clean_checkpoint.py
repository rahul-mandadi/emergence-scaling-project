import pandas as pd

checkpoint_path = 'data/checkpoints/checkpoint_results.csv'

try:
    print(f"Loading {checkpoint_path}...")
    
    # --- THIS IS THE FIX ---
    # We tell pandas to NOT convert empty strings to NaN.
    # This keeps the "" as a literal "" string.
    df = pd.read_csv(checkpoint_path, keep_default_na=False)
    # -----------------------

    print(f"Original row count: {len(df)}")

    # Now, we can reliably find all rows where 'raw_response' is literally ""
    bad_rows_mask = (df['raw_response'] == "")
    bad_row_count = bad_rows_mask.sum()

    if bad_row_count > 0:
        print(f"Found {bad_row_count} bad/failed rows to remove.")
        # Keep only the rows that are NOT in the bad_rows_mask
        good_df = df[~bad_rows_mask]
    else:
        print("No bad/failed rows found.")
        good_df = df # No changes needed

    # Save the cleaned data back to the same file
    good_df.to_csv(checkpoint_path, index=False)
    
    print(f"Checkpoint cleaned. New row count: {len(good_df)}")
    print("You can now safely re-run data_collector.py")

except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
except Exception as e:
    print(f"An error occurred: {e}")