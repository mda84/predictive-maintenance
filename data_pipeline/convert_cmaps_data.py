import os
import pandas as pd

# Column names based on NASA C-MAPSS dataset documentation.
COLUMN_NAMES = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

def convert_txt_to_df(txt_file_path, scenario_label):
    """
    Reads a NASA C-MAPSS text file (train or test) and returns a DataFrame with headers.
    Adds a 'scenario' column to identify the source (e.g., 'train_FD001' or 'test_FD001').
    """
    df = pd.read_csv(txt_file_path, sep=r"\s+", header=None)
    df.columns = COLUMN_NAMES
    df["scenario"] = scenario_label
    return df

def add_rul_column(df, rul_txt_path):
    """
    Reads a NASA RUL text file and computes the RUL for each row in df.
    
    Process:
      1) Read RUL file (each row corresponds to an engine; row 0 → engine 1, etc.).
      2) For each engine (unit_number), find its max cycle in df.
      3) Merge final_rul (from RUL file) with max cycle.
      4) Compute RUL = final_rul + (max_cycle - time_in_cycles)
    """
    # Step 1: Load the RUL file.
    rul_data = pd.read_csv(rul_txt_path, sep=r"\s+", header=None, names=["final_rul"])
    rul_data["unit_number"] = rul_data.index + 1  # engine IDs are 1-based
    
    # Step 2: For each engine in df, get the maximum cycle.
    max_cycle_df = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycle_df.rename(columns={"time_in_cycles": "max_cycle"}, inplace=True)
    
    # Step 3: Merge with RUL data.
    merged = max_cycle_df.merge(rul_data, on="unit_number", how="left")
    
    # Step 4: Merge with main df and compute RUL.
    df_merged = df.merge(merged, on="unit_number", how="left")
    df_merged["RUL"] = df_merged["final_rul"] + (df_merged["max_cycle"] - df_merged["time_in_cycles"])
    
    # Optionally drop intermediate columns.
    df_merged.drop(columns=["final_rul", "max_cycle"], inplace=True)
    return df_merged

def merge_cmaps_scenarios(base_dir, train_files, rul_files, output_csv="sensor_data.csv"):
    """
    Merges multiple training files (e.g., train_FD001.txt, etc.) with their corresponding RUL files.
    Produces a single CSV file with a computed RUL column.
    """
    all_dfs = []
    for train_file, rul_file in zip(train_files, rul_files):
        scenario_label = train_file.replace(".txt", "")  # e.g., 'train_FD001'
        train_path = os.path.join(base_dir, train_file)
        rul_path = os.path.join(base_dir, rul_file)
        df = convert_txt_to_df(train_path, scenario_label)
        df = add_rul_column(df, rul_path)
        all_dfs.append(df)
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(base_dir, output_csv)
    merged_df.to_csv(out_path, index=False)
    print(f"Merged training scenarios into {out_path} with shape {merged_df.shape}.")

def merge_cmaps_test(base_dir, test_files, rul_files, output_csv="sensor_test_data.csv"):
    """
    Merges multiple test files (e.g., test_FD001.txt, etc.) with their corresponding RUL files.
    Produces a single CSV file with a computed RUL column.
    """
    all_dfs = []
    for test_file, rul_file in zip(test_files, rul_files):
        scenario_label = test_file.replace(".txt", "")  # e.g., 'test_FD001'
        test_path = os.path.join(base_dir, test_file)
        rul_path = os.path.join(base_dir, rul_file)
        df = convert_txt_to_df(test_path, scenario_label)
        df = add_rul_column(df, rul_path)
        all_dfs.append(df)
    
    merged_df = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(base_dir, output_csv)
    merged_df.to_csv(out_path, index=False)
    print(f"Merged test scenarios into {out_path} with shape {merged_df.shape}.")

if __name__ == "__main__":
    # Set your base directory where the NASA C-MAPSS text files are stored.
    base_dir = "../data/CMAPSSData"
    
    # Lists for training and test files.
    train_files = [
        "train_FD001.txt",
        "train_FD002.txt",
        "train_FD003.txt",
        "train_FD004.txt"
    ]
    rul_train_files = [
        "RUL_FD001.txt",
        "RUL_FD002.txt",
        "RUL_FD003.txt",
        "RUL_FD004.txt"
    ]
    
    test_files = [
        "test_FD001.txt",
        "test_FD002.txt",
        "test_FD003.txt",
        "test_FD004.txt"
    ]
    rul_test_files = [
        "RUL_FD001.txt",  # Typically, test sets use the same RUL files as training sets.
        "RUL_FD002.txt",
        "RUL_FD003.txt",
        "RUL_FD004.txt"
    ]
    
    # Merge training data.
    merge_cmaps_scenarios(base_dir, train_files, rul_train_files, output_csv="sensor_data.csv")
    
    # Merge test data.
    merge_cmaps_test(base_dir, test_files, rul_test_files, output_csv="sensor_test_data.csv")
