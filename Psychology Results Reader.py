import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os

def analyze_results(file_paths):
    """
    Analyzes multiple result files to consolidate word recognition data,
    including the participant number from the filename.

    Args:
        file_paths (list): A list of paths to the Excel result files.

    Returns:
        pandas.DataFrame: A DataFrame containing the consolidated analysis
                          with a 'Participant' column.
    """
    all_data = []
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path)
            filename = os.path.basename(file_path)
            participant = ""
            for char in filename:
                if char.isdigit():
                    participant += char
                elif participant:
                    break
            if not participant:
                parts = filename.split(' ', 1)
                if parts:
                    participant = parts[0]

            df['Participant'] = participant
            all_data.append(df)
        except FileNotFoundError:
            pass # Error message removed
        except Exception as e:
            pass # Error message removed

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    grouped = combined_df.groupby('Participant').agg(
        Total_Associated_Words_Shown=('Associated', 'sum'),
        Correct_Associated_Word_Count=('Correct', lambda x: x[combined_df['Associated']].sum()),
        Total_Non_Associated_Words_Shown=('Associated', lambda x: (~x).sum()),
        Correct_Non_Associated_Word_Count=('Correct', lambda x: x[~combined_df['Associated']].sum())
    ).reset_index()

    return grouped

def select_files():
    """Opens a file dialog to allow the user to select multiple Excel files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(
        title="Select Result Files",
        filetypes=[("Excel files", "*.xlsx;*.xls")],
    )
    return file_paths

def save_analysis_to_excel(analysis_df, save_path):
    """Saves the analysis DataFrame to an Excel file at the specified path with a fixed filename."""
    output_filename = "Consolidated Psych Test Results.xlsx"
    full_file_path = os.path.join(save_path, output_filename)
    try:
        analysis_df.to_excel(full_file_path, index=False)
    except Exception as e:
        pass # Error message removed

if __name__ == "__main__":
    selected_files = select_files()
    if selected_files:
        results_df = analyze_results(selected_files)

        if not results_df.empty:
            save_path = r"C:\Users\fongj\OneDrive\Documents\Python Scripts\Personal Projects\Psychology Project"
            save_analysis_to_excel(results_df, save_path)
    # No output messages in the main block