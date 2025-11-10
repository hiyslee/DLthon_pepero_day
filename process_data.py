import pandas as pd
import numpy as np
import os

def process_conversations(cleaned_input_path, aug_input_path, output_path):
    """
    Processes conversation data by:
    1. Reading the cleaned conversation data.
    2. Randomly deleting 500 rows with the class "일반 대화".
    3. Reading an augmentation file ('agg.csv').
    4. Appending the conversations from the augmentation file as new rows with the class "일반 대화".
    5. Saving the result to a new CSV file.

    Args:
        cleaned_input_path (str): Path to the cleaned CSV file (e.g., 'aug_hub_cleaned.csv').
        aug_input_path (str): Path to the augmentation CSV file (e.g., 'agg.csv').
        output_path (str): Path to save the final processed CSV file.
    """
    try:
        # 1. Read the cleaned conversation data
        df_cleaned = pd.read_csv(cleaned_input_path, encoding='utf-8')
        print(f"Successfully loaded {cleaned_input_path}. Shape: {df_cleaned.shape}")

        # 2. Identify and randomly select 500 "일반 대화" rows to delete
        general_conv_indices = df_cleaned[df_cleaned['class'] == '일반 대화'].index
        
        if len(general_conv_indices) >= 500:
            # Randomly sample 500 indices to drop
            indices_to_drop = np.random.choice(general_conv_indices, 500, replace=False)
            
            # Drop the selected rows
            df_processed = df_cleaned.drop(indices_to_drop)
            print(f"Randomly deleted 500 rows of class '일반 대화'. New shape: {df_processed.shape}")
        else:
            print(f"Warning: Found less than 500 rows with class '일반 대화'. No rows were deleted.")
            df_processed = df_cleaned.copy()

        # 3. Read the augmentation file
        df_aug = pd.read_csv(aug_input_path, encoding='utf-8')
        print(f"Successfully loaded {aug_input_path}. Shape: {df_aug.shape}")

        # 4. Append the new conversations
        # The user wants to append the new conversations as new rows with the class "일반 대화".
        # We will extract the 'conversation' column and create a new DataFrame.
        new_conversations = df_aug['conversation']
        new_data = {
            'class': '일반 대화',
            'conversation': new_conversations
        }
        df_new = pd.DataFrame(new_data)

        # Concatenate the processed DataFrame with the new data
        df_final = pd.concat([df_processed, df_new], ignore_index=True)
        
        # The 'idx' column will be missing for the new rows and the index will be discontinuous.
        # We will reset the index and use it as the new 'idx'.
        df_final = df_final.reset_index(drop=True)
        df_final['idx'] = df_final.index

        print(f"Appended {len(df_new)} new conversations. Final shape: {df_final.shape}")

        # 5. Save the result
        df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Successfully saved the processed data to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define file paths
    data_dir = 'Data/aiffel-dl-thon-dktc-online-15'
    cleaned_csv_path = os.path.join(data_dir, 'aug_hub_cleaned.csv')
    aug_csv_path = os.path.join(data_dir, 'agg.csv')
    output_csv_path = os.path.join(data_dir, 'aug_hub_agg_cleaned.csv')
    
    # Get the absolute path of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct absolute paths for the data files
    absolute_cleaned_path = os.path.normpath(os.path.join(current_script_dir, cleaned_csv_path))
    absolute_aug_path = os.path.normpath(os.path.join(current_script_dir, aug_csv_path))
    absolute_output_path = os.path.normpath(os.path.join(current_script_dir, output_csv_path))

    process_conversations(absolute_cleaned_path, absolute_aug_path, absolute_output_path)
