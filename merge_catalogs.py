import os, sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
#from tqdm import tqdm

def merge_based_on_proximity(df_rec, df_noise, df_org, threshold=3):
    merged_results = []

    #for image_id in tqdm(df_rec['image_id'].unique(), desc="Processing image_id", unit="image_id"):
    for image_id in df_rec['image_id'].unique():
        df_rec_filtered = df_rec[df_rec['image_id'] == image_id]
        df_noise_filtered = df_noise[df_noise['image_id'] == image_id]
        df_org_filtered = df_org[df_org['image_id'] == image_id].reset_index(drop=True)

        exp_times = df_rec_filtered['new_exp_time'].unique()
        for new_exp_time in exp_times:
            df_rec_second_filtered = df_rec_filtered[df_rec_filtered['new_exp_time'] == new_exp_time].reset_index(drop=True)
            df_noise_second_filtered = df_noise_filtered[df_noise_filtered['new_exp_time'] == new_exp_time].reset_index(drop=True)

            rec_coords = df_rec_second_filtered[['x', 'y']].values
            noise_coords = df_noise_second_filtered[['x', 'y']].values
            org_coords = df_org_filtered[['x', 'y']].values

            merged_rows = []

            # Match rec with org
            if len(org_coords) > 0:
                tree = KDTree(org_coords)
                dist, indices_org = tree.query(rec_coords)
                valid_matches_org = dist <= threshold
            else:
                valid_matches_org = np.zeros(len(rec_coords), dtype=bool)

            for i, row_rec in df_rec_second_filtered.iterrows():
                row_rec_renamed = row_rec.rename(lambda x: x + '_rec' if x != 'image_id' else x).to_dict()

                if len(org_coords) > 0 and valid_matches_org[i].any():
                    for j in np.where(valid_matches_org[i])[0]:
                        row_org = df_org_filtered.iloc[indices_org[i][j]]
                        row_org_renamed = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict()
                        merged_row = {**row_rec_renamed, **row_org_renamed}
                        merged_row['i'] = i
                        merged_rows.append(merged_row)
                else:
                    # If no match in org, fill org columns with NaN
                    for col in df_org.columns:
                        if col != 'image_id':
                            row_rec_renamed[col + '_org'] = np.nan
                    row_rec_renamed['i'] = i
                    merged_rows.append(row_rec_renamed)

            merged_rows_df = pd.DataFrame(merged_rows)

            # Match merged_rows with noise
            if len(noise_coords) > 0:
                tree = KDTree(noise_coords)
                dist, indices_noise = tree.query(rec_coords)
                valid_matches_noise = dist <= threshold
            else:
                valid_matches_noise = np.zeros(len(rec_coords), dtype=bool)

            for index, row_rec in merged_rows_df.iterrows():
                i = row_rec['i']
                row_rec_dict = row_rec.to_dict()
                if len(noise_coords) > 0 and valid_matches_noise[i].any():
                    for j in np.where(valid_matches_noise[i])[0]: 
                        row_noise = df_noise_second_filtered.iloc[indices_noise[i][j]]
                        row_noise_renamed = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict()
                        merged_row = {**row_rec_dict, **row_noise_renamed}
                        merged_results.append(merged_row)
                        break
                else:
                    for col in df_noise.columns:
                        if col != 'image_id':
                            row_rec_dict[col + '_noise'] = np.nan
                    merged_results.append(row_rec_dict)

            # Find unmatched org/noise rows
            matched_org_indices = set(indices_org.flatten()) if len(org_coords) > 0 else set()
            matched_noise_indices = set(indices_noise.flatten()) if len(noise_coords) > 0 else set()

            unmatched_org_indices = set(range(len(df_org_filtered))) - matched_org_indices
            unmatched_noise_indices = set(range(len(df_noise_second_filtered))) - matched_noise_indices

            # Process unmatched org rows
            for index in unmatched_org_indices:
                row_org = df_org_filtered.iloc[index]
                new_row = row_org.rename(lambda x: x + '_org' if x != 'image_id' else x).to_dict()
                for col in df_noise.columns:
                    if col != 'image_id':
                        new_row[col + '_noise'] = np.nan
                for col in df_rec.columns:
                    if col != 'image_id':
                        new_row[col + '_rec'] = np.nan
                merged_results.append(new_row)

            # Process unmatched noise rows
            for index in unmatched_noise_indices:
                row_noise = df_noise_second_filtered.iloc[index]
                new_row = row_noise.rename(lambda x: x + '_noise' if x != 'image_id' else x).to_dict()
                for col in df_org.columns:
                    if col != 'image_id':
                        new_row[col + '_org'] = np.nan
                for col in df_rec.columns:
                    if col != 'image_id':
                        new_row[col + '_rec'] = np.nan
                merged_results.append(new_row)

    return pd.DataFrame(merged_results)

def process(file_dir, noise_filename, org_filename, rec_filename, workers, threshold=3):
    noise_df = pd.read_csv(os.path.join(file_dir, noise_filename))
    org_df = pd.read_csv(os.path.join(file_dir, org_filename))
    rec_df = pd.read_csv(os.path.join(file_dir, rec_filename))

    unique_image_ids = list(set(org_df['image_id'].unique()).union(set(rec_df['image_id'].unique()), set(noise_df['image_id'].unique())))
    chunks = np.array_split(unique_image_ids, workers)
    all_data = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for chunk in chunks:
            ids = set(chunk)  # Fixed
            chunk_org_df = org_df[org_df['image_id'].isin(ids)].reset_index(drop=True)
            chunk_rec_df = rec_df[rec_df['image_id'].isin(ids)].reset_index(drop=True)
            chunk_noise_df = noise_df[noise_df['image_id'].isin(ids)].reset_index(drop=True)
            
            futures.append(executor.submit(merge_based_on_proximity, chunk_rec_df, chunk_noise_df, chunk_org_df, threshold))

    for future in as_completed(futures):
        try:
            result = future.result()
            if result is not None:
                all_data.append(result)
        except Exception as err:
            print(f"Error processing chunk: {err}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_parquet(os.path.join(file_dir, 'phometrical_data.parquet'), index=False)

if __name__ == '__main__':
    directory = sys.argv[1]
    os.chdir(os.path.dirname(__file__))
    process(directory, 'noise_catalog.csv', 'org_catalog.csv', 'rec_catalog.csv', workers=16, threshold=3)
