import os, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import binned_statistic

filename = 'phometrical_data.parquet'
output_dir = './final_plots'
data_dir = './gal_data'
metadata_filename = 'metadata_filepath_enriched_with_noise.csv'
metrics_filename = 'results_galdata.csv'

paths = [
    './on_z_scale_gal_data_latest_latest_gal_data_on_gal_data',
    './off_z_scale_gal_data_latest_latest_gal_data_on_gal_data',
    './off_normal_scale_gal_data_latest_latest_gal_data_on_gal_data',

    './on_z_scale_normal_data_latest_latest_normal_data_on_gal_data',
    './off_z_scale_normal_data_latest_latest_normal_data_on_gal_data',
    './off_normal_scale_normal_data_latest_latest_normal_data_on_gal_data',
    
    './on_z_scale_1e4_random_gal_data_3000_8_6_2-25_2_faber_data',
    './off_z_scale_1e4_random_gal_data_3000_8_6_2-25_2_faber_data',
    './off_no_scale_1e4_random_gal_data_3000_8_6_2-25_2_faber_data',

    './off_new_no_scale_gal_data_8_6_2-25_2_faber_data',
    './off_new_z_scale_gal_data_8_6_2-25_2_faber_data',
    './on_new_z_scale_gal_data_8_6_2-25_2_faber_data'
]
paths = [os.path.join(path, filename) for path in paths]

tags = [
    'on_z_scale_latest_gal_data_on_gal_data',
    'off_z_scale_latest_gal_data_on_gal_data',
    'off_no_scale_latest_gal_data_on_gal_data',

    'on_z_scale_latest_normal_data_on_gal_data',
    'off_z_scale_latest_normal_data_on_gal_data',
    'off_no_scale_latest_normal_data_on_gal_data',

    'on_z_scale_gal_data_3000_gal_data',
    'off_z_scale_gal_data_3000_gal_data',
    'off_no_scale_gal_data_3000_gal_data',

    'off_new_no_scale_gal_data_gal_data',
    'off_new_z_scale_gal_data_gal_data',
    'on_new_z_scale_gal_data_gal_data'
]

def create_table(data_dir, metadata_filename, file_dir, filename):
    df = pd.read_csv(f'{data_dir}/{metadata_filename}')
    my_set = set(df.loc[df['location'].str.contains('eval|test', na=False), 'sci_data_set_name'].unique())

    df = pd.read_csv(os.path.join(file_dir, filename))
    df['dataset'] = df['image_id'].apply(lambda x: x.split('_')[0])
    df = df[df['dataset'].isin(my_set)]
    df['IOU_sum'] = df['IoU']*df['union']
    df['SNR_org_sum'] = df['SNR_org']*df['TP']
    df['SNR_rec_sum'] = df['SNR_rec']*df['TP']
    df['RFE_sum'] = df['RFE']*df['TP']
    df['exp_ratio'] = df['org_exp_time'] / df['new_exp_time']

    summarized_mosaic = df.groupby(['dataset', 'new_exp_time']).agg({
        'TP': 'sum', 
        'FP': 'sum', 
        'FN': 'sum',  
        'IOU_sum': 'sum',  
        'SNR_org_sum': 'sum', 
        'SNR_rec_sum': 'sum',  
        'union': 'sum',
        'org_exp_time' : 'mean',
        'RFE_sum' : 'sum',
        'PSNR_rec' : 'max',
        'PSNR_noisy' : "max",
        'SSIM_rec' : "mean",
        'SSIM_noisy' : 'mean'
    }).reset_index()

    summarized_mosaic['Precision'] = summarized_mosaic['TP'] / (summarized_mosaic['TP'] + summarized_mosaic['FP'])
    summarized_mosaic['Recall'] = summarized_mosaic['TP'] / (summarized_mosaic['TP'] + summarized_mosaic['FN'])
    summarized_mosaic['F-measure'] = 2 * (summarized_mosaic['Precision'] * summarized_mosaic['Recall']) / (summarized_mosaic['Precision'] + summarized_mosaic['Recall'])
    summarized_mosaic['IoU'] = summarized_mosaic['IOU_sum']/summarized_mosaic['union']
    summarized_mosaic['SNR_org'] = summarized_mosaic['SNR_org_sum']/summarized_mosaic['TP']
    summarized_mosaic['SNR_rec'] = summarized_mosaic['SNR_rec_sum']/summarized_mosaic['TP']
    summarized_mosaic['SNR_f'] = summarized_mosaic['SNR_org']/summarized_mosaic['SNR_rec']
    summarized_mosaic['RFE'] = summarized_mosaic['RFE_sum']/summarized_mosaic['TP']
    summarized_mosaic['exp_ratio'] = summarized_mosaic['org_exp_time'] / summarized_mosaic['new_exp_time']

    summarized_mosaic['IOU_sum'] = summarized_mosaic['IoU']*summarized_mosaic['union']
    summarized_mosaic['SNR_org_sum'] = summarized_mosaic['SNR_org']*summarized_mosaic['TP']
    summarized_mosaic['SNR_rec_sum'] = summarized_mosaic['SNR_rec']*summarized_mosaic['TP']
    summarized_mosaic['RFE_sum'] = summarized_mosaic['RFE']*summarized_mosaic['TP']
    summarized_mosaic['exp_ratio'] = np.ceil((summarized_mosaic['org_exp_time'] / summarized_mosaic['new_exp_time'])).astype(int)

    summarized = summarized_mosaic.groupby(['exp_ratio']).agg({
        'TP': 'sum', 
        'FP': 'sum', 
        'FN': 'sum',  
        'IOU_sum': 'sum',  
        'SNR_org_sum': 'sum', 
        'SNR_rec_sum': 'sum',  
        'union': 'sum',#,  
        'org_exp_time' : 'mean',
        'new_exp_time' : 'mean',
        'RFE_sum' : 'sum',
        'PSNR_rec' : 'mean',
        'PSNR_noisy' : "mean",
        'SSIM_rec' : "mean",
        'SSIM_noisy' : 'mean'
    }).reset_index()

    summarized['Precision'] = summarized['TP'] / (summarized['TP'] + summarized['FP'])
    summarized['Recall'] = summarized['TP'] / (summarized['TP'] + summarized['FN'])
    summarized['F-measure'] = 2 * (summarized['Precision'] * summarized['Recall']) / (summarized['Precision'] + summarized['Recall'])
    summarized['IoU'] = summarized['IOU_sum']/summarized['union']
    summarized['SNR_org'] = summarized['SNR_org_sum']/summarized['TP']
    summarized['SNR_rec'] = summarized['SNR_rec_sum']/summarized['TP']
    summarized['SNR_f'] = summarized['SNR_org']/summarized['SNR_rec']
    summarized['RFE'] = summarized['RFE_sum']/summarized['TP']

    part_1 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum'], axis=1).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True)

    summarized_mosaic['temp'] = 1
    summarized = summarized_mosaic.groupby(['temp']).agg({
        'TP': 'sum', 
        'FP': 'sum', 
        'FN': 'sum',  
        'IOU_sum': 'sum',  
        'SNR_org_sum': 'sum', 
        'SNR_rec_sum': 'sum',  
        'union': 'sum',#,  
        'org_exp_time' : 'mean',
        'new_exp_time' : 'mean',
        'RFE_sum' : 'sum',
        'PSNR_rec' : 'mean',
        'PSNR_noisy' : "mean",
        'SSIM_rec' : "mean",
        'SSIM_noisy' : 'mean'
    }).reset_index()

    summarized['Precision'] = summarized['TP'] / (summarized['TP'] + summarized['FP'])
    summarized['Recall'] = summarized['TP'] / (summarized['TP'] + summarized['FN'])
    summarized['F-measure'] = 2 * (summarized['Precision'] * summarized['Recall']) / (summarized['Precision'] + summarized['Recall'])
    summarized['IoU'] = summarized['IOU_sum']/summarized['union']
    summarized['SNR_org'] = summarized['SNR_org_sum']/summarized['TP']
    summarized['SNR_rec'] = summarized['SNR_rec_sum']/summarized['TP']
    summarized['SNR_f'] = summarized['SNR_org']/summarized['SNR_rec']
    summarized['RFE'] = summarized['RFE_sum']/summarized['TP']
    part_2 = summarized.drop(labels=['IOU_sum', 'SNR_org_sum', 'SNR_rec_sum', 'union', 'RFE_sum', 'temp'], axis=1)
    part_2['exp_ratio'] = 0
    final_table = pd.concat([part_1, part_2], ignore_index=True).sort_values(by=['exp_ratio'], ascending=False).reset_index(drop=True)
    return final_table

def convert_to_jansky(flux_e_per_s, PHOTPLAM=15369.17570896557, 
                      PHOTFLAM =1.92756031304868e-20):
    #via f_{\nu} = \frac{\lambda^2}{c}\cdot f_{\lambda}
    factor = 3.33564e4
    return flux_e_per_s*PHOTFLAM*PHOTPLAM**2*factor

def convert_to_angstrom(flux_e_per_s, PHOTFLAM =1.92756031304868e-20):
    return PHOTFLAM*flux_e_per_s

def calculate_abmag(flux_jansky):
    return -2.5*np.log10(flux_jansky) + 8.9

def add_columns(df):
    df['exp_ratio'] = df.apply(
        lambda row: row['exp_time_rec'] / row['new_exp_time_rec']
        if not pd.isna(row['exp_time_rec']) else
        (row['exp_time_noise'] / row['new_exp_time_noise'] if not pd.isna(row['exp_time_noise']) else 0.0),
        axis=1
    )
    df['exp_ratio'] = df['exp_ratio'].round(2)

    for col in ['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec', 'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise',
                'flux_err_org', 'flux_err_rec', 'flux_err_noise']:
        df['j_' + col] = df[col].apply(convert_to_jansky)
        if 'err' not in col:
            df['abmag_' + col] = df['j_' + col].apply(calculate_abmag)
        df['a_' + col] = df['j_' + col].apply(convert_to_angstrom)

    for flux, flux_err in zip(['flux_x_org', 'flux_y_org', 'flux_x_rec', 'flux_y_rec',
                               'flux_x_noise', 'flux_y_noise', 'cflux_org', 'cflux_rec', 'cflux_noise'],
                              ['flux_err_org', 'flux_err_org', 'flux_err_rec', 'flux_err_rec',
                                'flux_err_noise', 'flux_err_noise', 'flux_err_org', 'flux_err_org', 'flux_err_rec']):
        df['delta_' + flux]= df[flux]/df[flux_err] 
        df['delta_j_' + flux] = df['j_' + flux]/df['j_' + flux_err] 
        df['delta_a_' + flux] = df['a_' + flux]/df['a_' + flux_err] 
    return df
#Kron to Kron (2)
def create_only_rec_kron_plots(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, output_dir, filename):
    cmap_rec = plt.get_cmap(rec_cmap)
    x_max = 8

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    j = 0

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"

        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        ax = axes[i, j]

        # Ensure valid data (No NaNs)
        valid_mask = sub_df[noise_flux_col].isna() & sub_df[org_flux_col].notna() & sub_df[rec_flux_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        vmin = min(log_x_rec.min(), log_y_rec.min())
        vmax = max(log_x_rec.max(), log_y_rec.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        gridsize = max(100, int(np.sqrt(len(log_x_rec)/10)))
        #gridsize = 10

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        a, b = np.percentile(temp_df['y'], [1, 98])
        c, d = np.percentile(temp_df['x'], [1, 99])
        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['x'] >= c) & (temp_df['x'] <= d)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        start = log_x_rec.min()
        end = log_x_rec.max()
        one_to_one = np.linspace(start, end, 1000)

        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        # **Use hexbin plot**
        hexbin = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin, ax=ax)

        if i == 1:
            ax.set_xlabel(r'Org. Kron Radius $[arcsec]$')
        if j == 0:
            ax.set_ylabel(r'Rec. Kron Radius $[arcsec]$')
        ax.grid(True)

        # Add legend with data count
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='lower right')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        if i == 1:
            j += 1

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec. Kron Radius vs. Org. Kron Radius (Detected only in Rec.)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=250)
    ###plt.show()
    plt.close()
def create_only_rec_flux_plots(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, output_dir, filename):
    cmap_rec = plt.get_cmap(rec_cmap)
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    j = 0

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"

        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        ax = axes[i, j]

        # Ensure valid data (No NaNs)
        valid_mask = sub_df[noise_flux_col].isna() & sub_df[org_flux_col].notna() & sub_df[rec_flux_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        
        gridsize = max(100, int(np.sqrt(len(log_x_rec)/10)))

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        a, b = np.percentile(temp_df['y'], [1, 99])
        c, d = np.percentile(temp_df['x'], [1, 99])
        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['x'] >= c) & (temp_df['x'] <= d)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        start = log_x_rec.min()
        end = log_x_rec.max()

        one_to_one = np.linspace(start, end, 1000)
        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        # **Use hexbin plot**
        hexbin = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize, cmap=rec_cmap, mincnt=2, bins='log', linewidths=0.1)
        
        cbar = fig.colorbar(hexbin, ax=ax)

        if i == 1:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')   
        if j == 0:
            ax.set_ylabel(r'Rec. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')

        ax.grid(True)

        # Add legend with data count
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='lower right')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        if i == 1:
            j += 1

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec. $\log($Flux) vs. Org. $\log($Flux) (Detected only in Rec.)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=250)
    ###plt.show()
    plt.close()

## General Functions
def create_flux_plots(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    try:
        cmap_rec = plt.get_cmap(rec_cmap)
        cmap_noise = plt.get_cmap(noise_cmap)
    except ValueError:
        print(f"Invalid colormap names: {rec_cmap}, {noise_cmap}")
        return
    
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_flux_col].notna()][org_flux_col], [1, 98])
        sub_df = sub_df[(sub_df[org_flux_col] >= c) & (sub_df[org_flux_col] <= d)]
        valid_mask = ~sub_df[rec_flux_col].isna() & sub_df[org_flux_col].notna() 
        log_x_rec = np.log10(sub_df.loc[valid_mask, org_flux_col])
        log_y_rec = np.log10(sub_df.loc[valid_mask, rec_flux_col])

        valid_mask = ~sub_df[noise_flux_col].isna() & sub_df[org_flux_col].notna() 
        log_x_noise = np.log10(sub_df.loc[valid_mask, org_flux_col])
        log_y_noise = np.log10(sub_df.loc[valid_mask, noise_flux_col])

        N_noise = len(log_x_noise)
        N_rec = len(log_x_rec)
        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        temp_df_2 = pd.DataFrame({'x': log_x_noise, 'y': log_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df_2['y'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)
        c = np.log10(c)
        d = np.log10(d)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']

        temp_df_2 = temp_df_2[(temp_df_2['y'] >= a) & (temp_df_2['y'] <= b)]
        log_x_noise = temp_df_2['x']
        log_y_noise = temp_df_2['y']

        # Ensure the arrays are not empty
        if len(log_x_rec) == 0 or len(log_y_rec) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue
        if len(log_x_noise) == 0 or len(log_y_noise) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_rec)/10)))

        left = log_x_rec.min()
        right = log_x_rec.max()
        one_to_one = np.linspace(left, right, 1000)

        # Plot Reconstructed Image
        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_rec = fig.colorbar(hexbin_rec, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Reconstructed Image')
  
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        ax.set_ylabel(r'Rec. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')

        # Add N (count of data points) to the legend
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(np.median(log_x_rec))), lw=4, label=f'{label}, N={N_rec}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        # Plot Noisy Image
        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_noise = fig.colorbar(hexbin_noise, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Noisy Image')
  
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        ax.set_ylabel(r'Noisy Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')

        # Add N (count of data points) to the legend
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(np.median(log_x_noise))), lw=4, label=f'{label}, N={N_noise}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy $\log($Flux) vs. Org. $\log($Flux)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()
def create_kron_radii_plots(df, kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    cmap_rec = plt.get_cmap(rec_cmap)
    cmap_noise = plt.get_cmap(noise_cmap)

    norm = Normalize(vmin=df[kron_radius_org_col].min(), vmax=df[kron_radius_org_col].max())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[kron_radius_org_col].notna()][kron_radius_org_col], [1, 98])
        sub_df = sub_df[(sub_df[kron_radius_org_col] >= c) & (sub_df[kron_radius_org_col] <= d)]
        valid_mask = ~sub_df[kron_radius_rec_col].isna() & sub_df[kron_radius_org_col].notna() 
        log_x_rec = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_rec = sub_df.loc[valid_mask, kron_radius_rec_col]

        valid_mask = ~sub_df[kron_radius_noise_col].isna() & sub_df[kron_radius_org_col].notna() 
        log_x_noise = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_noise = sub_df.loc[valid_mask, kron_radius_noise_col]

        gridsize_rec = max(100, int(np.sqrt(len(log_y_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_y_noise)/10)))

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        temp_df_2 = pd.DataFrame({'x': log_x_noise, 'y': log_y_noise})
        
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df['y'], [1, 99])
        a = min(a, a_2)
        b = max(b, b_2)

        N_noise = len(log_x_noise)
        N_rec = len(log_x_rec)
        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']

        temp_df_2 = temp_df_2[(temp_df_2['y'] >= a) & (temp_df_2['y'] <= b)]
        log_x_noise = temp_df_2['x']
        log_y_noise = temp_df_2['y']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_rec, ax=ax)

        if j_rec == 0:
            ax.set_ylabel(r'Kron Radius $[arcsec]$')
        if i == 3:
            ax.set_xlabel(r'Org. Kron Radius $[arcsec]$')
        if i == 0:
            ax.set_title('Reconstructed Image')
            
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={N_rec}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_noise, ax=ax)

        if i == 3:
            ax.set_xlabel(r'$[arcsec]$')
        if i == 0:
            ax.set_title('Noisy Image')
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(log_x_noise.min())), lw=4, label=f'{label}, N={N_noise}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy Kron Radius vs. Org. Kron Radius", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()

## General Functions for Both detections
def create_flux_plots_both_det(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    try:
        cmap_rec = plt.get_cmap(rec_cmap)
        cmap_noise = plt.get_cmap(noise_cmap)
    except ValueError:
        print(f"Invalid colormap names: {rec_cmap}, {noise_cmap}")
        return
    
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_flux_col].notna()][org_flux_col], [1, 98])
        sub_df = sub_df[(sub_df[org_flux_col] >= c) & (sub_df[org_flux_col] <= d)]
        valid_mask = sub_df[rec_flux_col].notna() & sub_df[org_flux_col].notna() & sub_df[noise_flux_col].notna()
        log_x_rec = np.log10(sub_df.loc[valid_mask, org_flux_col])
        log_y_rec = np.log10(sub_df.loc[valid_mask, rec_flux_col])

        log_x_noise = np.log10(sub_df.loc[valid_mask, org_flux_col])
        log_y_noise = np.log10(sub_df.loc[valid_mask, noise_flux_col])

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec, 'x_2':log_x_noise, 'y_2':log_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df['y_2'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)
        c = np.log10(c)
        d = np.log10(d)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['y_2'] >= a) & (temp_df['y_2'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        log_x_noise = temp_df['x_2']
        log_y_noise = temp_df['y_2']

        # Ensure the arrays are not empty
        if len(log_x_rec) == 0 or len(log_y_rec) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue
        if len(log_x_noise) == 0 or len(log_y_noise) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_rec)/10)))

        left = log_x_rec.min()
        right = log_x_rec.max()
        one_to_one = np.linspace(left, right, 1000)

        # Plot Reconstructed Image
        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_rec = fig.colorbar(hexbin_rec, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Reconstructed Image')
  
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        ax.set_ylabel(r'Rec. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')

        # Add N (count of data points) to the legend
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(np.median(log_x_rec))), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        # Plot Noisy Image
        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_noise = fig.colorbar(hexbin_noise, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Noisy Image')
  
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        ax.set_ylabel(r'Noisy Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')

        # Add N (count of data points) to the legend
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(np.median(log_x_noise))), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy $\log($Flux) vs. Org. $\log($Flux) (Detected in Both)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()
def create_kron_radii_plots_both_det(df, kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    cmap_rec = plt.get_cmap(rec_cmap)
    cmap_noise = plt.get_cmap(noise_cmap)

    norm = Normalize(vmin=df[kron_radius_org_col].min(), vmax=df[kron_radius_org_col].max())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[kron_radius_org_col].notna()][kron_radius_org_col], [1, 98])
        sub_df = sub_df[(sub_df[kron_radius_org_col] >= c) & (sub_df[kron_radius_org_col] <= d)]
        valid_mask = sub_df[kron_radius_rec_col].notna() & sub_df[kron_radius_org_col].notna() & sub_df[kron_radius_noise_col].notna()
        log_x_rec = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_rec = sub_df.loc[valid_mask, kron_radius_rec_col]

        log_x_noise = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_noise = sub_df.loc[valid_mask, kron_radius_noise_col]

        gridsize_rec = max(100, int(np.sqrt(len(log_y_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_y_noise)/10)))

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec, 'x_2':log_x_noise, 'y_2':log_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df['y_2'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['y_2'] >= a) & (temp_df['y_2'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        log_x_noise = temp_df['x_2']
        log_y_noise = temp_df['y_2']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_rec, ax=ax)

        if j_rec == 0:
            ax.set_ylabel(r'Kron Radius $[arcsec]$')
        if i == 3:
            ax.set_xlabel(r'Org. Kron Radius $[arcsec]$')
        if i == 0:
            ax.set_title('Reconstructed Image')
            
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_noise, ax=ax)

        if i == 3:
            ax.set_xlabel(r'$[arcsec]$')
        if i == 0:
            ax.set_title('Noisy Image')
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(log_x_noise.min())), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy Kron Radius vs. Org. Kron Radius (Detected in Both)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()

##Only Detections Captured in the Rec as a function of ABMAG (Delta)
def create_only_delta_flux_mag_plots(df, org_flux_col, rec_flux_col, noise_flux_col, org_mag_col, rec_cmap, output_dir, filename):
    cmap_rec = plt.get_cmap(rec_cmap)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    j = 0

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"

        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        ax = axes[i, j]

        # Ensure valid data (No NaNs)
        valid_mask = sub_df[org_flux_col].notna() & sub_df[rec_flux_col].notna() & sub_df[org_mag_col].notna() & sub_df[noise_flux_col].isna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col] 
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        log_y_rec = log_y_rec/log_x_rec
        log_x_rec = sub_df.loc[valid_mask, org_mag_col]

        vmin = min(log_x_rec.min(), log_y_rec.min())
        vmax = max(log_x_rec.max(), log_y_rec.max())
        norm = Normalize(vmin=vmin, vmax=vmax)

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        a, b = np.percentile(temp_df['y'], [1, 99])
        c, d = np.percentile(temp_df['x'], [1, 99])
        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['x'] >= c) & (temp_df['x'] <= d)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        start = log_x_rec.min()
        end = log_x_rec.max()
        gridsize = max(100, int(np.sqrt(len(log_x_rec)/10)))
        #gridsize = 10

        one_to_one = np.linspace(start, end, 10000)
        ax.set_xlim([22.7, log_x_rec.max()])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)

        # **Use hexbin plot**
        hexbin = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin, ax=ax)

        if i == 1:
            ax.set_xlabel(r'AB MAG')
        if j == 0:
            ax.set_ylabel(r'Flux Ratio')
        ax.grid(True)

        # Add legend with data count
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper right')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        if i == 1:
            j += 1

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec. Flux Ratio vs. AB MAG (Detected only in Rec.)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=250)
    ###plt.show()
    plt.close()
def create_only_delta_kron_mag_plots(df, org_flux_col, rec_flux_col, noise_flux_col, org_mag_col, rec_cmap, output_dir, filename):
    cmap_rec = plt.get_cmap(rec_cmap)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    j = 0

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"

        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        ax = axes[i, j]

        # Ensure valid data (No NaNs)
        valid_mask = sub_df[org_flux_col].notna() & sub_df[rec_flux_col].notna() & sub_df[org_mag_col].notna() & sub_df[noise_flux_col].isna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col] 
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        log_y_rec -= log_x_rec
        log_y_rec = log_y_rec
        log_x_rec = sub_df.loc[valid_mask, org_mag_col]

        vmin = min(log_x_rec.min(), log_y_rec.min())
        vmax = max(log_x_rec.max(), log_y_rec.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        a, b = np.percentile(temp_df['y'], [1, 99])
        c, d = np.percentile(temp_df['x'], [1, 99])
        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['x'] >= c) & (temp_df['x'] <= d)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        start = log_x_rec.min()
        end = log_x_rec.max()

        gridsize = max(100, int(np.sqrt(len(log_x_rec)/10)))
        #gridsize = 10

        one_to_one = np.linspace(start, end, 10000)
        ax.set_xlim([22.7, log_x_rec.max()])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)

        # **Use hexbin plot**
        hexbin = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin, ax=ax)

        if i == 1:
            ax.set_xlabel(r'AB MAG')
        if j == 0:
            ax.set_ylabel(r'$\Delta$Kron Radius $[arcsec]$')
        ax.grid(True)

        # Add legend with data count
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper right')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        if i == 1:
            j += 1

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec. $\Delta$Kron Radius vs. AB MAG (Detected only in Rec.)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=250)
    ###plt.show()
    plt.close()

##Only Detections Captured in the Rec as a  function of their respective quantity (Delta)
def create_only_delta_rec_flux_plots(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, output_dir, filename):
    cmap_rec = plt.get_cmap(rec_cmap)
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    j = 0

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"

        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        ax = axes[i, j]

        # Ensure valid data (No NaNs)
        valid_mask = sub_df[noise_flux_col].isna() & sub_df[org_flux_col].notna() & sub_df[rec_flux_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        log_y_rec = log_y_rec/log_x_rec
        log_x_rec = np.log10(log_x_rec)
        
        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        a, b = np.percentile(temp_df['y'], [1, 99])
        c, d = np.percentile(temp_df['x'], [1, 99])
        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['x'] >= c) & (temp_df['x'] <= d)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        start = log_x_rec.min()
        end = log_x_rec.max()

        gridsize = max(100, int(np.sqrt(len(log_x_rec)/10)))

        # Plot the area between start and end x-values
        one_to_one = np.linspace(start, end, 10000)
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        # **Use hexbin plot**
        hexbin = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize, cmap=rec_cmap, mincnt=2, bins='log', linewidths=0.1)
        cbar = fig.colorbar(hexbin, ax=ax)

        if i == 1:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')   
        if j == 0:
            ax.set_ylabel(r'Rec. Flux Ratio')
        ax.grid(True)

        # Add legend with data count
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='lower right')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        if i == 1:
            j += 1

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec.Flux Ratio vs. Org. $\log($Flux) (Detected only in Rec.)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=250)
    ###plt.show()
    plt.close()
def create_only_delta_rec_kron_plots(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, output_dir, filename):
    cmap_rec = plt.get_cmap(rec_cmap)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    j = 0

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"

        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        ax = axes[i, j]

        # Ensure valid data (No NaNs)
        valid_mask = sub_df[noise_flux_col].isna() & sub_df[org_flux_col].notna() & sub_df[rec_flux_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        log_y_rec -= log_x_rec
        log_y_rec = log_y_rec

        vmin = min(log_x_rec.min(), log_y_rec.min())
        vmax = max(log_x_rec.max(), log_y_rec.max())
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        a, b = np.percentile(temp_df['y'], [1, 99])
        c, d = np.percentile(temp_df['x'], [1, 99])
        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['x'] >= c) & (temp_df['x'] <= d)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        start = log_x_rec.min()
        end = log_x_rec.max()

        start = np.min(log_x_rec)
        end = np.max(log_x_rec)  

        gridsize = max(100, int(np.sqrt(len(log_x_rec)/10)))
        #gridsize = 10

        one_to_one = np.linspace(start, end, 10000)
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)

        # **Use hexbin plot**
        hexbin = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin, ax=ax)

        if i == 1:
            ax.set_xlabel(r'Org. Kron Radius $[arcsec]$')
        if j == 0:
            ax.set_ylabel(r'Rec. Kron Radius $[arcsec]$')
        ax.grid(True)

        # Add legend with data count
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='lower right')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        if i == 1:
            j += 1

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec. $\Delta$Kron Radius vs. Org. Kron Radius (Detected only in Rec.)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=250)
    ###plt.show()
    plt.close()

##For Detections Captured both in the Rec and Noisy as a their respective quantity (Delta)
def create_delta_flux_plots(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    try:
        cmap_rec = plt.get_cmap(rec_cmap)
        cmap_noise = plt.get_cmap(noise_cmap)
    except ValueError:
        print(f"Invalid colormap names: {rec_cmap}, {noise_cmap}")
        return
    
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_flux_col].notna()][org_flux_col], [1, 98])
        sub_df = sub_df[(sub_df[org_flux_col] >= c) & (sub_df[org_flux_col] <= d)]
        valid_mask = sub_df[rec_flux_col].notna() & sub_df[org_flux_col].notna() & sub_df[noise_flux_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        delta_y_rec = log_y_rec/log_x_rec
        log_x_rec = np.log10(log_x_rec)

        log_x_noise = sub_df.loc[valid_mask, org_flux_col]
        log_y_noise = sub_df.loc[valid_mask, noise_flux_col]
        delta_y_noise = log_y_noise/log_x_noise
        log_x_noise = np.log10(log_x_noise)

        # Ensure the arrays are not empty
        if len(log_x_rec) == 0 or len(log_y_rec) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue
        if len(log_x_noise) == 0 or len(log_y_noise) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue

        gridsize_rec = max(100, int(np.sqrt(len(delta_y_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(delta_y_noise)/10)))

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': delta_y_rec, 'x_2':log_x_noise, 'y_2':delta_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df['y_2'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)
        c = np.log10(c)
        d = np.log10(d)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['y_2'] >= a) & (temp_df['y_2'] <= b)]
        log_x_rec = temp_df['x']
        delta_y_rec = temp_df['y']
        log_x_noise = temp_df['x_2']
        delta_y_noise = temp_df['y_2']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        # Plot Reconstructed Image
        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, delta_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_rec = fig.colorbar(hexbin_rec, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Reconstructed Image')
 
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        if j_rec == 0:
            ax.set_ylabel(r'Flux Ratio')

        # Add N (count of data points) to the legend
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(np.median(log_x_rec))), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        # Plot Noisy Image
        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, delta_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_noise = fig.colorbar(hexbin_noise, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Noisy Image')
 
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')

        # Add N (count of data points) to the legend
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(np.median(log_x_noise))), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Flux Ratio vs. $\log$(Flux) (Detected in Both)", fontsize=20)
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()
def create_delta_kron_radii_plots(df, kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    cmap_rec = plt.get_cmap(rec_cmap)
    cmap_noise = plt.get_cmap(noise_cmap)

    norm = Normalize(vmin=df[kron_radius_org_col].min(), vmax=df[kron_radius_org_col].max())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[kron_radius_org_col].notna()][kron_radius_org_col], [1, 98])
        sub_df = sub_df[(sub_df[kron_radius_org_col] >= c) & (sub_df[kron_radius_org_col] <= d)]
        valid_mask = sub_df[kron_radius_rec_col].notna() & sub_df[kron_radius_org_col].notna() & sub_df[kron_radius_noise_col].notna()
        log_x_rec = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_rec = sub_df.loc[valid_mask, kron_radius_rec_col]
        log_y_rec -= log_x_rec
        log_y_rec = log_y_rec

        log_x_noise = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_noise = sub_df.loc[valid_mask, kron_radius_noise_col]
        log_y_noise -= log_x_noise
        log_y_noise = log_y_noise

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec, 'x_2':log_x_noise, 'y_2':log_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df['y_2'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['y_2'] >= a) & (temp_df['y_2'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        log_x_noise = temp_df['x_2']
        log_y_noise = temp_df['y_2']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_noise)/10)))

        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_rec, ax=ax)

        if j_rec == 0:
            ax.set_ylabel(r'$\Delta$Kron Radius $[arcsec]$')
        if i == 3:
            ax.set_xlabel(r'Kron Radius $[arcsec]$')
        if i == 0:
            ax.set_title('Reconstructed Image')
            
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_noise, ax=ax)

        if i == 3:
            ax.set_xlabel(r'$[arcsec]$')
        if i == 0:
            ax.set_title('Noisy Image')
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(log_x_noise.min())), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy $\Delta$Kron Radius vs. Org. Kron Radius (Detected in Both)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()

##For Detections Captured both in the Rec and Noisy as a function of ABMAG (Delta)
def create_delta_flux_mag_plots(df, org_flux_col, rec_flux_col, noise_flux_col, org_mag_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    try:
        cmap_rec = plt.get_cmap(rec_cmap)
        cmap_noise = plt.get_cmap(noise_cmap)
    except ValueError:
        print(f"Invalid colormap names: {rec_cmap}, {noise_cmap}")
        return
    
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_mag_col].notna()][org_mag_col], [1, 98])
        sub_df = sub_df[(sub_df[org_mag_col] >= c) & (sub_df[org_mag_col] <= d)]
        valid_mask = ~sub_df[rec_flux_col].isna() & sub_df[org_flux_col].notna() & sub_df[org_mag_col].notna() & sub_df[noise_flux_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        delta_y_rec = log_y_rec/log_x_rec
        log_x_rec = sub_df.loc[valid_mask, org_mag_col]

        log_x_noise = sub_df.loc[valid_mask, org_flux_col]
        log_y_noise = sub_df.loc[valid_mask, noise_flux_col]
        delta_y_noise = log_y_noise/log_x_noise
        log_x_noise = sub_df.loc[valid_mask, org_mag_col]

        # Ensure the arrays are not empty
        if len(log_x_rec) == 0 or len(log_y_rec) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue
        if len(log_x_noise) == 0 or len(log_y_noise) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': delta_y_rec, 'x_2':log_x_noise, 'y_2':delta_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df['y_2'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['y_2'] >= a) & (temp_df['y_2'] <= b)]
        log_x_rec = temp_df['x']
        delta_y_rec = temp_df['y']
        log_x_noise = temp_df['x_2']
        delta_y_noise = temp_df['y_2']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_noise)/10)))

        # Plot Reconstructed Image
        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, delta_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_rec = fig.colorbar(hexbin_rec, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Reconstructed Image')
        if i == 3:
            ax.set_xlabel(r'AB MAG')
        if j_rec == 0:
            ax.set_ylabel(r'Flux Ratio')

        # Add N (count of data points) to the legend
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(np.median(log_x_rec))), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        # Plot Noisy Image
        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, delta_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_noise = fig.colorbar(hexbin_noise, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Noisy Image')
 
        if i == 3:
            ax.set_xlabel(r'AB MAG')

        # Add N (count of data points) to the legend
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(np.median(log_x_noise))), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Flux Ratio vs. AB MAG (Detected in Both)", fontsize=20)
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()
def create_delta_kron_radii_mag_plots(df, kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, org_mag_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    cmap_rec = plt.get_cmap(rec_cmap)
    cmap_noise = plt.get_cmap(noise_cmap)

    norm = Normalize(vmin=df[kron_radius_org_col].min(), vmax=df[kron_radius_org_col].max())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_mag_col].notna()][org_mag_col], [1, 98])
        sub_df = sub_df[(sub_df[org_mag_col] >= c) & (sub_df[org_mag_col] <= d)]
        valid_mask = sub_df[kron_radius_rec_col].notna() & sub_df[kron_radius_org_col].notna() & sub_df[org_mag_col].notna() & sub_df[kron_radius_noise_col].notna()
        log_x_rec = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_rec = sub_df.loc[valid_mask, kron_radius_rec_col]
        log_y_rec -= log_x_rec
        log_y_rec = log_y_rec
        log_x_rec = sub_df.loc[valid_mask, org_mag_col]

        log_x_noise = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_noise = sub_df.loc[valid_mask, kron_radius_noise_col]
        log_y_noise -= log_x_noise
        log_y_noise = log_y_noise
        log_x_noise = sub_df.loc[valid_mask, org_mag_col]

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec, 'x_2':log_x_noise, 'y_2':log_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df['y_2'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b) & (temp_df['y_2'] >= a) & (temp_df['y_2'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']
        log_x_noise = temp_df['x_2']
        log_y_noise = temp_df['y_2']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_noise)/10)))

        ax = axes[i, j_rec]
        ax.set_box_aspect(1)
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_rec, ax=ax)

        if j_rec == 0:
            ax.set_ylabel(r' $\Delta$Kron Radius $[arcsec]$')
        if i == 3:
            ax.set_xlabel(r'AB MAG')
        if i == 0:
            ax.set_title('Reconstructed Image')
            
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        ax = axes[i, j_noise]
        ax.set_box_aspect(1)
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_noise, ax=ax)

        if i == 3:
            ax.set_xlabel(r'AB MAG')
        if i == 0:
            ax.set_title('Noisy Image')
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(log_x_noise.min())), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy $\Delta$Kron Radius vs. AB MAG (Detected in Both)", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()

##For Detections combined as a their respective quantity (Delta)
def create_delta_flux_plots_combined(df, org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    try:
        cmap_rec = plt.get_cmap(rec_cmap)
        cmap_noise = plt.get_cmap(noise_cmap)
    except ValueError:
        print(f"Invalid colormap names: {rec_cmap}, {noise_cmap}")
        return
    
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_flux_col].notna()][org_flux_col], [1, 98])
        sub_df = sub_df[(sub_df[org_flux_col] >= c) & (sub_df[org_flux_col] <= d)]
        valid_mask = sub_df[rec_flux_col].notna() & sub_df[org_flux_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        delta_y_rec = log_y_rec/log_x_rec
        log_x_rec = np.log10(log_x_rec)

        valid_mask = sub_df[org_flux_col].notna() & sub_df[noise_flux_col].notna()
        log_x_noise = sub_df.loc[valid_mask, org_flux_col]
        log_y_noise = sub_df.loc[valid_mask, noise_flux_col]
        delta_y_noise = log_y_noise/log_x_noise
        log_x_noise = np.log10(log_x_noise)

        # Ensure the arrays are not empty
        if len(log_x_rec) == 0 or len(log_y_rec) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue
        if len(log_x_noise) == 0 or len(log_y_noise) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue

        gridsize_rec = max(100, int(np.sqrt(len(delta_y_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(delta_y_noise)/10)))

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': delta_y_rec})
        temp_df_2 = pd.DataFrame({'x': log_x_noise, 'y': delta_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df_2['y'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)
        c = np.log10(c)
        d = np.log10(d)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b)]
        log_x_rec = temp_df['x']
        delta_y_rec = temp_df['y']

        temp_df_2 = temp_df_2[(temp_df_2['y'] >= a) & (temp_df_2['y'] <= b)]
        log_x_noise = temp_df_2['x']
        delta_y_noise = temp_df_2['y']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        # Plot Reconstructed Image
        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, delta_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_rec = fig.colorbar(hexbin_rec, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Reconstructed Image')
 
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        if j_rec == 0:
            ax.set_ylabel(r'Flux Ratio')

        # Add N (count of data points) to the legend
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(np.median(log_x_rec))), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        # Plot Noisy Image
        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, delta_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_noise = fig.colorbar(hexbin_noise, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Noisy Image')
 
        if i == 3:
            ax.set_xlabel(r'Org. Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')

        # Add N (count of data points) to the legend
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(np.median(log_x_noise))), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Flux Ratio vs. $\log$(Flux)", fontsize=20)
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()
def create_delta_kron_radii_plots_combined(df, kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    cmap_rec = plt.get_cmap(rec_cmap)
    cmap_noise = plt.get_cmap(noise_cmap)

    norm = Normalize(vmin=df[kron_radius_org_col].min(), vmax=df[kron_radius_org_col].max())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[kron_radius_org_col].notna()][kron_radius_org_col], [1, 98])
        sub_df = sub_df[(sub_df[kron_radius_org_col] >= c) & (sub_df[kron_radius_org_col] <= d)]
        valid_mask = sub_df[kron_radius_rec_col].notna() & sub_df[kron_radius_org_col].notna()
        log_x_rec = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_rec = sub_df.loc[valid_mask, kron_radius_rec_col]
        log_y_rec -= log_x_rec
        log_y_rec = log_y_rec

        valid_mask =  sub_df[kron_radius_org_col].notna() & sub_df[kron_radius_noise_col].notna()
        log_x_noise = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_noise = sub_df.loc[valid_mask, kron_radius_noise_col]
        log_y_noise -= log_x_noise
        log_y_noise = log_y_noise

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        temp_df_2 = pd.DataFrame({'x': log_x_noise, 'y': log_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df_2['y'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']

        temp_df_2 = temp_df_2[(temp_df_2['y'] >= a) & (temp_df_2['y'] <= b)]
        log_x_noise = temp_df_2['x']
        log_y_noise = temp_df_2['y']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_noise)/10)))

        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_rec, ax=ax)

        if j_rec == 0:
            ax.set_ylabel(r'$\Delta$Kron Radius $[arcsec]$')
        if i == 3:
            ax.set_xlabel(r'Kron Radius $[arcsec]$')
        if i == 0:
            ax.set_title('Reconstructed Image')
            
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_noise, ax=ax)

        if i == 3:
            ax.set_xlabel(r'$[arcsec]$')
        if i == 0:
            ax.set_title('Noisy Image')
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(log_x_noise.min())), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy $\Delta$Kron Radius vs. Org. Kron Radius", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()

##For Detections combined as a function of ABMAG (Delta)
def create_delta_flux_mag_plots_combined(df, org_flux_col, rec_flux_col, noise_flux_col, org_mag_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    try:
        cmap_rec = plt.get_cmap(rec_cmap)
        cmap_noise = plt.get_cmap(noise_cmap)
    except ValueError:
        print(f"Invalid colormap names: {rec_cmap}, {noise_cmap}")
        return
    
    norm = Normalize(vmin=df[org_flux_col].min(), vmax=df[org_flux_col].max())

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_mag_col].notna()][org_mag_col], [1, 98])
        sub_df = sub_df[(sub_df[org_mag_col] >= c) & (sub_df[org_mag_col] <= d)]
        valid_mask = ~sub_df[rec_flux_col].isna() & sub_df[org_flux_col].notna() & sub_df[org_mag_col].notna()
        log_x_rec = sub_df.loc[valid_mask, org_flux_col]
        log_y_rec = sub_df.loc[valid_mask, rec_flux_col]
        delta_y_rec = log_y_rec/log_x_rec
        log_x_rec = sub_df.loc[valid_mask, org_mag_col]

        valid_mask = sub_df[org_flux_col].notna() & sub_df[org_mag_col].notna() & sub_df[noise_flux_col].notna()
        log_x_noise = sub_df.loc[valid_mask, org_flux_col]
        log_y_noise = sub_df.loc[valid_mask, noise_flux_col]
        delta_y_noise = log_y_noise/log_x_noise
        log_x_noise = sub_df.loc[valid_mask, org_mag_col]

        # Ensure the arrays are not empty
        if len(log_x_rec) == 0 or len(log_y_rec) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue
        if len(log_x_noise) == 0 or len(log_y_noise) == 0:
            print(f"Warning: Empty data for exp_ratio {exp_ratio}, skipping.")
            continue

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': delta_y_rec})
        temp_df_2 = pd.DataFrame({'x': log_x_noise, 'y': delta_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df_2['y'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b)]
        log_x_rec = temp_df['x']
        delta_y_rec = temp_df['y']

        temp_df_2 = temp_df_2[(temp_df_2['y'] >= a) & (temp_df_2['y'] <= b)]
        log_x_noise = temp_df_2['x']
        delta_y_noise = temp_df_2['y']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_noise)/10)))

        # Plot Reconstructed Image
        ax = axes[i, j_rec]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_rec = ax.hexbin(log_x_rec, delta_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_rec = fig.colorbar(hexbin_rec, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Reconstructed Image')
        if i == 3:
            ax.set_xlabel(r'AB MAG')
        if j_rec == 0:
            ax.set_ylabel(r'Flux Ratio')

        # Add N (count of data points) to the legend
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(np.median(log_x_rec))), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        # Plot Noisy Image
        ax = axes[i, j_noise]
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        ax.set_box_aspect(1)
        #ax.plot(one_to_one, one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, delta_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar_noise = fig.colorbar(hexbin_noise, ax=ax, norm=norm)
        if i == 0:
            ax.set_title('Noisy Image')
 
        if i == 3:
            ax.set_xlabel(r'AB MAG')

        # Add N (count of data points) to the legend
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(np.median(log_x_noise))), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Flux Ratio vs. AB MAG", fontsize=20)
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()
def create_delta_kron_radii_mag_combined(df, kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, org_mag_col, rec_cmap, noise_cmap, output_dir, filename):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    cmap_rec = plt.get_cmap(rec_cmap)
    cmap_noise = plt.get_cmap(noise_cmap)

    norm = Normalize(vmin=df[kron_radius_org_col].min(), vmax=df[kron_radius_org_col].max())
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        j_rec = 0
        j_noise = 1
        label = r"all $\gamma$'s"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'

        if index > 3:
            j_rec = 2
            j_noise = 3
        i = index % 4

        c, d = np.percentile(sub_df[sub_df[org_mag_col].notna()][org_mag_col], [1, 98])
        sub_df = sub_df[(sub_df[org_mag_col] >= c) & (sub_df[org_mag_col] <= d)]
        valid_mask = sub_df[kron_radius_rec_col].notna() & sub_df[kron_radius_org_col].notna() & sub_df[org_mag_col].notna()
        log_x_rec = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_rec = sub_df.loc[valid_mask, kron_radius_rec_col]
        log_y_rec -= log_x_rec
        log_y_rec = log_y_rec
        log_x_rec = sub_df.loc[valid_mask, org_mag_col]

        valid_mask = sub_df[kron_radius_org_col].notna() & sub_df[org_mag_col].notna() & sub_df[kron_radius_noise_col].notna()
        log_x_noise = sub_df.loc[valid_mask, kron_radius_org_col]
        log_y_noise = sub_df.loc[valid_mask, kron_radius_noise_col]
        log_y_noise -= log_x_noise
        log_y_noise = log_y_noise
        log_x_noise = sub_df.loc[valid_mask, org_mag_col]

        temp_df = pd.DataFrame({'x': log_x_rec, 'y': log_y_rec})
        temp_df_2 = pd.DataFrame({'x': log_x_noise, 'y': log_y_noise})
        a, b = np.percentile(temp_df['y'], [1, 99])
        a_2, b_2 = np.percentile(temp_df_2['y'], [1, 99])

        a = min(a, a_2)
        b = max(b, b_2)

        temp_df = temp_df[(temp_df['y'] >= a) & (temp_df['y'] <= b)]
        log_x_rec = temp_df['x']
        log_y_rec = temp_df['y']

        temp_df_2 = temp_df_2[(temp_df_2['y'] >= a) & (temp_df_2['y'] <= b)]
        log_x_noise = temp_df_2['x']
        log_y_noise = temp_df_2['y']

        left = np.min(log_x_rec)
        right = np.max(log_x_rec)
        one_to_one = np.linspace(left, right, 1000)

        gridsize_rec = max(100, int(np.sqrt(len(log_x_rec)/10)))
        gridsize_noise = max(100, int(np.sqrt(len(log_x_noise)/10)))

        ax = axes[i, j_rec]
        ax.set_box_aspect(1)
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        hexbin_rec = ax.hexbin(log_x_rec, log_y_rec, gridsize=gridsize_rec, cmap=rec_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_rec, ax=ax)

        if j_rec == 0:
            ax.set_ylabel(r' $\Delta$Kron Radius $[arcsec]$')
        if i == 3:
            ax.set_xlabel(r'AB MAG')
        if i == 0:
            ax.set_title('Reconstructed Image')
            
        gradient_line_rec = mlines.Line2D([0], [0], color=cmap_rec(norm(log_x_rec.min())), lw=4, label=f'{label}, N={len(log_x_rec)}')
        #ax.legend(handles=[gradient_line_rec], loc='upper left')
        ax.legend(handles=[gradient_line_rec], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

        ax = axes[i, j_noise]
        ax.set_box_aspect(1)
        ax.set_xlim([c, d])
        ax.set_ylim([a, b])
        #ax.plot(one_to_one,one_to_one, c='red', linestyle='dashed', alpha=0.5)
        hexbin_noise = ax.hexbin(log_x_noise, log_y_noise, gridsize=gridsize_noise, cmap=noise_cmap, bins='log', mincnt=2, linewidths=0.1)
        cbar = fig.colorbar(hexbin_noise, ax=ax)

        if i == 3:
            ax.set_xlabel(r'AB MAG')
        if i == 0:
            ax.set_title('Noisy Image')
        gradient_line_noise = mlines.Line2D([0], [0], color=cmap_noise(norm(log_x_noise.min())), lw=4, label=f'{label}, N={len(log_x_noise)}')
        #ax.legend(handles=[gradient_line_noise], loc='upper left')
        ax.legend(handles=[gradient_line_noise], loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)
        ax.grid(True)  # Add gridlines

    # Adjust layout to prevent overlap
    plt.suptitle(r"Rec., Noisy $\Delta$Kron Radius vs. AB MAG", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()


def binned_median(x, y, bins):
    # Convert pd.NA to np.nan
    x, y = x.astype(float), y.astype(float)
    # Remove NaNs
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]

    # Compute binned statistics
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='median', bins=bins)
    bin_p25, _, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 25), bins=bins)
    bin_p75, _, _ = binned_statistic(x, y, statistic=lambda y: np.nanpercentile(y, 75), bins=bins)
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Remove NaN bins
    valid_bins = ~np.isnan(bin_means)
    return bin_centers[valid_bins], bin_means[valid_bins], bin_p25[valid_bins], bin_p75[valid_bins]

def new_metrics(df, exp_time_col='exp_ratio'):
    results = []
    for exp_ratio, group in df.groupby(exp_time_col):
        tp = group[(~group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())]
        fp = group[(group['flux_x_org'].isna()) & (~group['flux_x_rec'].isna())]
        fn = group[(~group['flux_x_org'].isna()) & (group['flux_x_rec'].isna())]
        
        tp_count, fp_count, fn_count = len(tp), len(fp), len(fn)
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'exp_ratio': int(exp_ratio),
            'TP': tp_count,
            'FP': fp_count,
            'FN': fn_count,
            'Precision': precision,
            'Recall': recall,
            'F-measure': f_measure
        })
    
    summary_df = pd.DataFrame(results)
    
    # Compute overall statistics
    overall_tp = summary_df['TP'].sum()
    overall_fp = summary_df['FP'].sum()
    overall_fn = summary_df['FN'].sum()
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f_measure = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    overall_stats = pd.DataFrame([{
        'exp_ratio': 0,
        'TP': overall_tp,
        'FP': overall_fp,
        'FN': overall_fn,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F-measure': overall_f_measure
    }])
    return pd.concat([summary_df, overall_stats], ignore_index=True)

def create_flux_flux_error_diagram(df, flux_org_col, flux_noise_col, flux_rec_col, flux_org_err_col,
                                    flux_noise_err_col, flux_rec_err_col, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Create a 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    j = 0
    df['nsr_org'] = df[flux_org_col]/df[flux_org_err_col]
    df['nsr_rec'] = df[flux_rec_col]/df[flux_rec_err_col]
    df['nsr_noise'] = df[flux_noise_col]/df[flux_noise_err_col]
    for index, exp_ratio in enumerate(np.sort(df['exp_ratio'].unique())):
        sub_df = df
        i = (index % 2)
        label = r"Total"
        
        if int(exp_ratio) != 0:
            sub_df = df[df['exp_ratio'] == exp_ratio]
            label = rf'$\gamma={int(exp_ratio)}$'
        
        ax = axes[i, j]
        # Define logarithmic bins
        bins = np.logspace(np.log10(sub_df[flux_org_col].abs().min()), np.log10(sub_df[flux_org_col].abs().max()), 20)

        # Compute binned statistics for each dataset
        gt_x, gt_median, gt_p25, gt_p75 = binned_median(sub_df[flux_org_col], sub_df['nsr_org'], bins)
        noisy_x, noisy_median, noisy_p25, noisy_p75 = binned_median(sub_df[flux_noise_col], sub_df['nsr_noise'], bins)
        rec_x, rec_median, rec_p25, rec_p75 = binned_median(sub_df[flux_rec_col], sub_df['nsr_rec'], bins)

        # Ground Truth
        ax.plot(gt_x, gt_median, color="black", label="Ground Truth")
        ax.fill_between(gt_x, gt_p25, gt_p75, color="black", alpha=0.2)

        # Noisy
        ax.plot(noisy_x, noisy_median, color="darkblue", label="Noisy")
        ax.fill_between(noisy_x, noisy_p25, noisy_p75, color="darkblue", alpha=0.2)

        # Reconstructed
        ax.plot(rec_x, rec_median, color="darkgreen", label=r"Reconstructed")
        ax.fill_between(rec_x, rec_p25, rec_p75, color="darkgreen", alpha=0.2)

        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.set_title(f"{label}")
        ax.legend()

        if j == 0:
            ax.set_ylabel(r'SNR$')
        if i == 1:
            ax.set_xlabel(r'Flux $[erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$')
        if i == 1:
            j += 1
    # Add a main title
    fig.suptitle(r"SNR vs. Flux", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, filename), dpi=500)
    ###plt.show()
    plt.close()

def main():
    all_metrics = []
    for path_to_data, tag in zip(paths, tags):
        print(path_to_data)
        if not os.path.exists(path_to_data):
            continue
        edited_filepath = os.path.join(os.path.dirname(path_to_data), f'edited_{os.path.basename(path_to_data)}')
        if not os.path.exists(edited_filepath):
            df = pd.read_parquet(path_to_data)
            df = add_columns(df)
            df.to_parquet(edited_filepath, index=False) 
        else:
            df = pd.read_parquet(edited_filepath)

        metrics = create_table(data_dir, metadata_filename, os.path.dirname(path_to_data), metrics_filename)

        bright_df = df[(df['abmag_cflux_rec'] <= 25.0) | (df['abmag_cflux_org'] <= 25.0)].reset_index(drop=True)
        bright_metrics = new_metrics(bright_df, exp_time_col='exp_ratio')
        bright_metrics.rename({col:'bright_' + col for col in bright_df if col != 'exp_ratio'})

        final_metrics = pd.merge(bright_metrics, metrics, on=['exp_ratio'])
        final_metrics['tag'] = tag
        all_metrics.append(final_metrics)
        bright_df = df[df['abmag_cflux_org'] <= 25.0].dropna(subset=['abmag_cflux_org']).reset_index(drop=True)
        output_dir_tag = os.path.join(output_dir, tag)
        for my_df, my_tag in zip([bright_df, df], ('25', 'all')):
            my_output_dir = os.path.join(output_dir_tag, my_tag)
            if not os.path.exists(my_output_dir):
                os.makedirs(my_output_dir, exist_ok=True)
            rec_cmap = 'viridis'
            noise_cmap = 'cividis'
            org_flux_col = 'a_cflux_org'
            rec_flux_col = 'a_cflux_rec'
            noise_flux_col = 'a_cflux_noise'

            kron_radius_org_col = 'kron_radius_org'
            kron_radius_rec_col = 'kron_radius_rec'
            kron_radius_noise_col = 'kron_radius_noise'

            org_mag_col = 'abmag_cflux_org'
            flux_org_err_col = 'a_flux_err_org'
            flux_noise_err_col = 'a_flux_err_noise'
            flux_rec_err_col = 'a_flux_err_rec'

            plot_functions = [
                (create_delta_flux_plots, 'flux_ratio_combined_both.png', [org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap]),
                (create_delta_kron_radii_plots, 'delta_kron_combined_both.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap]),
                
                (create_delta_flux_mag_plots, 'flux_ratio_mag_combined_both.png', [org_flux_col, rec_flux_col, noise_flux_col, org_mag_col, rec_cmap, noise_cmap]),
                (create_delta_kron_radii_mag_plots, 'delta_kron_mag_combined_both.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, org_mag_col, rec_cmap, noise_cmap]),
                
                (create_only_delta_kron_mag_plots, 'delta_kron_mag_rec.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, org_mag_col, rec_cmap]),
                (create_only_delta_flux_mag_plots, 'flux_ratio_mag_rec.png', [org_flux_col, rec_flux_col, noise_flux_col, org_mag_col, rec_cmap]),
                
                (create_only_delta_rec_kron_plots, 'delta_kron_rec.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap]),
                (create_only_delta_rec_flux_plots, 'flux_ratio_rec.png', [org_flux_col, rec_flux_col, noise_flux_col, rec_cmap]),
                
                (create_kron_radii_plots, 'kron_combined.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap]),
                (create_flux_plots, 'flux_combined.png', [org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap]),
                
                (create_kron_radii_plots_both_det, 'kron_combined_both.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap]),
                (create_flux_plots_both_det, 'flux_combined_both.png', [org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap]),
                
                (create_only_rec_flux_plots, 'flux_rec.png', [org_flux_col, rec_flux_col, noise_flux_col, rec_cmap]),
                (create_only_rec_kron_plots, 'kron_rec.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap]),

                (create_delta_flux_mag_plots_combined, 'flux_ratio_mag_combined.png', [org_flux_col, rec_flux_col, noise_flux_col, org_mag_col, rec_cmap, noise_cmap]),
                (create_delta_kron_radii_mag_combined, 'delta_kron_mag_combined.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, org_mag_col, rec_cmap, noise_cmap]),
                
                (create_delta_flux_plots_combined, 'flux_ratio_combined.png', [org_flux_col, rec_flux_col, noise_flux_col, rec_cmap, noise_cmap]),
                (create_delta_kron_radii_plots_combined, 'delta_kron_combined.png', [kron_radius_org_col, kron_radius_rec_col, kron_radius_noise_col, rec_cmap, noise_cmap])
            ]

            for func, filename, cols in plot_functions:
                try:
                    func(my_df, *cols, my_output_dir, my_tag + "_" + filename)
                except Exception as e:
                    logging.warning(f"Error in {func.__name__} with filename {filename}: {e}")

            filename = 'snr.png'
            try:
                create_flux_flux_error_diagram(df, org_flux_col, noise_flux_col, rec_flux_col, flux_org_err_col,
                                    flux_noise_err_col, flux_rec_err_col, my_output_dir, my_tag + "_" + filename)
            except:
                pass

    if len(all_metrics):
        all_metrics = pd.concat(all_metrics, ignore_index=True).sort_values(by=['tag', 'exp_ratio'])
        all_metrics.to_csv(os.path.join(output_dir, 'all_metrics.csv'), index=False)

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    main()