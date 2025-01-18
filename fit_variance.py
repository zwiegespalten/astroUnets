import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# File path to the metadata CSV
filepath_metadata = f'{os.path.dirname(__file__)}/our_data/metadata_filepath_enriched_with_noise.csv'
filepath_metadata_cropped = f'{os.path.dirname(__file__)}/our_data/cropped_images_stats.csv'

# Load the metadata
df = pd.read_csv(filepath_metadata)

# Group by EXPTIME and compute the mean of variance (std**2) for each group
df['variance'] = df['std'] ** 2
grouped = df.groupby('sci_actual_duration')['variance'].mean().reset_index()

# Rename columns for clarity
grouped.columns = ['sci_actual_duration', 'mean_variance']

def linear_function(x, a, b):
    return a * x + b

# Filter the data to remove outliers
#Q1 = grouped['mean_variance'].quantile(0.25)
#Q3 = grouped['mean_variance'].quantile(0.75)
#IQR = Q3 - Q1
#filtered_data = grouped[(grouped['mean_variance'] >= Q1 - 1.5 * IQR) & (grouped['mean_variance'] <= Q3 + 1.5 * IQR)]
filtered_data = grouped
# Ensure there is data to fit
if len(filtered_data) > 0:
    exposure_time_filtered = filtered_data['sci_actual_duration'].values
    variance_filtered = filtered_data['mean_variance'].values

    # Remove zero or negative values before taking logarithm
    valid_indices = (exposure_time_filtered > 0) & (variance_filtered > 0)
    exposure_time_filtered = exposure_time_filtered[valid_indices]
    variance_filtered = variance_filtered[valid_indices]

    val_dict = {e: v for e, v in zip(exposure_time_filtered, variance_filtered)}
    val_dict = {e: v for e, v in val_dict.items() if e < 15000}
    exposure_time_filtered = np.array(list(val_dict.keys()))
    variance_filtered = np.array(list(val_dict.values()))

    # Take logarithm of the data for log-log plotting
    log_exposure_time_filtered = np.log(exposure_time_filtered)
    log_variance_filtered = np.log(variance_filtered)

    popt, pcov = curve_fit(
        linear_function,
        log_exposure_time_filtered,
        log_variance_filtered,
        p0=[-0.9, 0.1]
    )
    a, b = popt  # Best-fit parameters

    # Calculate expected variance (log scale)
    log_expected_variance = linear_function(log_exposure_time_filtered, *popt)
    expected_variance = np.exp(log_expected_variance)

    # Calculate reduced chi-squared
    residuals = variance_filtered - expected_variance
    chi_squared = np.sum((residuals ** 2) / variance_filtered)
    degrees_of_freedom = len(variance_filtered) - len(popt)
    chi_squared_reduced = chi_squared / degrees_of_freedom

    # Plot the results
    plt.figure(figsize=(12, 7))
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Variance of Background Noise (log scale)')
    plt.xlabel('Exposure Time (log scale)')
    plt.scatter(exposure_time_filtered, variance_filtered, alpha=0.6, label='Cleaned Data')
    plt.plot(
        exposure_time_filtered,
        expected_variance,
        color='red',
        label=f'Fit: Linearized Power Law (χ²_red={chi_squared_reduced:.2f})'
    )
    plt.legend()
    plt.show()
    functions = [
        ("Linearized Power Law", linear_function, [-0.9, 0.1])
    ]

for label, func, initial_guess in functions:
    try:
        # Fit the function to the log-transformed data
        params, covariance = curve_fit(func, log_exposure_time_filtered, log_variance_filtered, p0=initial_guess)
        standard_errors = np.sqrt(np.diag(covariance))
        
        # Generate fitted values on the log scale
        log_x_values = np.linspace(min(log_exposure_time_filtered), max(log_exposure_time_filtered), 500)
        log_y_values = func(log_x_values, *params)

        # Exponentiate to revert to the original scale
        x_values = np.exp(log_x_values)
        y_values = np.exp(log_y_values)  # No *2 here; variance is already on the correct scale

        # Filter x_values and y_values to keep them within the original data range
        valid_indices = (x_values >= min(exposure_time_filtered)) & (x_values <= max(exposure_time_filtered))
        x_values = x_values[valid_indices]
        y_values = y_values[valid_indices]

        # Format the parameters and uncertainties to display in the legend
        param_str = ', '.join([f'{param:.3f} ± {error:.3f}' for param, error in zip(params, standard_errors)])
        plt.plot(x_values, y_values, label=f'{label} Fit\nParams: {param_str}', linewidth=2, c='r')

        # Save the parameters in terms of the original power law: sigma^2 = a * x^t
        df = pd.DataFrame({
            't': [params[0]],  # Exponent t
            'a': [np.exp(params[1])],  # Coefficient a
            'dt': [standard_errors[0]],  # Uncertainty in t
            'da': [np.exp(params[1]) * standard_errors[1]]  # Uncertainty in a
        })
        df.to_csv(f'{os.path.dirname(__file__)}/our_data/fit_info.csv', index=False)
    except Exception as e:
        print(f"Error fitting {label}: {e}")

# Customize the plot
plt.legend(loc='upper right')
plt.title('Log-Log Fit of Exposure Time vs Variance')
plt.grid()
plt.savefig(f'{os.path.dirname(__file__)}/our_data/fit.png', dpi=96)
plt.show()