import io
import os, logging, time
import asyncio, aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from astroquery.mast.missions import MastMissions
from astropy.io import fits
import astropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mission = 'hst'
filters = {'sci_spec_1234' : 'F160W,F160W;*,*;F160W,*;F160W;*',
        'sci_obs_type' : 'image',
        'sci_aec' : 'S',
        'sci_instrume' : 'wfc3',
}
main_column = 'sci_actual_duration'
save_dir = f'{os.getcwd()}/hts_wfc3_F160W'
N_OF_REQUESTS = 10000
N_OF_REQUESTS_TEMP = 0
START = False
LOCK = asyncio.Lock()

def filter_out_mast(mission, filters):
    """
    Retrieves and filters metadata from the MAST (Mikulski Archive for Space Telescopes) based on specified mission 
    and filter criteria. It queries the MAST API to return relevant metadata, handles pagination, and returns the 
    result as a pandas DataFrame.

    Parameters:
        mission (str): The name of the space mission (e.g., 'HST', 'TESS') to query metadata from.
        filters (dict): A dictionary containing filter criteria where keys are column names and values are filter values.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the filtered metadata retrieved from the MAST archive. 
                      Returns None if there is an error or no data is found.

    Notes:
        - The function uses pagination to handle cases where the number of results exceeds the API limit.
        - The filter criteria must match column names available for the specified mission.
        - If no filters are provided, all metadata from the specified mission will be returned.
        - The function uses the `MastMissions` class to query MAST and convert the result into a pandas DataFrame.

    Example:
        filters = {'target_name': 'HD 12345', 'observation_type': 'science'}
        metadata = filter_out_mast('TESS', filters)
    """

    kwargs = {}
    length = 1
    offset = 0
    limit = 5000
    results = []

    try:
        missions = MastMissions(mission=mission)
        # getting the column names in the mission
        if missions:
            columns = missions.get_column_list()
            columns = columns['name']
        else:
            return None

        # checking whether the filter is a dictionary and whether the given filters are covered by the mission
        if isinstance(filters, dict):
            keys = filters.keys()
            for key in keys:
                if key in columns:
                    kwargs[key] = filters[key]

        # iterating with offset as there seems to be a limit of maximum returnees
        index = 0
        while length:
            result = missions.query_criteria(select_cols=[],
                                            limit=limit,
                                            offset=offset,
                                            **kwargs)
            if result:
                results.append(result)
            length = len(result)
            
            if index == 0 and length < offset:
                break
            offset += length
            index += 1

        results = astropy.table.vstack(results).to_pandas()
        return results
    except Exception as err:
        logging.warning(f"An error occurred while retrieving metadata: {err}")
        return None
    
def plot_histogram(data, bins=20, label=None, xlabel='Exposure Time (s)', ylabel='Frequency', title='Histogram of Exposure', output_filename='histogram.png',loc='upper left', c='b', rotation=60):
    """
    Plots a histogram of the given data and saves it as an image file. The histogram is plotted with statistical 
    information, including mean, median, standard deviation, and variance.

    Parameters:
        data (array-like): Input data to be plotted as a histogram.
        bins (int, optional): Number of bins in the histogram. Default is 20.
        label (str, optional): Label for the histogram. If None, the label is automatically generated with statistical information. Default is None.
        xlabel (str, optional): Label for the x-axis. Default is 'Exposure Time (s)'.
        ylabel (str, optional): Label for the y-axis. Default is 'Frequency'.
        title (str, optional): Title of the plot. Default is 'Histogram of Exposure'.
        output_filename (str, optional): The name of the output image file where the histogram will be saved. Default is 'histogram.png'.
        loc (str, optional): Location for the legend. Default is 'upper left'.
        c (str, optional): Color for the histogram bars. Default is 'b' (blue).
        rotation (int, optional): Rotation angle for the x-axis labels. Default is 60.

    Returns:
        None: The function saves the plot as an image and displays it. It does not return any value.

    Notes:
        - The histogram is displayed on a logarithmic scale for the y-axis.
        - Vertical lines are drawn to indicate the mean (red) and median (purple) of the data.
        - The plot is saved as a PNG file with a resolution of 300 dpi.

    Example:
        plot_histogram(data, bins=30, xlabel='Magnitude', ylabel='Count', title='Histogram of Magnitudes', output_filename='mag_histogram.png')
    """
    # Calculate statistics
    mean = data.mean()
    median = data.median()
    std = data.std()
    var = data.var()

    if label is None:
        label = f'mean: {round(mean, 2)}, median: {round(median, 2)}\nstd: {round(std, 2)}, var: {round(var, 2)}\n N: {len(data)}'

    # Create histogram
    counts, bins_edges, patches = plt.hist(data, bins=bins, density=False, alpha=0.75, color=c, edgecolor='black', label=label)
    xticks = np.linspace(bins_edges.min(), bins_edges.max(), bins+1)
    diff = (xticks[1] - xticks[0])/2
    plt.xticks(np.linspace(bins_edges.min() + diff, bins_edges.max() + diff, bins+1), rotation=rotation)
    plt.grid()
    plt.axvline(mean, color='r', linestyle='--', label='mean')
    plt.axvline(median, color='purple', linestyle='--', label='median')
    plt.legend(loc=loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

async def download_image(id_, url, save_dir, session, semaphore, filename):
    """
    Downloads an image from a given URL asynchronously and saves it to the specified directory.

    Parameters:
        id_ (str): The unique identifier for the image.
        url (str): The URL from which the image will be downloaded.
        save_dir (str): The directory where the image will be saved.
        session (aiohttp.ClientSession, optional): The session used to make the HTTP request. If not provided, a new session is created.
        semaphore (asyncio.Semaphore): A semaphore to control concurrency and prevent overloading the server.
        filename (str): The name of the file under which the image will be saved.

    Returns:
        bool: Returns `True` if the image was successfully downloaded and saved, `False` otherwise.

    Notes:
        - The function uses `aiohttp` for asynchronous HTTP requests with a timeout of 15 seconds.
        - If the image is successfully retrieved, it is saved as a FITS file.
        - The function handles exceptions and logs warnings if errors occur during the download.
        - A semaphore is used to limit the number of concurrent downloads.
        
    Example:
        await download_image(id_, url, save_dir, session, semaphore, filename)
    """
    async with semaphore:
        try:
            if not session:
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
            
            async with session.get(url) as response:
                if response.status == 200:

                    content = await response.read()
                    os.makedirs(save_dir, exist_ok=True)
                    #file_name = f"{id_}_raw.fits"
                    file_path = os.path.join(save_dir, filename)

                    with fits.open(io.BytesIO(content)) as hdul:
                        hdul.writeto(file_path, overwrite=True)
                    return True 
        except Exception as err:
            logging.warning(f'an error occured while downloading: {err}')
        return False

async def download_images(ids, urls, save_dir, max_requests=5, reset_after=10):
    """
    Downloads multiple images from a list of URLs asynchronously and saves them to the specified directory.

    Parameters:
        ids (iterable): A list or iterable of unique identifiers for each image.
        urls (iterable): A list or iterable of URLs pointing to the images to be downloaded.
        save_dir (str): The directory where the downloaded images will be saved.
        max_requests (int, optional): The maximum number of concurrent download requests. Defaults to 5.
        reset_after (int, optional): The number of requests after which the session should be reset. Defaults to 10.

    Returns:
        bool: Returns `True` if all images were successfully downloaded, `False` if any download failed, or `None` if an error occurred.

    Notes:
        - The function uses `aiohttp` for asynchronous HTTP requests with a timeout of 15 seconds.
        - After every `reset_after` requests, the session is closed and reopened to avoid potential connection issues.
        - The images are saved as FITS files using the `download_image` function.
        - Any invalid URLs (e.g., `NaN` values) are skipped without causing an error.

    Example:
        await download_images(ids, urls, save_dir, max_requests=10, reset_after=15)
    """
    semaphore = asyncio.Semaphore(max_requests)
    # Use a session reset mechanism
    session = None
    try:
        for i, (id_, url) in enumerate(zip(ids, urls)):
            if i % reset_after == 0:
                if session:
                    await session.close()
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
            if pd.isna(url):
                continue
            filename = url.split('%2F')[-1]
            result = await download_image(id_, url, save_dir, session, semaphore, filename)
        if session:
            await session.close()
        return result
    except Exception as err:
        logging.warning(f'an error occurred while downloading urls: {err}')
    if session:
        await session.close()
    return None
    
async def generate_urls(product_identifier):
    """
    Generates a list of URLs for downloading FITS files based on the given product identifier.

    Parameters:
        product_identifier (str): A product identifier (typically a string representing the base part of the filename, excluding the last 3 characters).
    Returns:
        list: A list of URLs to retrieve the corresponding FITS files for different permutations of characters.
    Notes:
        - The function generates URLs by appending permutations of two characters from a predefined character set to the base product identifier.
        - Each generated URL points to a product on the Hubble Space Telescope's MAST (Mikulski Archive for Space Telescopes).
        - The resulting URLs follow the pattern:
          `https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name={product_identifier}%2F{base_part}{perm}q_raw.fits`
        - If an error occurs during the URL generation process, it logs the error and returns `None`.

    Example:
        urls = await generate_urls('HST-12345')
    """

    char_set = '0123456789abcdefghijklmnopqrstuvwxyz'
    try:
        base_part = product_identifier[:-3].lower()

        permutations = []
        for char in char_set:
            for char_ in char_set:
                perm = f'{char}{char_}'
                permutations.append(perm)

        urls = []
        for perm in permutations:
            short_filename = f"{base_part}{perm}q_raw.fits"
            url = f"https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name={product_identifier}%2F{short_filename}"
            urls.append(url)
        return urls
    except Exception as err:
        logging.warning(f'an error occurred while generating URLs: {err}')
        return None

async def process_url(lock, table, column, url_column, id_, url, save_dir, session, semaphore, download=False):
    """
    Processes a URL to either update a DataFrame with the URL or download the corresponding file, depending on the 'download' flag.

    Parameters:
        lock (asyncio.Lock): A lock to ensure thread-safe access to shared resources (used for modifying the DataFrame).
        table (pandas.DataFrame): The DataFrame containing metadata where the URL is stored.
        column (str): The column name in the DataFrame used to find the row corresponding to 'id_'.
        url_column (str): The column name in the DataFrame where the URL is to be stored.
        id_ (str): The identifier used to locate the row in the DataFrame to update with the URL.
        url (str): The URL to process (download or store in the DataFrame).
        save_dir (str): Directory where the downloaded file will be saved, if 'download' is True.
        session (aiohttp.ClientSession): The active session to be used for making HTTP requests. If not provided, a new session will be created.
        semaphore (asyncio.Semaphore): Semaphore to limit the number of concurrent download requests.
        download (bool): A flag to indicate whether the file should be downloaded (`True`), or just the URL should be stored (`False`).

    Returns:
        str or None: Returns the URL if successful, or None if there was an error during processing.

    Notes:
        - The function checks if the URL request returns a successful status (HTTP 200).
        - If `download` is `True`, the content is downloaded and saved as a FITS file in the specified `save_dir`.
        - The `table` is updated with the `url_column` corresponding to the `id_` if the download or URL update is successful.
        - An asyncio lock (`lock`) is used to prevent race conditions when modifying the DataFrame.
        - If an error occurs during the request or processing, it logs the error and returns `None`.

    Example:
        url = await process_url(lock, df, 'id', 'url', 'product123', 'http://example.com/file', './downloads', session, semaphore, download=True)
    """
    async with semaphore:
        try:
            if not session:
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
            
            async with session.get(url) as response:
                if response.status == 200:
                    print(url)
                    with lock:
                        table.loc[table[column] == id_, url_column] = url

                    if download:
                        content = await response.read()
                        os.makedirs(save_dir, exist_ok=True)
                        file_name = f"{id_}_raw.fits"
                        file_path = os.path.join(save_dir, file_name)

                        with fits.open(io.BytesIO(content)) as hdul:
                            hdul.writeto(file_path, overwrite=True)

                    loop = asyncio.get_running_loop()
                    loop.stop() 
                    return url     
        except Exception as err:
            logging.warning(f"An error occurred while processing the {url}: {err}")
        return None

async def process_urls(table, id_, lock, save_dir, column, url_column, max_requests=5, reset_after=10, download=False):
    """
    Processes multiple URLs associated with a product identifier, attempting to retrieve and download the corresponding file or update the DataFrame with the URL.

    This function generates URLs for a given `id_`, processes them one by one, and updates the provided `table` DataFrame with the first successful URL. 
    It optionally downloads the corresponding FITS file if `download` is set to True. A session reset mechanism is used to ensure a fresh session is used periodically.

    Parameters:
        table (pandas.DataFrame): The DataFrame containing metadata where the URL is stored.
        id_ (str): The product identifier for which URLs will be generated and processed.
        lock (asyncio.Lock): A lock to ensure thread-safe access to shared resources (used for modifying the DataFrame).
        save_dir (str): Directory where the downloaded file will be saved, if 'download' is True.
        column (str): The column name in the DataFrame used to find the row corresponding to 'id_'.
        url_column (str): The column name in the DataFrame where the URL is to be stored.
        max_requests (int, optional): The maximum number of concurrent requests allowed. Default is 5.
        reset_after (int, optional): The number of requests to make before resetting the session. Default is 10.
        download (bool, optional): A flag to indicate whether the file should be downloaded (`True`), or just the URL should be stored (`False`).

    Returns:
        str or None: Returns the first successfully processed URL if found, or None if no URLs were successfully processed or if there was an error.

    Notes:
        - The function generates a list of URLs using `generate_urls` based on the `id_`.
        - A semaphore limits the number of concurrent requests to avoid overloading the server.
        - The function uses a session reset mechanism, where the session is closed and reopened after a certain number of requests (`reset_after`).
        - If `download` is `True`, the corresponding FITS file is downloaded and saved in the `save_dir`.
        - A lock (`lock`) is used to ensure safe concurrent updates to the DataFrame.
        - If an error occurs during the URL processing, the function logs the error and returns `None`.

    Example:
        result = await process_urls(df, 'product123', lock, './downloads', 'id', 'url', max_requests=5, download=True)
    """
    semaphore = asyncio.Semaphore(max_requests)
    url_generator = await generate_urls(id_)
    result = None
    # Use a session reset mechanism
    session = None
    try:
        for i, url in enumerate(url_generator):
            if i % reset_after == 0:
                if session:
                    await session.close()
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))

            result = await process_url(lock, table, column, url_column, id_, url, save_dir, session, semaphore, download)
            if result:
                break
        if session:
            await session.close()
        return result
    except Exception as err:
        logging.warning(f'an error occurred while processing urls: {err}')

    if session:
        await session.close()
    try:
        loop = asyncio.get_running_loop()
        loop.stop() 
    except Exception as err:
        pass
    return None

def wrap_process_urls(table, id_, lock, save_dir, column, url_column, max_requests=10000, reset_after=5, download=False):
    """
    A wrapper function to run the asynchronous `process_urls` function in a synchronous context.

    This function creates a new event loop and runs the `process_urls` function until it completes.
    It is useful when you need to call an asynchronous function from synchronous code (e.g., in a non-async environment).

    Parameters:
        table (pandas.DataFrame): The DataFrame containing metadata where the URL will be stored.
        id_ (str): The product identifier for which URLs will be generated and processed.
        lock (asyncio.Lock): A lock to ensure thread-safe access to shared resources (used for modifying the DataFrame).
        save_dir (str): Directory where the downloaded file will be saved, if 'download' is True.
        column (str): The column name in the DataFrame used to find the row corresponding to 'id_'.
        url_column (str): The column name in the DataFrame where the URL is to be stored.
        max_requests (int, optional): The maximum number of concurrent requests allowed. Default is 10000.
        reset_after (int, optional): The number of requests to make before resetting the session. Default is 5.
        download (bool, optional): A flag to indicate whether the file should be downloaded (`True`), or just the URL should be stored (`False`).

    Returns:
        str or None: Returns the first successfully processed URL if found, or None if no URLs were successfully processed or if there was an error.

    Notes:
        - This function is a wrapper for the asynchronous `process_urls` function, allowing it to be called from synchronous code.
        - A new event loop is created and set for the current thread, which is then used to run the `process_urls` function.
        - All parameters of the wrapped `process_urls` function are passed along to `process_urls`.

    Example:
        result = wrap_process_urls(df, 'product123', lock, './downloads', 'id', 'url', max_requests=5, download=True)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_urls(table, id_,lock, save_dir, column, url_column, max_requests, reset_after, download))

def request_and_save(table, column, url_column, save_dir, output_name, max_requests=10000, reset_after=5, max_workers=16, download=False, period=60, step=10):
    """
    Processes a table by fetching URLs (and optionally downloading files) in parallel and periodically saving the table to a CSV file.

    This function runs a background thread that periodically saves the table to a CSV file while processing URLs asynchronously using a ThreadPoolExecutor. 
    The function attempts to process each row in the DataFrame by generating and fetching URLs, updating the table with the results.

    Parameters:
        table (pandas.DataFrame): The DataFrame containing the data to process, with columns `column` and `url_column`.
        column (str): The column name in the table that holds the product identifiers.
        url_column (str): The column name in the table where the URLs will be stored after processing.
        save_dir (str): Directory where any downloaded files will be saved (if `download` is `True`).
        output_name (str): The name of the CSV file where the table will be saved periodically.
        max_requests (int, optional): The maximum number of concurrent requests allowed. Default is 10000.
        reset_after (int, optional): The number of requests to make before resetting the session. Default is 5.
        max_workers (int, optional): The maximum number of parallel workers for processing URLs. Default is 16.
        download (bool, optional): If `True`, download the files after processing the URLs. Default is `False`.
        period (int, optional): The total time (in seconds) to wait before saving the table. Default is 60 seconds.
        step (int, optional): The interval (in seconds) between each check for saving the table. Default is 10 seconds.

    Returns:
        pandas.DataFrame: The updated DataFrame with URLs processed and saved.

    Notes:
        - This function utilizes a background thread that saves the table periodically.
        - It uses a `ThreadPoolExecutor` for parallel processing of URLs.
        - The background thread stops once all URLs are processed or the `cond["flag"]` is set to `False`.
        - The function assumes that the table's column specified in `column` contains product identifiers and the column specified in `url_column` will hold the resulting URLs.

    Example:
        updated_table = request_and_save(df, 'id', 'url', './downloads', 'output.csv', download=True)
    """
    if column not in table or url_column not in table:
        return 

    cond = {'flag': True}
    lock = threading.Lock()

    def thread_function(table, lock, cond, output_name, period, step):
        if not isinstance(table, pd.DataFrame):
            return 
        while cond["flag"]:
            for _ in range(period // step):
                if not cond['flag']:
                    break
                time.sleep(step)
            if not cond['flag']:
                break
            with lock:
                if isinstance(table, pd.DataFrame) and not table.empty:
                    table.to_csv(output_name, index=False)
        with lock:
            if isinstance(table, pd.DataFrame) and not table.empty:
                table.to_csv(output_name, index=False)

    # Start the background thread
    thread = threading.Thread(target=thread_function, args=(table, lock, cond, output_name, period, step))
    thread.start()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for id_, url in zip(table[column].values, table[url_column].values):
            if not pd.isna(url):
                continue
            
            futures.append(executor.submit(wrap_process_urls, table, id_, lock, save_dir, column, url_column, max_requests, reset_after, download))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:
                logging.warning(f'An error occurred while processing futures: {err}')

    # Signal the thread to stop and wait for it to finish
    cond['flag'] = False
    thread.join()
    return table

def update_file(input_filepath, column, url_column, save_dir, max_requests=10000, reset_after=5, max_workers=16, download=False, period=60, step=10):
    """
    Updates a CSV file by processing URLs for each entry and saving the table with the resulting URLs.

    This function reads a CSV file from the specified `input_filepath`, processes the URLs (if not already processed), 
    and updates the table with the processed URLs. It then saves the updated table back to the same file. 
    The function uses `request_and_save` to fetch URLs and optionally download files.

    Parameters:
        input_filepath (str): The path to the input CSV file to be updated.
        column (str): The column name in the CSV file that holds the product identifiers.
        url_column (str): The column name in the CSV file where the URLs will be stored after processing.
        save_dir (str): Directory where any downloaded files will be saved (if `download` is `True`).
        max_requests (int, optional): The maximum number of concurrent requests allowed. Default is 10000.
        reset_after (int, optional): The number of requests to make before resetting the session. Default is 5.
        max_workers (int, optional): The maximum number of parallel workers for processing URLs. Default is 16.
        download (bool, optional): If `True`, download the files after processing the URLs. Default is `False`.
        period (int, optional): The total time (in seconds) to wait before saving the table. Default is 60 seconds.
        step (int, optional): The interval (in seconds) between each check for saving the table. Default is 10 seconds.

    Returns:
        pandas.DataFrame: The updated DataFrame with URLs processed and saved to the input file.

    Notes:
        - If the `url_column` doesn't exist in the input file, it is created with `None` values for each row.
        - The function calls `request_and_save` to process URLs and download files if needed.
        - The table is periodically saved to the input file during processing.

    Example:
        updated_table = update_file('data.csv', 'id', 'url', './downloads', download=True)
    """
        
    if os.path.exists(input_filepath):
        df = pd.read_csv(input_filepath)
        if url_column not in df:
            df[url_column] = [None]*len(df)
        return request_and_save(df, column, url_column, save_dir, input_filepath, max_requests, reset_after, max_workers, download, period, step)
    else:
        return

if False:
    table = filter_out_mast(mission, filters)
    higher_exposure = table[table['sci_actual_duration'] > 4000]
    higher_exposure.sort_values(by=['sci_actual_duration'], ascending=False, inplace=True)
    higher_exposure.reset_index(drop=True, inplace=True)
    higher_exposure.to_csv('higher_exposure_hst_wfc3_f160W_metadata.csv', index=False)

    xlabel = 'Exposure Time (s)'
    ylabel = 'Frequency'
    title = 'Histogram of Exposure of HST Images with F160W Filter and WFC3 above 4000'
    output_filename = 'over_4000_histogram_exp_hst.png'
    bins = 20
    loc = 'upper right'
    plot_histogram(higher_exposure[main_column], bins=bins, xlabel=xlabel, ylabel=ylabel, title=title, output_filename=output_filename, loc=loc, c='cyan')

    xlabel = 'Exposure Time (s)'
    ylabel = 'Frequency'
    title = 'Histogram of Exposure of HST Images with F160W Filter and WFC3'
    output_filename = 'histogram_exp_hst.png'
    bins = 20
    loc = 'upper right'
    table.sort_values(by=[main_column], ascending=False, inplace=True)
    table.to_csv('hst_wfc3_f160W_metadata.csv', index=False)
    plot_histogram(table[main_column], bins=bins, xlabel=xlabel, ylabel=ylabel, title=title, output_filename=output_filename, loc=loc, c='seagreen')

    higher_exposure['URL'] = np.zeros(len(higher_exposure))
    request_and_save(higher_exposure, 'sci_data_set_name', 'URL', save_dir, max_requests=N_OF_REQUESTS, reset_after=5, max_workers=64, download=False)
    higher_exposure.to_csv('higher_exposure_hst_wfc3_f160W_metadata.csv', index=False)

    "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=" + "IE37NBANQ" + "%2Fie37nban" + "q_raw.fits"
    "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=" + "IBR212020" + "%2Fibr212tm" + "q_raw.fits"
    "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=" + "IF8A79030" + "%2Fif8a79ai" + "q_raw.fits" 
    "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=" + "IDMA20020" + "%2Fidma20xi" + "q_raw.fits"
    "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=" + "IB6V03090" + "%2Fib6v03ig" + "q_raw.fits"











