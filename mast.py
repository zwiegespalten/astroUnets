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

async def download_image(id_, url, save_dir, session, semaphore):
    async with semaphore:
        try:
            if not session:
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
            
            async with session.get(url) as response:
                if response.status == 200:

                    content = await response.read()
                    os.makedirs(save_dir, exist_ok=True)
                    file_name = f"{id_}_raw.fits"
                    file_path = os.path.join(save_dir, file_name)

                    with fits.open(io.BytesIO(content)) as hdul:
                        hdul.writeto(file_path, overwrite=True)
                    return True 
        except Exception as err:
            logging.warning(f'an error occured while downloading: {err}')
        return False

async def download_images(ids, urls, save_dir, max_requests=5, reset_after=10):
    semaphore = asyncio.Semaphore(max_requests)
    # Use a session reset mechanism
    session = None
    try:
        for i, (id_, url) in enumerate(zip(ids, urls)):
            if i % reset_after == 0:
                if session:
                    await session.close()
                session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))

            result = await download_image(id_, url, save_dir, session, semaphore)
        if session:
            await session.close()
        return result
    except Exception as err:
        logging.warning(f'an error occurred while downloading urls: {err}')
    if session:
        await session.close()
    return None
    
async def generate_urls(product_identifier):
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(process_urls(table, id_,lock, save_dir, column, url_column, max_requests, reset_after, download))

def request_and_save(table, column, url_column, save_dir, output_name, max_requests=10000, reset_after=5, max_workers=16, download=False, period=60, step=10):
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











