import os, logging, time, shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from astropy.io import fits
from network import network

############DATA HANDLING ##################################
############################################################
############################################################
def save_fits(image, name, path):
    try:
        hdu = fits.PrimaryHDU(image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(path + name + '.fits', overwrite=True)
        return True
    except Exception as err:
        logging.warning(f'an error occurred while saving the image: {err}')
        return

def text_write(filepath, epoch, cnt, loss,mean_loss, time, name):
    try:
        with open(filepath, "a") as file:
            file.write(str(epoch) + "\t" + str(cnt) + "\t" + str(loss) + "\t" + str(mean_loss) + "\t" + str(time) + "\t" + str(name) + "\n")
    except Exception as err:
        logging.warning(f'an error occurred while writing to the txt file: {err}')
        return
    
############################################################
############################################################    
############################################################   

#############CROPPING, NOISE ADDITION AND DATA AUGMENTATIONS###################################################
###############################################################################################################
###############################################################################################################

def black_level(H, W, out, ps=256, steps=100):
    lowest_per = 1
    indices = (0,0)
    if H < ps or W < ps:
        return
    try:
        for i in range(steps):
            xx = np.random.randint(0, H - ps)
            yy = np.random.randint(0, W - ps)
            arr = out[xx:xx + ps, yy:yy + ps].reshape(-1)
            per = len(arr[arr == 0.000])/len(arr)

            if lowest_per > per:
                indices = (xx, yy)
        return indices
    except Exception as err:
        logging.warning(f'an error occurred while finding the most appropriate image: {err}')
        return

def poisson_noise(img_data, ratio=0.5, ex_time=1, xx=0, yy=0, ps=256, ron=3, dk=7, save=False,
                   sv_name=None, path=None, patch_noise=False):
    #ratio defined as 'I_new / I_original = r'
    try:
        if patch_noise:
            img_data = img_data[xx:xx + ps, yy:yy + ps]
    except Exception as err:
        logging.warning(f'an error occurred while cropping: {err}')
        return
    
    try:
        width, height = img_data.shape[0:2]
        img = img_data * ex_time
        
        dark_current_noise = np.random.normal(0, np.sqrt(dk*ex_time*ratio/(60*60)), (width, height))  
        readout_noise = np.random.normal(0, ron, (width, height))
        poisson_noise_img = np.random.poisson(img*ratio)
        
        noisy_img = (poisson_noise_img + readout_noise + dark_current_noise) / (ex_time*ratio)
        noisy_img = np.where(img_data == 0.0, 0.0, noisy_img)
    except Exception as err:
        logging.warning(f'an error occurred while adding noise: {err}')
        return
    
    if save:
        save_fits(noisy_img, sv_name, path)
    return noisy_img

def out_in_image(filepath, ps=256, steps=100, ratio=0.5, ron=3, dk=7, save=False,
                  sv_name=None, path=None, patch_noise=False, type_of_image='SCI',
                  low=60, high=10000):
    out = None
    ex_time = None
    try:
        with fits.open(filepath) as f:
            for i in range(len(f)):
                hdu = f[i]
                if isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)) and hdu.data is not None and isinstance(hdu.data, np.ndarray):
                    if out is None and type_of_image.lower() == str(hdu.name).lower():
                        out = hdu.data
                if ex_time is None and 'EXPTIME' in hdu.header:
                    ex_time = hdu.header['EXPTIME'] 
    except Exception as err:
        logging.warning(f'an error occurred while reading from the .fits file: {err}')
        return
    if out is None or ex_time is None or ex_time*ratio < low or ex_time*ratio > high:
        return 
    # ground truth
    try:
        H, W = out.shape[0], out.shape[1]
        xx, yy = black_level(H, W, out, ps=ps, steps=steps)

        out = np.where(out < 0.00000000e+00, 0.00000000e+00  , out)
        out_img = np.zeros((ps, ps, 1))
        out_img[:, :, 0] = out[xx:xx + ps, yy:yy + ps]
        gt_patch = np.expand_dims(out_img, axis=0)

        # input image
        in_imag =  poisson_noise(out[xx:xx + ps, yy:yy + ps], ratio=ratio,
                                 ex_time=ex_time, xx=xx, yy=yy, ps=ps, ron=ron,
                                 dk=dk, save=save, sv_name=sv_name, path=path,
                                   patch_noise=patch_noise)
    
        in_img = np.zeros((ps, ps, 1))
        in_img[:, :, 0] = in_imag
        in_patch = np.expand_dims(in_img, axis=0)
    except Exception as err:
        logging.warning(f'an error occurred while image handling: {err}')
        return
    return  gt_patch, in_patch, ex_time*ratio

def data_augment(images, start=2, stop=6, ps=256, steps=100, ratio=0.5, ron=3, dk=7, save=False,
                  sv_name=None, path=None, patch_noise=False, type_of_image='SCI', low=60, high=10000):
    if start >= stop:
        raise ValueError("`start` must be less than `stop`.")
    try:
        temp = {image: random.sample(range(start, stop), stop - start) for image in images}
        while temp:
            key = random.choice(list(temp))
            if temp[key]:
                ratio = temp[key].pop()
                ratio = 1 / ratio
                yield out_in_image(key, ps, steps, ratio=ratio,
                                   ron=ron, dk=dk, save=save, sv_name=sv_name,
                                   path=path, patch_noise=patch_noise,
                                   type_of_image=type_of_image,
                                   low=low, high=high), key
            else:
                temp.pop(key)
    except Exception as err:
        logging.error(f"Error during data augmentation: {err}")
        raise Exception('EXITING')
    
############ VALIDATION AND TRAINING ##########################################################################
###############################################################################################################
###############################################################################################################

def restore_checkpoint(checkpoint, checkpoint_dir, checkpoint_prefix, epoch):
    """Helper function to restore a checkpoint."""
    filename = f'{checkpoint_prefix}_{epoch:04d}.ckpt'
    suffix = [f.split('.')[1].split('-')[-1] for f in os.listdir(checkpoint_dir) if f.startswith(filename) and f.endswith('.index')]
    if suffix:
        filename = f'{filename}-{suffix[0]}'
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        try:
            checkpoint.restore(checkpoint_path).assert_consumed()
            return epoch
        except Exception as err:
            logging.warning(f"An error occurred while trying to restore the epoch {epoch}: {err}")
    return None

def load_checkpoint(checkpoint, checkpoint_dir, start_from_best, start_from_last, start_from_scratch):
    """Load the appropriate checkpoint based on the flags."""
    if start_from_best == start_from_last:
        raise ValueError("Cannot start from both 'best' and 'last' checkpoints simultaneously.")
    if start_from_scratch and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    start_epoch = 0
    
    # Get lists of available best epochs and model epochs
    best_epochs = sorted({int(f.split('.')[0].split('_')[-1]) for f in os.listdir(checkpoint_dir) if f.startswith('best_')})
    epochs = sorted({int(f.split('.')[0].split('_')[1]) for f in os.listdir(checkpoint_dir) if f.startswith('model_')})
    
    # Try restoring from the best epochs first
    if best_epochs and start_from_best and not start_from_last:
        for epoch in reversed(best_epochs):
            start_epoch = restore_checkpoint(checkpoint, checkpoint_dir, 'best_model', epoch)
            if start_epoch:
                return start_epoch
        start_from_best = False
        start_from_last = True
        logging.warning('no best checkpoint has been found, changing to the last working checkpoint')

    # If no best epoch was restored, or we are falling back to the last checkpoint
    if epochs and start_from_last and not start_from_best:
        for epoch in reversed(epochs):
            start_epoch = restore_checkpoint(checkpoint, checkpoint_dir, 'model', epoch)
            if start_epoch:
                return start_epoch
    if not start_epoch:
        logging.warning("No valid checkpoint found, starting from scratch.")
    return 0

def validate(model, eval_data, G_loss_fn, kwargs_data):
    try:
        if not isinstance(model, tf.keras.models.Model):
            logging.warning("The provided model is not a valid tf.keras model.")
            return
        if not callable(G_loss_fn):
            logging.warning("The provided loss function is not callable.")
            return
        if not eval_data:
            logging.warning("Evaluation data is empty. Validation cannot proceed.")
            return

        filepath_eval = random.choice(eval_data)
        eval_result = out_in_image(filepath_eval, **kwargs_data)
        if eval_result is None:
            return
        try:
            out_eval, in_eval, exp = eval_result
        except ValueError as e:
            return
        
        eval_prediction = model(in_eval, training=False)
        eval_current = G_loss_fn(out_eval, eval_prediction)
        return eval_current, exp
    except Exception as e:
        logging.warning(f"An unexpected error occurred during validation: {e}")
        return
    
def one_step_training(model, result, G_loss_fn, optimizer):
    if not isinstance(model, tf.keras.models.Model):
        raise ValueError("The 'model' must be an instance of tf.keras.models.Model.")
    if not callable(G_loss_fn):
        raise ValueError("The 'G_loss_fn' must be callable.")
    #if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
    #    raise ValueError("The 'optimizer' must be an instance of tf.keras.optimizers")
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        return
    try:
        gt_patch, in_patch, exp = result
    except ValueError as e:
        return
    
    with tf.GradientTape() as tape:

        predicted = model(in_patch, training=True)
        G_current = G_loss_fn(gt_patch, predicted)
        gradients = tape.gradient(G_current, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return gt_patch, in_patch, predicted, G_current, exp

def train_network(input_shape, folder, n_epochs, kwargs_data, kwargs_network, network_name='model',
         learning_rate=1e-4, optimizer=Adam, change_learning_rate=(2000, 1e-5), G_loss_fn=tf.keras.losses.MeanAbsoluteError(),
         start_from_scratch=False, start_from_best=False, start_from_last=True,
         save_freq=500, period_percent=10, eval_percent=20, period_save=30):
    
    # Ensures that the dateset exists and checks
    if os.path.exists(folder):
        os.chdir(folder)
    else:
        raise FileNotFoundError(f'No directory: {folder}')
    #if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
    #    raise ValueError("The 'optimizer' must be an instance of tf.keras.optimizers.Optimizer.")
    if not callable(G_loss_fn):
        raise ValueError("The 'G_loss_fn' must be callable.")
    
    #Reading the imge filepaths from files residing in the respective folders
    training_path = f'{os.getcwd()}/training'
    eval_path = f'{os.getcwd()}/eval'
    if os.path.exists(training_path):
        train_data = [f'{training_path}/{f}' for f in os.listdir(training_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {training_path}')
    if os.path.exists(eval_path):
        eval_data = [f'{eval_path}/{f}' for f in os.listdir(eval_path) if f.endswith('.fits')]
    else:
        raise FileNotFoundError(f'No file at: {eval_path}')
    
    #Creation of folders
    try:
        results_folder = os.path.join(os.getcwd(), 'models', network_name)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder, exist_ok=True)
        os.chdir(results_folder)
    except Exception as err:
        logging.warning(f'an error occurred while finding the output path: {err}')
        return
    name_txt_file = "train.txt"
    name_txt_file_eval = "eval_loss.txt"
    name_txt_file_epoch ="train_epoch.txt"

    # Model definition
    model = network(input_shape, **kwargs_network)
    optimizer = optimizer(learning_rate=learning_rate)

    # Checkpoints
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # Restoring checkpoints
    start_epoch = load_checkpoint(checkpoint, checkpoint_dir, start_from_best, start_from_last, start_from_scratch)

    # Definition of parameters
    G_current = []
    best_loss = float('inf')

    #Start Training
    for epoch in range(start_epoch + 1, n_epochs + 1):
        begin = time.time()
        epoch_loss = []
        epoch_eval_loss = []

        # change_learning_rate[0] contains the epoch when the change should occur
        # change_learning_rate[1] cotains the new value for the learning rate
        if epoch > change_learning_rate[0]:
            learning_rate = change_learning_rate[1]
            optimizer.learning_rate.assign(learning_rate)

        temp = np.random.permutation(train_data) 
        for cnt, (result, filepath) in enumerate(data_augment(images=temp, **kwargs_data)):
            st = time.time()

            result = one_step_training(model, result, G_loss_fn, optimizer)
            if result is None:
                continue
            gt_patch, in_patch, predicted, G_current, exp = result

            epoch_loss.append(G_current.numpy())
            print(f"{epoch} {cnt} Loss={np.mean(G_current):.3f} Time={time.time() - st:.3f} Exposure={exp:.0f}")

            # Periodic logging
            if cnt % max(1, ((len(temp) * period_percent) // 100)) == 0:
                text_write(name_txt_file, epoch, cnt, epoch_loss[-1], np.mean(epoch_loss), time.time()-st, filepath.split('/')[-1])

            # Validation
            if cnt % max(1, ((len(temp) * eval_percent) // 100)) == 0:
                time_eval = time.time()
                result = validate(model, eval_data,  G_loss_fn, kwargs_data)
                if result is not None:
                    eval_current, eval_exp = result
                    diff = eval_current.numpy().mean()
                    if diff < best_loss:
                        checkpoint_path = f'{checkpoint_dir}/best_model_{epoch:04d}.ckpt'
                        #if os.path.exists(checkpoint_dir):
                            #for file in os.listdir(checkpoint_dir):
                            #    if f'best_model_' in file:
                            #        try:
                            #            os.remove(os.path.join(checkpoint_dir, file))
                            #        except Exception as err:
                            #            logging.warning(f'an error occurred while deleting the old best model files: {err}')
                        checkpoint.save(checkpoint_path)
                        best_loss = diff

                    epoch_eval_loss.append(eval_current.numpy())
                    text_write(name_txt_file_eval, epoch, cnt, eval_current.numpy(), np.mean(epoch_eval_loss), time.time()-time_eval, filepath.split('/')[-1])
            
            # Save outputs periodically
            if epoch % save_freq == 0 and cnt % max(1, ((len(temp) * period_save) // 100)) == 0:
                if not os.path.exists(f"./results/{epoch:04d}"):
                    os.makedirs(f"./results/{epoch:04d}", exist_ok=True)
                sv_name = "/" + filepath.split('/')[-1][:-5]
                save_fits(gt_patch[0,:,:,0], sv_name, f"./results/{epoch:04d}")
                save_fits(predicted[0].numpy(), sv_name + "_output", f"./results/{epoch:04d}")
                save_fits(in_patch[0,:,:,0], sv_name + "_noise", f"./results/{epoch:04d}")

        # Save model checkpoints
        if epoch % save_freq == 0:
            checkpoint_path = f'{checkpoint_dir}/model_{epoch:04d}.ckpt'
            checkpoint.save(checkpoint_path)
        text_write(name_txt_file_epoch, epoch, len(train_data), np.mean(epoch_loss),  '0', time.time()-begin, filepath.split('/')[-1])

def __main__():

    kwargs_data = {
        'ps':256,
        'steps':100,
        'dk':7, 
        'ron':3,
        'patch_noise':False,
        'ratio':0.5,
        'save':False,
        'sv_name':None,
        'path':None,
        'patch_noise':False, 
        'type_of_image':'SCI',
        'low' : 60,
        'high' : 10000
    }
    kwargs_network = {
        'depth':5,
        'kernel_size':3,
        'filter_size':2,
        'pooling_size':2,
        'n_of_initial_channels':32, 
        'func':tf.keras.layers.LeakyReLU,
        'batch_normalization':True,
        'exp' : None,
        'exp_time':None
    }

    input_shape = (256, 256, 1)
    folder = './original_dataset'
    n_epochs = 100
    network_name='model'
    learning_rate=1e-4
    optimizer=Adam
    change_learning_rate=(2000, 1e-5)
    G_loss_fn=tf.keras.losses.MeanAbsoluteError()
    start_from_scratch=False
    start_from_best=True
    start_from_last=False
    save_freq=500
    period_percent=10
    eval_percent=20
    period_save=30

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train_network(input_shape, folder, n_epochs, kwargs_data, kwargs_network,
                  network_name=network_name, learning_rate=learning_rate,
                  optimizer=optimizer,change_learning_rate=change_learning_rate,
                  G_loss_fn=G_loss_fn, start_from_scratch=start_from_scratch,
                  start_from_best=start_from_best, start_from_last=start_from_last,
                  save_freq=save_freq, period_percent=period_percent,
                  eval_percent=eval_percent, period_save=period_save
                  )
__main__()