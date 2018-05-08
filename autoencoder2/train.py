import numpy as np
import h5py
import os
import random
from patcher import Patcher
import threading
import gc
from multiprocessing import Process, Queue, Value
from PIL import Image
import sys
import time

def log(settings, lvl, msg):
    verbosity = settings['verbosity']

    if lvl <= verbosity:
        print(msg)
    sys.stdout.flush()

if __name__ == '__main__':
    import utils

    settings = utils.settings
    patch_queue = Queue(settings['train']['preload_max'])

    get_times = []
    yield_times = []

    img_folder = settings['data_path'] + '/' + settings['train']['img_folder']
    imgs = os.listdir(img_folder)
    nimg = len(imgs)
    random.shuffle(imgs)

def patch_generator(settings, proc_data):
    id = proc_data['id']
    num_gen = proc_data['num_gen']
    
    log(settings, 5, 'Generator process {}/{} started.'.format(id+1, num_gen))

    imgs = proc_data['imgs']
    nimg = len(imgs)
    done = proc_data['done']
    patch_queue = proc_data['patch_queue']

    img_folder = settings['data_path'] + '/' + settings['train']['img_folder']
    lbl_folder = settings['data_path'] + '/' + settings['train']['lbl_folder']
    patch_size = settings['patch_size']
    batch_size = settings['train']['batch_size']
    synth = settings['preprocess']['synthetic']

    if synth:
        bg_folder = settings['data_path'] + '/' + settings['preprocess']['synth_folder'] + '/'

    idx = id
    
    while done.value == 0:
        train_img = imgs[idx]

        train_lbl = lbl_folder + '/' + train_img.replace(settings['train']['img_ext'], settings['train']['lbl_ext'])
        train_img = img_folder + '/' + train_img

        if (not os.path.exists(train_lbl)) or (not os.path.exists(train_img)):
            continue

        log(settings, 7, 'Patching...')
        patcher = Patcher.from_image(settings, train_img, train_lbl, _dim=(patch_size, patch_size))
        patches, labels = patcher.patchify(settings)

        log(settings, 7, 'Enqueue...')
        patch_queue.put((patches, labels))

        if synth == True:
            i = 0
            bg_imgs = os.listdir(bg_folder)
            patch_size = settings['patch_size']
            synth_imgs = []
            for i in range(len(patches)):
                patch = patches[i]
                label = labels[i]
            
                rand = random.randint(0, len(bg_imgs) - 1)
                bg = Image.open(bg_folder + bg_imgs[rand])
                bg_data = np.asarray( bg )
            
                i0 = random.randint(0, bg_data.shape[0] - patch_size - 1)
                j0 = random.randint(0, bg_data.shape[1] - patch_size - 1)
            
                bg_patch = bg_data[i0:i0+patch_size, j0:j0+patch_size]
                synth_patch = np.zeros((patch_size, patch_size, 3))
                synth_patch[:,:,:] = bg_patch
                indices = np.where(label.reshape((patch_size, patch_size)) > 0)

                synth_patch[indices] = patch[indices]

                synth_imgs.append(synth_patch)

            patch_queue.put((synth_imgs,labels))


        idx = (idx + num_gen) % nimg

def data_feeder(settings):
    while True:
        log(settings, 7, 'Dequeue...')
        start = time.time() * 1000
        patches, labels = patch_queue.get()
        end = time.time() * 1000

        get_times.append(end - start)

        log(settings, 7, 'Yield...')
        start = time.time() * 1000
        yield np.array(patches), np.array(labels)
        end = time.time() * 1000

        yield_times.append(end - start)

if __name__ == '__main__':
    # Start the loading thread asap
    log(settings, 3, 'Starting generator processes...')
    gen_procs = []
    done = Value('i', 0)

    for i in range(settings['train']['num_gen']):
        proc_data = {
            'id': i,
            'num_gen': settings['train']['num_gen'],
            'done': done,
            'imgs': imgs,
            'patch_queue': patch_queue
        }

        gen_proc = Process(target=patch_generator, args=(settings, proc_data))
        gen_proc.start()
        gen_procs.append(gen_proc)

    log(settings, 3, 'Building Model...')
    from models import *
    model = pick_model(settings)

    if settings['verbosity'] >= 5:
        model.summary()

    model.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

    log(settings, 0, 'Training...')
    
    report_samples = settings['train']['report_samples']
    samples_per_epoch = report_samples * settings['train']['batch_size']
    tot_epoch = settings['train']['nb_epochs'] * nimg / report_samples

    log(settings, 5, 'Images per sample : {}'.format(settings['train']['batch_size']))
    log(settings, 5, 'Samples per report: {}'.format(report_samples))
    log(settings, 5, 'Samples per epoch : {}'.format(samples_per_epoch))
    log(settings, 5, 'Total # of epochs : {}'.format(tot_epoch))

    model.fit_generator(data_feeder(settings), steps_per_epoch=report_samples, nb_epoch=tot_epoch, verbose=2, nb_worker=1)

    done.value = 1

    avg_yield_time = np.mean(np.array(yield_times))

    print('Time stats:')
    print('Average dequeue time: {}ms'.format(np.mean(np.array(get_times))))
    print('Average yield time: {}ms'.format(avg_yield_time))
    print('Approx patch/sec: {}'.format(settings['train']['batch_size'] / avg_yield_time * 1000))

    model.save_weights(settings['weights_file'])

    print('Waiting for generator threads to stop...')

    for p in gen_procs:
        # TODO: Find a better solution:
        p.terminate()
