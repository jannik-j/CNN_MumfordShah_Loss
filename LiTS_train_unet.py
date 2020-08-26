import time
from options.train_options import TrainOptions
from LiTS_getDatabase_unet import DataProvider_LiTS
from models.models import create_model
from util.visualizer import Visualizer
from math import ceil

"""
Aus der originalen Implementation
Skript zum Training eines Netzwerks auf dem LiTS Datensatz
"""

# Parsen der Kommandozeilenargumente
opt = TrainOptions().parse()

input_min       = 0
input_max       = 400

# Laden der Trainingsdaten durch die Klasse DataProvider_LiTS
data_train = DataProvider_LiTS(opt.inputSize, opt.fineSize, opt.segType, opt.semi_rate, opt.input_nc,
                               opt.dataroot, a_min=input_min, a_max=input_max, mode="train")
dataset_size = data_train.n_data
print('#training images = %d' % dataset_size)
training_iters = int(ceil(data_train.n_data/float(opt.batchSize)))

total_steps = 0
# Erstellung und Initialisierung eines neuen Netzwerks
model = create_model(opt)
# Initilaisierung einer visdom-Session
visualizer = Visualizer(opt)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    """Schleife über die Epochen"""
    epoch_start_time = time.time()

    for step in range(1, training_iters+1):
        """Schleife über die Minibatches"""
        # Laden der Daten eines Minibatches
        batch_x, batch_y, path = data_train(opt.batchSize)
        data = {'A': batch_x, 'B': batch_y,
                'A_paths': path, 'B_paths': path}

        iter_start_time = time.time()
        visualizer.reset()
        total_steps += 1
        model.set_input(data)
        # Auswertung des Netzwerks und Durchführung eines Schritts des Optimierers
        model.optimize_parameters()

        # Visualisierung des Trainingsfortschritts über visdom
        if step % opt.display_step == 0:
            save_result = step % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, step, training_iters, errors, t, 'Train')

        if step % opt.plot_step == 0:
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, step / float(training_iters), opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    # Speicherung der aktuellen Netzwerkparameter
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # Update der Lernrate
    model.update_learning_rate()
