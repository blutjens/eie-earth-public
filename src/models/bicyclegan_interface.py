""" Wrapper for using models from the BicycleGAN repo.
"""

import os
import sys

import torch

from base_interface import MLProjectBase

# Fetch the models for wrapping, first adding the submodule location to the Python path
# import pdb; pdb.set_trace()
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'models/BicycleGAN'))
from BicycleGAN.models.bicycle_gan_model import BiCycleGANModel
from BicycleGAN.models import create_model


class WrappedBiCycleGAN(BiCycleGANModel, MLProjectBase):
    """

    """
    def __init__(self, params):
        """

        Params
        ------
        params : dict

        """
        # Translate our params to options expected by BiCycleGAN repo
        self.opt = self._params2opt(params)
        self.model = create_model_from_opt(self.opt)

    def _set_opt_for_eval(self):
        self.opt.num_threads = 1   # test code only supports num_threads=1
        self.opt.batch_size = 1   # test code only supports batch_size=1
        self.opt.serial_batches = True  # no shuffle

    def create_model_from_opt(self, opt):
        """ Use BiCycleGANModel factory function to init model.
        """
        self.model = create_model(opt)
        self.model.setup(opt)
        return self.model

    # TODO: implement training from https://github.com/junyanz/BicycleGAN/blob/master/train.py#L30
    def fit(self):
        """
        """
        # Create a visualizer that display/save images and plots
        visualizer = Visualizer(opt)

        total_train_iters = 0

        # Run training; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        for epoch in range(epoch_count, niter + niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                iter_start_time = time.time()  # timer for computation per iteration
                if total_train_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_train_iters += opt.batch_size
                epoch_iter += opt.batch_size

                # Unpack data from dataset and apply preprocessing
                self.model.set_input(data)
                if not self.model.is_train():
                    print('skip this batch')
                    continue
                # Calculate loss functions, get gradients, update network weights
                self.model.optimize_parameters()

                if total_train_iters % opt.display_freq == 0:
                    # Display images on visdom and save images to a HTML file
                    save_result = total_train_iters % opt.update_html_freq == 0
                    self.model.compute_visuals()
                    visualizer.display_current_results(self.model.get_current_visuals(), epoch, save_result)

                if total_train_iters % opt.print_freq == 0:
                    # Print training losses and save logging information to the disk
                    losses = self.model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

                if total_train_iters % opt.save_latest_freq == 0:
                    # Cache our latest self.model every <save_latest_freq> iterations
                    print("saving the latest self.model (epoch {}, total_train_iters {})".format(epoch, total_train_iters))
                    self.model.save_networks('latest')

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:              # cache our self.model every <save_epoch_freq> epochs
                print("saving the self.model at the end of epoch {}, iters {}".format(epoch, total_train_iters))
                self.model.save_networks('latest')
                self.model.save_networks(epoch)

            print("End of epoch {} / {} \t Time Taken: {} sec".format(epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            self.model.update_learning_rate()                     # update learning rates at the end of every epoch.
        print("Done training")

    def predict(self):
        """
        """
        # Load a model for evaluation
        self._set_opt_for_eval()
        model = create_model_from_opt(self.opt)
        model.eval()
        print("Loading model {}".format(opt.model))

        # Create webpage for viewing
        web_dir = os.path.join(self.opt.results_dir, self.opt.phase + '_sync' if self.opt.sync else self.opt.phase)
        webpage = html.HTML(web_dir, "Training = {}, Phase = {}, Class = {}".format(self.opt.name, self.opt.phase, self.opt.name))

        # Sample random z
        if self.opt.sync:
            z_samples = self.model.get_z_random(self.opt.n_samples + 1, self.opt.nz)

        # Test stage
        for i, data in enumerate(islice(dataset, self.opt.num_test)):
            model.set_input(data)
            print('process input image %3.3d/%3.3d' % (i, opt.num_test))
            if not self.opt.sync:
                z_samples = self.model.get_z_random(self.opt.n_samples + 1, self.opt.nz)
            for nn in range(self.opt.n_samples + 1):
                encode = nn == 0 and not self.opt.no_encode
                real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
                if nn == 0:
                    images = [real_A, real_B, fake_B]
                    names = ['input', 'ground truth', 'encoded']
                else:
                    images.append(fake_B)
                    names.append('random_sample%2.2d' % nn)

            img_path = 'input_%3.3d' % i
            save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

        webpage.save()
        print("Testing model done, saved images in {}".format(img_path))
        return img_path
