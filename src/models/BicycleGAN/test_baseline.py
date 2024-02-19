import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html


# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    #print('data slice i: {}'.format(data))
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz) # nz = latent vector size
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            # images = [real_A, real_B, fake_B]
            # names = ['input', 'ground truth', 'encoded']  # adding image_mask_paths = [P_pre, P_post]
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'synthesized']
        else:
            images.append(fake_B)
            names.append('random_synthesized_sample%2.2d' % nn)
            # saving also the masks
            #images.append(P_post)
            #names.append('random_sample_P_post%2.2d' % nn)

    img_path = 'input_%3.3d' % i
    name_file = model.get_image_paths()[0][12:-4]
    #print('Saving images in webpage {}\n names {}\n img_path {}, aspect_ratio {}, crop_size:{}'.format(webpage, names, img_path, opt.aspect_ratio, opt.crop_size))
    save_images(webpage, images, names, name_file, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)


    # ToDo: Adapt the file structure of outputs to fit the file structure of team
    # of eie_xview_processed:
    # results/<model>/<modality>/<split>/<im_name> with, for example,
    # model = pix2pix_scratch_1024 ; 
    # modality = flood_mask; 
    # split = Train_A ; 
    # im_name = hurricane-florence_00000172_pre_disaster.png 
webpage.save()
