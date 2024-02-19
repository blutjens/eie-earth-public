import argparse

# Get im ids, masks, and load images
from PIL import Image
# TODO: remove this line and check in flood_color/conditional_binary_sloshed why the images are truncated
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as T
import os, glob
import numpy as np

# Plotting
import matplotlib.transforms as mtrans
import matplotlib.pyplot as plt

# Metrics
from lpips_pytorch import LPIPS, lpips

# Initialize lpips metric 
# TODO: init object in main() fn; not globally
lpips = LPIPS(
    net_type='alex', # Choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1', # Currently, v0.1 is supported
)

# TODO: creates masks from (scratch_1024_plus) in ./segmentation_model -> temporary_evaluatepy_paralellwork_segmentmasks.ipynb
def IoU_metric(gt_mask, gen_mask, threshold=0.5):
    '''Calculates IoU over two given masks
    Input: 
    gt_mask (np.array(h,w)): Non-binary ground truth segmentation mask with positive class == 0 and negative class == {1,255}
    gen_mask (np.array(h,w)): Non-binary generated segmentation mask with positive class == 0 and negative class == {1,255}'''
    
    # Make the image binary
    gt_mask = np.where(gt_mask  < threshold , 0, 1)
    gen_mask = np.where(gen_mask < threshold , 0, 1)

    # TODO: why is intersect <.5 and not == 0 ?
    #Chris: after making it binary, == 0 works the same. In case it is not binary, using threshold here instead of 0, makes it work too.
    intersect = (gt_mask + gen_mask < threshold ).sum()
    # TODO: why is union (1-...) > .5 and not (gt_mask + gen_mask) <= 1.? 
    # Chris : gt_mask + gen_mask <= 1 is not the union. We want to find those pixels where either gt or gen or both are 0. 
    # The easiest is to invert the masks (1-mask), and them add them. If the addition is 1 or more (or thershold), 
    # then it means any of the two (or the two) were 1.
    union = ((1-gt_mask)+(1-gen_mask ) > threshold ).sum()
    IoU = intersect/union
    
    return IoU

def FVPS_Score1(IoU, LPIPS, epsilon=0.00000000001):
    # Calculates Flood Visualization Plausability Score (FVPS1) as harmonic mean of IoU and 1-LPIPS
    if LPIPS > 1:
        LPIPS = 1
        
    return 2/(1/(IoU+epsilon)+1/(1-LPIPS+epsilon))

def gen_baseline_ims(n_pics=20, sloshed = True, plot = True, config=None):
    """
    Generate folder of baseline images that paint flood brown into pre-flooding images
    Args:
        n_pics int: maximum number of images
        sloshed bool: Generate sloshed (i.e., downscaled) imagery if true
        plot bool: Generate plot if true
        config dict(): 
    """
    # TODO: extend do more than just Test split

    # Define flood color
    # TODO: Compute specific flood color for each event
    flood_color = "#998d6f"

    # Define and create directories to store image, mask, and plot
    im_baseline_dir = "../" + config['im_baseline_dir'] 
    if sloshed:
        im_baseline_dir += "_sloshed/"
    else:
        im_baseline_dir += "/"
        
    split_prefix = config['split_prefix']# test-set_test_images_labels_targets_test_images_"
    events = [config['event']]# ['hurricane-harvey_']#, 'hurricane-florence_', 'hurricane-michael_', 'hurricane-matthew_'] # 
    suffix_baseline = config['suffix_baseline'] # 'pre_disaster_synthesized_image.jpg'
    suffix_mask = config['suffix_mask'] # 'pre_disaster_synthesized_image_fake_B.png'
    
    plt_dir = "../figures/flood_color/"+config['model_name'] #
    if not os.path.exists(im_baseline_dir):
        os.makedirs(im_baseline_dir)
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)

    # Get all pre-flooding images 
    # Get all seg masks
    for event in events:
        print(event)
        #im_paths = get_im_paths(n_pics=n_pics, every=1, model="", experiments=[], event=event)
        #im_paths = get_im_paths(n_pics=n_pics, every=1, models_and_experiments=[], event=event)
        im_paths = get_im_paths(n_pics=n_pics, every=1, 
                        models_and_experiments=["Pix2pixHD/"+config['model_name'].replace('/','')], 
                         event=config['event'], im_suffix=config['im_suffix'], 
                         suffix_pre=config['suffix_pre'], suffix_post=config['suffix_post'], 
                         dataroot=config['dataroot'], new_paths=True, verbose=False)

        im_shape = (1024, 1024)
        if plot: fig, axs = plt.subplots(n_pics, 3, figsize = (4.*2, 2.5*n_pics), dpi=300)

        # Generate baseline image for each image
        for i, im_path in enumerate(im_paths):
            if i%20==0: print(i)
            # Init generated baseline image
            im_baseline = Image.new("RGB", im_shape, flood_color)

            # Load images and masks
            ims_pil, _ = load_ims(im_path, ['pre', 'post'], is_mask=False, im_shape=im_shape)
            im_pre_np = np.asarray(ims_pil['pre'])
            if sloshed:
                _ , masks = get_binary_masks(im_path, ['post'], im_shape)
            else:
                masks, _ = get_binary_masks(im_path, ['post'], im_shape)
            mask_rgb = np.repeat(masks['post'][:, :, np.newaxis], 3, axis=2)

            # Replace mask in pre-flood image with flood color
            im_baseline = np.where(np.logical_not(mask_rgb), np.asarray(im_baseline), im_pre_np)

            if plot:
                axs[i,0].imshow(np.asarray(ims_pil['pre']))
                axs[i,0].imshow(masks['post'], alpha=.3, cmap='viridis_r', vmin=0, vmax=np.max(masks['post']))
                axs[i,1].imshow(im_baseline)#p.multiply(np.asarray(im_baseline), masks['post'], dims=0))
                axs[i,2].imshow(np.asarray(ims_pil['post']))

            # Save image
            im_baseline_pil = Image.fromarray(np.uint8(im_baseline))
            im_id_txt = '%08d'%(get_im_id(im_path['path_pre']))
            im_path = os.path.join(im_baseline_dir, split_prefix + 'B/' + event + im_id_txt + "_"+ suffix_baseline)
            im_baseline_pil.save(im_path, "JPG") # TODO: reduce runtime of this line; currently ~.5sec
            # Save mask (for evaluation script)
            mask_rgb_pil = Image.fromarray(np.uint8(mask_rgb*255.))
            mask_path = os.path.join(im_baseline_dir, split_prefix + 'mask/' + event + im_id_txt + "_"+ suffix_mask)
            mask_rgb_pil.save(mask_path, "PNG")
            
        # Save in new folder
        if plot:
            axs[0, 0].set_title('pre')
            axs[0, 1].set_title('colored pre')
            axs[0, 2].set_title('post')
            [axi.set_axis_off() for axi in axs.ravel()]
            fig.tight_layout()
            plt.show()
            fig.savefig(os.path.join(plt_dir, event + str(n_pics)))

def get_im_id(im_path):
    # Returns image id (int) from image path (str)
    im_id = [int(s) for s in im_path.split('_') if s.isdigit()] # Extract id from filename
    if len(im_id) > 1: raise 'filename contains multiple image ids' # Check that only one id has been extracted
    return int(im_id[0])

def get_im_ids(im_path_wildcard, n_pics, every=1, start=None, verbose=False):
    '''Returns the first n_pics image ids and the filename str that match the wildcard
    Input:
    im_path_wildcard (str): wildcard string to all queried images
    n_pics (int): number of queried im_ids; enter high number to query all images (TODO: make querying all images more intuitive)
    every (int): Skips every n pictures, useful for more varied plotting
    Output:
    im_ids (list(int)): The first sorted n_pics image ids that match the wildcard
    im_ids_str (list(str)): The associated image id as reflect in the filename'''
    print('wildcard', im_path_wildcard)
        
    im_ids = []
    for i, im_name in enumerate(glob.glob(im_path_wildcard)):
        im_ids.append(get_im_id(im_name)) 
    im_ids.sort() 
    
    im_ids_str = ['%08d_'%(im_id) for im_id in im_ids]
    im_ids = im_ids
    im_ids_str = im_ids_str
        
    if verbose: print('sorted im_ids', im_ids)
    if verbose: print('max num images: ', len(im_ids_str))
    # Clip im_ids to n_pics
    if n_pics == 'all':
        n_pics = len(im_ids_str)
    n_pics = np.minimum(n_pics, len(im_ids_str))
    idx = np.arange(0,n_pics*every,every).tolist()
    
    if start is not None:
        print(start)
        idx = np.arange(start,start+n_pics*every,every).tolist()
        print(idx)
        
    im_ids = [im_ids[i] for i in idx]
    im_ids_str = [im_ids_str[i] for i in idx]

    return im_ids, im_ids_str

def get_im_paths(n_pics=5, every=1, models_and_experiments=["baseline/flood_color", "Pix2pixHD/conditional_binary"], 
                 event='hurricane-harvey_', start=None, im_suffix='_disaster', 
                 suffix_pre='pre_disaster.png', suffix_post='post_disaster.png', 
                 dataroot='data/eie_floods/naip/xbd/images/', new_paths=False,
                 verbose=False):
    '''Retrieves relative filepaths to queried imagery
    Input
    n_pics (int): Number of queried images
    # experiments (list(str)): Folder names of queried experiments
    models_and_experiments list(str): directory in format "model/experiment"
    event (string): Name of queried event
    im_suffix (string): image suffix, e.g., _disaster or _melt
    dataroot (string): root path to data
    new_paths (boolean): Use new clean paths if True else use old paths
    Output
    im_paths (dict()): Dictionary with relative filepath to pre, post, and generated image'''
    
    im_dir = '../results/'
    dataroot = dataroot if new_paths else 'data/xBD/'
    im_dir_pre = '../' + dataroot
    im_dir_post = '../' + dataroot
    mask_dir_post = '../' + dataroot
    mask_dir_gen = '../results/'

    model_and_experiment_dirs_gen = [directory for directory in models_and_experiments]
    #model_dir_gen = model + '/'
    #experiment_dirs_gen = [experiment + '/' for experiment in experiments]
    
    split_dir_pre = 'test_A/'
    split_dir_post = 'test_B/'
    split_dir_gen = 'test_B/' if new_paths else '/'
    split_dir_mask_post = 'test_mask/'
    split_dir_mask_gen = 'test_mask/' if new_paths else '/'
    
    split_prefix = '' if new_paths else 'test-set_test_images_labels_targets_test_images_'
    
    suffix_gen = suffix_post.replace('.png','.jpg') if new_paths else 'pre_disaster_synthesized_image.jpg'
    suffix_mask_post = suffix_post if new_paths else 'post_disaster_fake_B.png'
    suffix_mask_gen = suffix_post if new_paths else 'pre_disaster_synthesized_image_fake_B.png'

    # Get im_ids of all generated images (by just looking at the post flooding filenames)
    im_ids, im_ids_str = get_im_ids(
        im_path_wildcard = im_dir_post + split_dir_post + split_prefix + event + "*" + suffix_post,
        n_pics = n_pics,
        every=every,
        start=start,
        verbose=verbose
    )
    
    im_paths = []
     
    for i, im_id in enumerate(im_ids_str):
        im_paths_one_im  = {
            'path_pre' : im_dir_pre + split_dir_pre + split_prefix + event + im_id  + suffix_pre,
            'path_post' : im_dir_post + split_dir_post + split_prefix + event + im_id  + suffix_post,
            'mask_path_post' : mask_dir_post + split_dir_mask_post + split_prefix + event + im_id + suffix_mask_post,
            }
        for e, model_and_experiment_dir_gen in enumerate(model_and_experiment_dirs_gen):
            #model, experiment = model_and_experiment_dir_gen.split("/")
            #im_paths_one_im['path_gen_' + experiment] = im_dir + model_dir_gen + experiment_dirs_gen[e] + split_dir_gen + split_prefix + event + im_id  + suffix_gen
            #im_paths_one_im['mask_path_gen_' + experiment] =  mask_dir_gen + model_dir_gen + experiment_dirs_gen[e] + split_dir_mask_gen + split_prefix + event + im_id + suffix_mask_gen
            im_paths_one_im['path_gen_' + model_and_experiment_dir_gen] = im_dir + model_and_experiment_dir_gen + "/" + split_dir_gen + split_prefix + event + im_id + suffix_gen
            im_paths_one_im['mask_path_gen_' + model_and_experiment_dir_gen] =  mask_dir_gen + model_and_experiment_dir_gen + "/" + split_dir_mask_gen + split_prefix + event + im_id + suffix_mask_gen
        
        im_paths.append(im_paths_one_im)

    return im_paths

def load_ims(im_paths, im_plt_titles, is_mask=False, im_shape=(1024,1024)):
    '''Loads images from paths
    Input:
    im_paths dict(str): Paths to RGB images (not RGBA)
    im_plt_titles list(str): experiment names
    is_mask (bool): if true, load masks
    im_shape tuple(int, int): desired shape (h,w) of image in px
    Output
    ims_pil dict(PIL.Image): images, with vals/dims from source file'''
    ims_pil = {}
    for i, im_plt_title in enumerate(im_plt_titles):
        im_key = 'path_' + im_plt_title if not is_mask else 'mask_path_' + im_plt_title
        try:
            ims_pil[im_plt_title] = Image.open(im_paths[im_key])
            # Convert RGBA to RGB
            if ims_pil[im_plt_title].getbands() == ('R', 'G', 'B', 'A'):
                ims_pil[im_plt_title] = ims_pil[im_plt_title].convert('RGB')
        except:
            if not im_key == 'mask_path_pre': print('[Warning] Images of key %s not found, but still respected in avg eval:'%(im_key))
            # Non-existing images/masks are loaded as all white (ie., all non-flooded)
            ims_pil[im_plt_title] = Image.new('RGB', im_shape)#, (800,1280), (255, 255, 255))

    return ims_pil, im_shape

def get_binary_masks(mask_paths, im_plt_titles, mask_shape=(1024,1024), mask_sloshed_res=(18,18)):
    '''Returns binary segmentation mask of given image path 
    Input
    mask_paths (str): Paths to mask images
    im_plt_titles list(str): experiment names
    mask_shape tuple(int, int): desired shape (h,w) of mask in px
    mask_sloshed_res tuple(int,it): desired resolution (h,w) of sloshed mask in px
    Output:
    masks_binary dict(im_plt_title: np.array(w,h,1)): First channel of segmentation mask; thresholded to binary {0,1}
    masks_sloshed dict(im_plt_title: np.array(mask_sloshed_shape,1)): downsampled segmentation mask'''

    masks_binary = {}
    masks_sloshed = {}
    masks_pil, _ = load_ims(mask_paths, im_plt_titles, is_mask=True, im_shape=mask_shape)
    for i, im_plt_title in enumerate(im_plt_titles):
        # TODO: import conversion from src.models.pix2pixhd.dataset.base_dataset.py get_transforms()
        masks_gray = masks_pil[im_plt_title].convert("L") 
        masks_binary_pil =  masks_gray.convert("1", dither=None)
        masks_sloshed_pil = masks_binary_pil.resize(mask_sloshed_res, Image.BICUBIC) 
        masks_sloshed_pil = masks_sloshed_pil.resize(mask_shape, Image.NEAREST)

        masks_binary[im_plt_title] = np.asarray(masks_binary_pil).astype('float32')#/255.
        masks_sloshed[im_plt_title] = np.asarray(masks_sloshed_pil).astype('float32')#/255.

    return masks_binary, masks_sloshed

class Eval_plot:
    def __init__(self, plt_seg_masks=True, n_pics=5, 
                 im_plt_titles=['pre', 'post', 'conditional_binary'], 
                 models_and_experiments_abbr=None,
                 dpi=50, fontsize=12, plt_mask_w_background=True):
        
        self.plt_seg_masks = plt_seg_masks
        self.plt_mask_w_background = plt_mask_w_background # if true plots masks with background

        self.im_plt_titles=im_plt_titles
        if models_and_experiments_abbr == None:
            self.im_plt_titles_abbr = im_plt_titles
        else:
            # TODO: define experiment abbreviations
            if not ('pre'==self.im_plt_titles[0] and 'post'==self.im_plt_titles[1]):
                raise NotImplementedError("TODO: implement abbreviated naming of plot_eval_metrics_table if 'pre' 'post' are not listed in experiments")
            self.im_plt_titles_abbr = ['pre', 'post'] + models_and_experiments_abbr
        
        self.fontsize = fontsize
        self.dpi = dpi
        n_rows = 2*n_pics if plt_seg_masks else n_pics
        # TODO: find out how to plot table without having to do a +3 placeholder here
        self.fig, self.axs = plt.subplots(n_rows, len(self.im_plt_titles) + 3, 
                                          figsize = (4.*len(self.im_plt_titles), 3*n_rows), dpi=self.dpi)

        if n_pics == 1 and not plt_seg_masks:
            self.axs = self.axs.reshape(1,len(self.im_plt_titles))

        self.counter_plt = 0

    def draw_horizontal_line_between_subplots(self):
        # Draws a horizontal line between every 2nd subplot
        # Source: https://stackoverflow.com/questions/26084231/draw-a-separator-or-lines-between-subplots?rq=1
        
        # Get the bounding boxes of the axes including text decorations
        r = self.fig.canvas.get_renderer()
        get_bbox = lambda ax: ax.get_tightbbox(r).transformed(self.fig.transFigure.inverted())
        bboxes = np.array(list(map(get_bbox, self.axs.flat)), mtrans.Bbox).reshape(self.axs.shape)

        #Get the minimum and maximum extent, get the coordinate half-way between those
        ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(self.axs.shape).max(axis=1)
        ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(self.axs.shape).min(axis=1)
        ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

        # Draw a horizontal lines at those coordinates
        for i, y in enumerate(ys):
            if i%2==1:
                line = plt.Line2D([0,1],[y,y], transform=self.fig.transFigure, color="black")
                self.fig.add_artist(line)

    def plot_im_row(self, ims_pil):
        # Plot row of images
        # Input:
        # ims_pil (dict()): dictionary of images from experiments
        for i, im_plt_title in enumerate(self.im_plt_titles):
            self.axs[self.counter_plt, i].imshow(np.array(ims_pil[im_plt_title]))
            self.axs[self.counter_plt, i].set_title(self.im_plt_titles_abbr[i])

        self.counter_plt += 1

    def plot_seg_mask_row(self, ims_pil, masks, show_mask_w_background=False):
        '''Plot row image and overlayed segmentation mask
        Input:
        ims_pil (dict(PIL.Image)): images with experiment name as key
        masks (dict(np.array)): masks with experiment name as key'''
        if self.plt_seg_masks:
            for i, im_plt_title in enumerate(self.im_plt_titles):
                if im_plt_title == 'pre': # Pre image does not contain a mask; so skip
                    self.axs[self.counter_plt, i].imshow(np.asarray(ims_pil[im_plt_title]))
                    continue
                if show_mask_w_background: 
                    self.axs[self.counter_plt, i].imshow(np.asarray(ims_pil[im_plt_title]))
                    alpha=.5
                    cmap='viridis_r'
                else:
                    alpha=1.
                    cmap='Blues_r'

                self.axs[self.counter_plt, i].imshow(masks[im_plt_title], alpha=alpha, cmap=cmap, vmin=0, vmax=np.max(masks[im_plt_title]), interpolation='nearest')
                self.axs[self.counter_plt, i].set_title(self.im_plt_titles_abbr[i])

            self.counter_plt += 1
    
    def plot_eval_metrics_table(self, eval_metrics, metrics, experiments):
        # Adds evaluation metrics as table to the right side of plots 
        # Input
        # eval_metrics (dict()): to-plot evaluation metrics with experiment as key
        
        # Get column androw labels
        columns = metrics
        rows = [] 
        for i, experiment in enumerate(experiments):
            rows.append(self.im_plt_titles_abbr[2+i])
 
        # Fill in table data
        data = []
        for row_id, experiment in enumerate(experiments):
            data.append(['%.4f' % eval_metrics[metric + '_post_gen_' + experiment]  for metric in metrics])

        # Add a table at the bottom of the axes
        the_table = self.axs[self.counter_plt-1,-1].table(cellText=data,
                        rowLabels=rows, #rowColours=colors,
                        colLabels=columns,
                        loc='left', fontsize=12,
                        )
        #plt.subplots_adjust(right=-0.2)

    def plot_eval_metrics(self, eval_metrics, event, im_paths):
        '''Adds evaluation metrics as blank text to the right side of plots 
        Input
        eval_metrics (dict()): to-plot evaluation metrics with experiment as key
        event (str): event name
        im_paths (dict()): image path to each experiment'''
    
        for im_plt_title in self.im_plt_titles:
            # Image id to plot over evaluation metrics
            im_id_txt = event + '%08d'%(get_im_id(im_paths['path_' + im_plt_title]))

            transform = self.axs[self.counter_plt-1,-1].transAxes
            top = 0.95
            for k, eval_metric_name in enumerate(eval_metrics):
                eval_txt = eval_metric_name + '=%.4f'%(eval_metrics[eval_metric_name])
                self.axs[self.counter_plt-1,-1].text(1.05, top-k*.01*self.fontsize, eval_txt, transform=transform, fontsize=self.fontsize)#bbox=props)
            # Add image id to plot:
            self.axs[self.counter_plt-1,-1].text(1.05, 1.10, im_id_txt, transform=transform, fontsize=self.fontsize)#bbox=props)

    def display_and_save(self, plt_filename="", do_display=True):
        # Displays and saves plot
        # plt_filename (str): filename of plot; if None, doesn't save

        # Format plot
        [axi.set_axis_off() for axi in self.axs.ravel()]
        self.fig.tight_layout()
        #plt.subplots_adjust(right=0.2)
        #self.draw_horizontal_line_between_subplots()
        if do_display:
            self.fig.show()

        # Display and save
        if plt_filename is not None:
            self.fig.savefig(plt_filename)
        

#def eval_im(im_path, im_plt_titles, im_shape, experiments, metrics, metric_fns, eval_plot, plot=False):
def eval_im(im_path, im_plt_titles, im_shape, models_and_experiments, metrics, metric_fns, eval_plot, plot=False):
    '''
    # Evaluates and plots the experiment set of one image pair  
    # Input 
    # im_path dict(str: str): paths to all experiments of the given image 
    # im_plt_titles list(str): experiment names
    # im_shape tuple(int, int): desired shape (h,w) of mask in px
    # experiments list(str): experiment names
    # models_and_experiments list(str): directories in format ["model/experiment"]
    # metrics list(str): metric names
    # metric_fns list(fn): metric functions that match the metric names
    # eval_plot matplotlib.pyplot.plt(): plot of images and metrics
    # plot (bool): if true plot else don't plot
    # Output:
    # eval_metrics dict(str: float): value of evaluation metric/experiment
    # eval_plot matplotlib.pyplot.plt(): plot of images and metrics
    '''
    
    eval_metrics = {}

    # Load images and masks
    ims_pil, _ = load_ims(im_path, im_plt_titles, im_shape=im_shape)
    masks, _ = get_binary_masks(im_path, im_plt_titles, im_shape)

    # Compute evaluation metrics for each experiment     
    trf_to_pytorch = T.Compose([T.ToTensor(),]) # TODO: check which data preproc. lpips reqs (ToTensor normalizes to [.0, 1.])

    for metric, metric_fn in zip(metrics, metric_fns):
        # TODO: add baseline evaluation more beautiful
        # if metric == 'lpips': eval_metrics['lpips_post_pre'] = metric_fn(trf_to_pytorch(ims_pil['post']), trf_to_pytorch(ims_pil['pre'])).cpu().numpy()

        for model_and_experiment in models_and_experiments:
            if metric == 'lpips': eval_metrics[metric + '_post_gen_' + model_and_experiment] = metric_fn(trf_to_pytorch(ims_pil['post']), trf_to_pytorch(ims_pil['gen_' + model_and_experiment])).cpu().numpy()
            elif metric == 'IoU': eval_metrics[metric + '_post_gen_' + model_and_experiment] = metric_fn(masks['post'], masks['gen_' + model_and_experiment])
            elif metric == 'FVPS1': eval_metrics[metric + '_post_gen_' + model_and_experiment] = metric_fn(eval_metrics['IoU_post_gen_' + model_and_experiment],eval_metrics['lpips_post_gen_' + model_and_experiment])

        """
        for experiment in experiments:
            if metric == 'lpips': eval_metrics[metric + '_post_gen_' + experiment] = metric_fn(trf_to_pytorch(ims_pil['post']), trf_to_pytorch(ims_pil['gen_' + experiment])).cpu().numpy()
            elif metric == 'IoU': eval_metrics[metric + '_post_gen_' + experiment] = metric_fn(masks['post'], masks['gen_' + experiment])
            elif metric == 'FVPS1': eval_metrics[metric + '_post_gen_' + experiment] = metric_fn(eval_metrics['IoU_post_gen_' + experiment],eval_metrics['lpips_post_gen_' + experiment])
        """
    # Plot
    if plot:
        eval_plot.plot_im_row(ims_pil)
        eval_plot.plot_eval_metrics_table(eval_metrics, metrics, models_and_experiments)
        eval_plot.plot_seg_mask_row(ims_pil, masks) 

    return eval_metrics, eval_plot


def eval_main(events=['hurricane-harvey_'], models_and_experiments=['Pix2pixHD/conditional_binary'],
            models_and_experiments_abbr=['Pix2pixHD/bin'],
              #model='Pix2pixHD',
             n_pics=20,
            start=None,
             #experiments=['baseline', 'conditional_binary'], 
             metrics=['lpips', 'IoU'], metric_fns=[lpips, IoU_metric], 
             plot=False, plt_seg_masks=False, plt_mask_w_background=True, dpi=50,
             every=1, im_shape=(1024,1024),
             do_display = True, new_paths=False, config=None):
    '''
    """
    Evaluates and plots the experiment set of n_pics images of one event
    Args:
        events list(str): events filename name
        models_and_experiments list(str): directories in format ["model/experiment"]; also used as key in dictionaries
        models_and_experiments_abbr list(str): abbreviated names of directories; used for plotting
        model str: model name
        n_pics int: number of images to evaluate
        every int: evaluate every nth image
        im_shape tuple(int, int): desired shape (h,w) of mask in px
        metrics list(str): metric names
        metric_fns list(fn): metric functions that match the metric names
        plot bool: if true plot else don't plot
        plt_seg_masks bool: if true plot segmentation masks
        plt_mask_w_background bool: if true plot mask with 
        new_paths boolean: Use new clean paths if True else use old paths
        config dict(): Dictionary with most filepaths

    Returns:
        evals list(dict['model_and_experiment']:value): list of metrics per image 
    # ------------------------------------
    '''
    total_evals = []
    for event in events:
        # Get image paths of all experiment sets
        im_paths = get_im_paths(n_pics=n_pics, every=every,
                models_and_experiments=models_and_experiments,
                event=event,
                im_suffix=config['im_suffix'], 
                suffix_pre=config['suffix_pre'], suffix_post=config['suffix_post'], 
                dataroot=config['dataroot'], new_paths=new_paths)
        # im_paths = get_im_paths(n_pics, every, models_and_experiments, event, start, new_paths=new_paths)

        #im_plt_titles = ['pre', 'post'] + ['gen_' + experiment for experiment in experiments]
        im_plt_titles = ['pre', 'post'] + ['gen_' + m_and_e for m_and_e in models_and_experiments]
        #print('Experiments: ', im_plt_titles)
        print('Model + Experiments: ', im_plt_titles)
        print('Metrics: ', metrics)

        # Init plot
        eval_plot = Eval_plot(plt_seg_masks=plt_seg_masks, plt_mask_w_background=plt_mask_w_background, 
                              n_pics=n_pics, im_plt_titles=im_plt_titles, 
                              models_and_experiments_abbr=models_and_experiments_abbr,
                              dpi=dpi, fontsize=10) if plot else None

        ## Evaluate each image pair:
        evals = []
        for i, im_path in enumerate(im_paths):
            #eval_metrics, eval_plot = eval_im(im_path, im_plt_titles, im_shape, experiments, metrics, metric_fns, eval_plot, plot)
            eval_metrics, eval_plot = eval_im(im_path, im_plt_titles, im_shape, models_and_experiments, metrics, metric_fns, eval_plot, plot)

            evals.append(eval_metrics)

            if (i+1)%50 == 0:
                print("Evaluating image pair", i)

        # Print and plot evaluation
        print('Average evaluation for event, ', event)
        for eval_metric in evals[0]:
            print('%s: %.3f'% (eval_metric, float(sum(e[eval_metric] for e in evals)) / len(evals)))
            # For Geometric mean my guess is:
            # np.power(np.prod(e[eval_metrics for e in evals]), 1/len(evals))

        if plot: 
            plt_path = '../figures/'+''.join([m_and_e.replace("/", "_") + '_' for m_and_e in models_and_experiments])+ 'eval_w_wo_' + event + '_start_'+ str(start) +'_n_' + str(n_pics) + '_dpi_' + str(eval_plot.dpi) +'.png'
            eval_plot.display_and_save(plt_path, do_display)
            print('plot created at:', plt_path)
        
        # Return avg statistics for event
        for e in evals:
            total_evals.append(e)

    # Print evaluation averaged over all events
    print('Average evaluation for all events')
    for eval_metric in total_evals[0]:
        print('%s: %.3f'% (eval_metric, float(sum(e[eval_metric] for e in total_evals)) / len(total_evals)))
  
    return evals

if __name__ == "__main__":
    # Test command: 
    """
    # TODO: figure out how to pass lists
    python evaluate.py --n_pics 3 --event 'hurricane-harvey_' --model 'Pix2pixHD' --plot False
    """ 
    #### parser ####
    parser = argparse.ArgumentParser(description='Evaluate Climate Extreme Visualization Model')
    parser.add_argument('--n_pics', 
                default=2,
                type=int,
                help='number of images to evaluate')
    parser.add_argument('--event',
                default='hurricane-harvey_',
                type=str,
                choices=['hurricane-harvey_', 'hurricane-florence_', 'hurricane-matthew_', 'hurricane-michael_'],
                help='event filename name')
    parser.add_argument('--metrics',
                default=['lpips', 'IoU', 'FVPS1'],
                help='metrics to evaluate')
    parser.add_argument('--metric_fns',
                default=[lpips, IoU_metric, FVPS_Score1],
                help='metrics function being called for metrics')
    parser.add_argument('--plot', 
                default=False,
                help='if true plot else dont')
    parser.add_argument('--plt_seg_masks', 
                default=True,
                help='if true plot segmentation masks else dont')
    parser.add_argument('--models_and_experiments', 
                #default=['flood_color/conditional_binary', 'BicycleGAN/baseline', 'Pix2pixHD/conditional_binary'],
                default=['flood_color/conditional_binary_segmented', 'flood_color/conditional_binary_sloshed_segmented','Pix2pixHD/baseline', 'Pix2pixHD/conditional_binary','Pix2pixHD/conditional_binary_sloshed', 'BicycleGAN-HD/trial_0_global_gen', 'BicycleGAN/conditional_binary','BicycleGAN/conditional_binary_sloshed'],#,'BicycleGAN/conditional_binary_sloshed']#'flood_color/conditional_binary_segmented', 'flood_color/conditional_binary_sloshed_segmented', 
                        type=str,
                #choices=['flood_color/conditional_binary', 'BicycleGAN/baseline', 'Pix2pixHD/conditional_binary',
                #        'Pix2pixHD/conditional_binary', 'Pix2pixHD/conditional_binary_sloshed'],
                help='model and experiment to evaluate')
    parser.add_argument('--models_and_experiments_abbr', 
                #default=['flood_color/bin', 'BicycleGAN/base', 'Pix2pixHD/bin'],
                default=['baseline/bin', 'baseline/sloshed','P2PHD/baseline', 'P2PHD/bin','P2PHD/sloshed', 'BGAN/bin_trial','BGAN/bin','BGAN/sloshed'],#'baseline/bin', 'baseline/sloshed',, 
                type=str,
                help='abbreviations of model and experiment names')
    parser.add_argument('--config', 
                default='scripts/flood.sh',
                type=str,
                help='path to config file')
    """
    parser.add_argument('--model', 
                default='Pix2pixHD',
                type=str,
                choices=['Pix2pixHD', 'BicycleGAN'],
                help='model to evaluate')
    parser.add_argument('--experiments',
                default =['baseline', 'conditional_binary', 'conditional_binary_sloshed'],
                help='experiment names')
    """
    args = parser.parse_args()
    
    #eval_main(args.event, args.model, args.n_pics, args.experiments, args.metrics, args.metric_fns, args.plot)
    eval_main(events=[args.event], models_and_experiments=args.models_and_experiments, 
              models_and_experiments_abbr=args.models_and_experiments_abbr, 
              n_pics=args.n_pics, start=0, metrics=args.metrics, metric_fns=args.metric_fns, 
              plot=args.plot, plt_seg_masks=args.plt_seg_masks, plt_mask_w_background=False,
              dpi=300, every=1, do_display=False, new_paths=False, config=args.config)