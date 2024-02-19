import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    from data.aligned_physics_dataset import AlignedPhysicsDataset
    from data.aligned_physics_bin_dataset import AlignedPhysicsBinDataset
    from data.aligned_physics_bin_sloshed_dataset import AlignedPhysicsBinSloshedDataset
    if opt.dataset_mode == "aligned":
        dataset = AlignedDataset()
    elif opt.dataset_mode == "physics_aligned":
        dataset = AlignedPhysicsDataset()
    elif opt.dataset_mode == "physics_aligned_bin":
        dataset = AlignedPhysicsBinDataset()
    elif opt.dataset_mode == "physics_aligned_bin_sloshed":
        dataset = AlignedPhysicsBinSloshedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
