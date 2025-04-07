# 继承BaseDataset，重写_getitem方法，加载数据
from data.BaseDataset import BaseDataset

class ContrastiveDataset(BaseDataset):
    def __init__(self, csv_file, args, transform=None, downsample_pd=None):
        super().__init__(csv_file, args, transform=transform, downsample_pd=downsample_pd)

    def __getitem__(self, idx):
        # 加载数据
        # subject_id = self.filtered_data.iloc[idx]['PATNO']
        # event_id = self.filtered_data.iloc[idx]['EVENT_ID']
        label = int(self.filtered_data.iloc[idx]['APPRDX'])

        # 加载数据
        dti = self._load_dti(idx, 'FA')
        mri = self._load_mri(idx)

        data = (dti, mri)
        return data, label

