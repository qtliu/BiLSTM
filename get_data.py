import torch
from datasets import load_dataset
from datasets import load_from_disk

# 定义数据集
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, split):
#         self.dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, item):
#         text = self.dataset[item]['text']
#         label = self.dataset[item]['label']
#         return text, label

if __name__ == '__main__':
    # dataset = Dataset('train')
    # print(len(dataset))
    # print(dataset[0])

    dataset = load_from_disk('./data/ChnSentiCorp')
    print(dataset)
    dataset['train'].to_csv(path_or_buf='./data/ChnSentiCorp_train.csv')
    dataset['validation'].to_csv(path_or_buf='./data/ChnSentiCorp_val.csv')
    dataset['test'].to_csv(path_or_buf='./data/ChnSentiCorp_test.csv')

