import torch
from datasets import load_dataset  # hugging-face dataset用来加载数据集
from torch.utils.data import Dataset # pytorch定义模型
from torch.utils.data import DataLoader
import torch.nn as nn
from model import BiLSTM
from torch.nn.functional import one_hot
import torch.optim as optim
from tqdm import tqdm

from transformers import BertTokenizer, BertModel

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')

# 定义数据集
class MydataSet(Dataset):
    def __init__(self, path, split):
        self.dataset = load_dataset('csv', data_files=path, split=split)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']
        return text, label

    def __len__(self):
        return len(self.dataset)

# 定义批处理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=500,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1
    labels = torch.LongTensor(labels)  # 该批次的labels

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels

class Saver:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score,
                    'best_score': best_score},
                   name)

if __name__ == '__main__':
    # 定义超参数
    batch_size = 32
    epochs = 5
    dropout = 0.4
    rnn_hidden = 768
    rnn_layer = 1
    class_num = 2
    lr = 0.001
    early_stop = 2

    # 设置GPU环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load train data
    train_dataset = MydataSet('./data/ChnSentiCorp_train.csv', 'train')
    # 装载训练集 drop_last=True 将会丢弃最后一个不完整的批次数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                              drop_last=True)
    dev_dataset = MydataSet('./data/ChnSentiCorp_val.csv', 'train')
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                              drop_last=True)
    # 检查一个批次是否编码成功
    # for i, (input_ids, attention_mask, token_type_ids,
    #         labels) in enumerate(train_loader):
    #     # print(len(train_loader))
    #     print(input_ids[0])  # 第一句话分词后在bert-base-chinese字典中的word_to_id
    #     print(token.decode(input_ids[0]))  # 检查第一句话的id_to_word
    #     print(input_ids.shape)  # 一个批次16句话，每句话被word_to_id成500维
    #     # print(attention_mask.shape)  # 对于使用者而言，不是重要的。含义上面有说明，感兴趣可以做实验测试
    #     # print(token_type_ids.shape)  # 对于使用者而言，不是重要的。含义上面有说明，感兴趣可以做实验测试
    #     print(labels)  # 该批次的labels
    #     break

    model = BiLSTM(drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    # 模型转移至gpu
    model.to(device)
    # 选择损失函数
    criterion = nn.CrossEntropyLoss()
    # 选择优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_acc = 0
    early_stop_count = 0
    save = Saver(model, optimizer, None) # 保存训练好的模型

    # 训练加验证
    for epoch in range(epochs):
        if early_stop_count >= early_stop:
            print("Early stop!")
            break
        model.train()
        # Train
        pbar = tqdm(train_loader)
        for input_ids, attention_mask, token_type_ids, labels in pbar:
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            labels = labels.long().to(device)

            one_hot_labels = one_hot(labels, num_classes=2)  # one_hot输入一定是正整数，如label为字符等，我们需要创建字典，将它们label to id得到对应的ids，再进行one-hot。
            # 将one_hot_labels类型转换成float
            one_hot_labels = one_hot_labels.to(dtype=torch.float)
            optimizer.zero_grad()  # 清空梯度
            output, _ = model.forward(input_ids, attention_mask, token_type_ids)
            # print(out.shape)
            loss = criterion(output, one_hot_labels)  # 计算损失
            loss.backward()  # backward,计算grad
            optimizer.step()  # 更新参数
            pbar.set_description('train loss:{:.4f}'.format(loss))
        pbar.close()

        # Eval
        model.eval()
        correct = 0
        total = 0
        pbar = tqdm(dev_loader)
        for input_ids, attention_mask, token_type_ids, labels in pbar:
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            labels = labels.long().to(device)
            with torch.no_grad():
                output, _ = model.forward(input_ids, attention_mask, token_type_ids)
            output = output.argmax(dim=1) # output.argmax(dim=1) 操作将会返回一个包含每行最大值索引的一维张量
            correct += (output == labels).sum().item()
            total += len(labels)
        pbar.close()
        acc = correct / total
        print('val acc:', acc)

        early_stop_count += 1
        if acc > best_acc:
            best_acc = acc
            save(acc, best_acc, 'checkpoint_best_acc.pt')
            early_stop_count = 0




    # for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
    #     input_ids = input_ids.long().to(device)
    #     attention_mask = attention_mask.long().to(device)
    #     token_type_ids = token_type_ids.long().to(device)
    #     labels = labels.long().to(device)
    #
    #     one_hot_labels = one_hot(labels, num_classes=2) # one_hot输入一定是正整数，如label为字符等，我们需要创建字典，将它们label to id得到对应的ids，再进行one-hot。
    #     # 将one_hot_labels类型转换成float
    #     one_hot_labels = one_hot_labels.to(dtype=torch.float)
    #     optimizer.zero_grad()  # 清空梯度
    #     output = model.forward(input_ids, attention_mask, token_type_ids)
    #     # print(out.shape)
    #     loss = criterion(output, one_hot_labels)  # 计算损失
    #     print(loss)
    #     loss.backward()  # backward,计算grad
    #     optimizer.step()  # 更新参数


