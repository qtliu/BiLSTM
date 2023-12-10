import torch
from datasets import load_dataset  # hugging-face dataset用来加载数据集
from torch.utils.data import Dataset # pytorch定义模型
from torch.utils.data import DataLoader
import torch.nn as nn
from model import BiLSTM
from torch.nn.functional import one_hot
import torch.optim as optim
from tqdm import tqdm
from train import MydataSet, collate_fn, Saver

from transformers import BertTokenizer, BertModel

if __name__ == '__main__':
    # 加载检查点，定义超参数
    checkpoint = torch.load('checkpoint_best_acc.pt', map_location='cpu')
    batch_size = 16
    dropout = 0.4
    rnn_hidden = 768
    rnn_layer = 1
    class_num = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载字典和分词工具
    token = BertTokenizer.from_pretrained('bert-base-chinese')

    # load test data
    test_dataset = MydataSet('./data/ChnSentiCorp_test.csv', 'train')
    # 装载训练集 drop_last=True 将会丢弃最后一个不完整的批次数据
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                              drop_last=True)

    model = BiLSTM(drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    # 加载检查点参数
    model.load_state_dict(checkpoint['param'])
    # 模型转移至gpu
    model.to(device)

    # Eval
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(test_loader)

    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in pbar:
            input_ids = input_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            labels = labels.long().to(device)
            output, sent_emb = model.forward(input_ids, attention_mask, token_type_ids)
            output = output.argmax(dim=1)  # output.argmax(dim=1) 操作将会返回一个包含每行最大值索引的一维张量
            correct += (output == labels).sum().item()
            total += len(labels)

            # print(token.batch_decode(input_ids))
            # print('\n')
            # print('batch sentences:')
            # for ids in input_ids:
            #     print(token.decode(ids))
            # print('batch embeddings:')
            # for emb in sent_emb:
            #     print(emb.cpu().data.numpy().tolist())
            # break

        pbar.close()
        acc = correct / total
        print('test acc:', acc)



