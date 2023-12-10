import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# 定义模型，上游使用bert预训练，下游任务选择双向LSTM模型，最后加一个全连接层
class BiLSTM(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型
        self.embedding = BertModel.from_pretrained('bert-base-chinese')

        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)

            # 生成下游RNN层以及全连接层
            self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=1, batch_first=True,
                                bidirectional=True, dropout=self.drop)
            self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
            # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。但是使用其他损失函数时还是需要加入softmax层的。

    def forward(self, input_ids, attention_mask, token_type_ids):
        # bert预训练
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # print(embedding)
        # print("===============")
        # print(embedding[0])
        # print(embedding[0].shape)
        embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state 和 embedding[0] 效果是一样的。
        # print("--------------------------------")
        # print(embedding)
        # print(embedding.shape)
        out, (h_n, c_n) = self.lstm(embedded)
        # print(out.shape)  # [128, 200 ,1536]  因为是双向的，所以out是两个hidden_dim拼接而成。768*2 = 1536
        # h_n[-2, :, :] 为正向lstm最后一个隐藏状态。
        # h_n[-1, :, :] 为反向lstm最后一个隐藏状态
        # print(out[:, -1, :768] == h_n[-2, :, :])  # 正向lstm最后时刻的输出在output最后一层
        # print(out[:, 0, 768:] == h_n[-1, :, :])  # 反向lstm最后时刻的输出在output第一层
        output_tmp = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # print(output.shape)
        output = self.fc(output_tmp)
        return output, output_tmp

