from gensim.models import KeyedVectors
import numpy as np
import torch

glove_input_file = './glove.840B.300d.txt'
model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)

def sentence_to_vector(words, model):
    print(words)
    vectors = [model[word] for word in words if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


clip_feat = torch.zeros(0, model.vector_size).cuda()

# 打开源文件 miniImagenet_label.list
with open('./list/miniImagenet_label.list', 'r', encoding='utf-8') as file_a:
    # 逐行读取文件内容并转换为向量
    for line in file_a:
        line = line.replace('_', ' ').strip()  # 替换下划线为一个空格，并去除行尾的空白字符（如果有的话）
        line = line.split(' ')  # 按照空格分割字符串  # 去除行尾的换行符
        line = [word.lower() for word in line]
        print(line)
        sentence_vector = sentence_to_vector(line, model)
        sentence_tensor = torch.tensor(sentence_vector).view(1, -1).cuda()  # 将向量转换为 CUDA 张量
        clip_feat = torch.cat((clip_feat, sentence_tensor), dim=0)

# 将张量保存为 NumPy 数组
np.save('glove', clip_feat.cpu().numpy())
# 提取句子的特征向量

