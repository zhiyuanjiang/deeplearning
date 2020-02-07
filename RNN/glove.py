import numpy as np
import time

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map

def cosine_similarity(u, v):
    val = np.dot(u, v.T).sum()
    u = np.sqrt(np.power(u, 2).sum())
    v = np.sqrt(np.power(v, 2).sum())
    return 1.*val/u/v

s = time.time()
words, word_to_vec = read_glove_vecs('F:\\deeplearning-data\\word2vec-data\\glove.txt')
e = time.time()
print('load glove : '+str(e-s))

# father = word_to_vec["father"]
# mother = word_to_vec["mother"]
# ball = word_to_vec["ball"]
# crocodile = word_to_vec["crocodile"]
# france = word_to_vec["france"]
# italy = word_to_vec["italy"]
# paris = word_to_vec["paris"]
# rome = word_to_vec["rome"]
#
# print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
# print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
# print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    解决“A与B相比就类似于C与____相比一样”之类的问题

    参数：
        word_a -- 一个字符串类型的词
        word_b -- 一个字符串类型的词
        word_c -- 一个字符串类型的词
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        best_word -- 满足(v_b - v_a) 最接近 (v_best_word - v_c) 的词
    """

    # 把单词转换为小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # 获取对应单词的词向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    # 获取全部的单词
    words = word_to_vec_map.keys()

    # 将max_cosine_sim初始化为一个比较大的负数
    max_cosine_sim = -100
    best_word = None

    # 遍历整个数据集
    for word in words:
        # 要避免匹配到输入的数据
        if word in [word_a, word_b, word_c]:
            continue
        # 计算余弦相似度
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} <====> {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec)))
