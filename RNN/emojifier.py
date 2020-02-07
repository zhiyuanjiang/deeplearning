import csv
import numpy as np
import emoji
import matplotlib.pyplot as plt

def read_csv(filename = 'F:\\deeplearning-data\\word2vec-data\\train_emoji.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y


emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

def convert_to_one_hot(data):
    m = len(data)
    data_one_hot = np.zeros((5, m))
    for i in range(m):
        data_one_hot[data[i], i] = 1
    return data_one_hot


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


def sentence_to_avg(sentence, word_to_vec):
    """
    将句子转换为单词列表，提取其GloVe向量，然后将其平均。

    参数：
        sentence -- 字符串类型，从X中获取的样本。
        word_to_vec -- 字典类型，单词映射到50维的向量的字典

    返回：
        avg -- 对句子的均值编码，维度为(50,)
    """

    # 第一步：分割句子，转换为列表。
    words = sentence.lower().split()

    # 初始化均值词向量
    avg = np.zeros((50, 1))

    # 第二步：对词向量取平均。
    for w in words:
        avg += word_to_vec[w].reshape(50, 1)
    avg = np.divide(avg, len(words))
    return avg

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))


def model(X, Y, word_to_vec, learning_rate=0.01, num_iterations=400):
    """
    在numpy中训练词向量模型。

    参数：
        X -- 输入的字符串类型的数据，维度为(m, 1)。
        Y -- 对应的标签，0-7的数组，维度为(m, 1)。
        word_to_vec_map -- 字典类型的单词到50维词向量的映射。
        learning_rate -- 学习率.
        num_iterations -- 迭代次数。

    返回：
        pred -- 预测的向量，维度为(m, 1)。
        W -- 权重参数，维度为(n_y, n_h)。
        b -- 偏置参数，维度为(n_y,)
    """
    np.random.seed(1)

    # 定义训练数量
    m = Y.shape[0]
    n_y = 5
    n_h = 50

    # 使用Xavier初始化参数
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y, 1))

    # 将Y转换成独热编码
    Y_oh = convert_to_one_hot(Y)

    # 优化循环
    loss_y = []
    for t in range(num_iterations):
        correct = 0
        for i in range(m):
            # 获取第i个训练样本的均值
            avg = sentence_to_avg(X[i], word_to_vec)

            # 前向传播
            z = np.dot(W, avg) + b
            a = softmax(z)
            assert(a.shape == (5, 1))
            # 计算第i个训练的损失
            cost = -np.sum(Y_oh[:,i].reshape(5, 1) * np.log(a))

            pred = np.argmax(a, axis=0)
            if pred == Y[i]:
                correct += 1

            # 计算梯度
            dz = a - Y_oh[:,i].reshape(5, 1)
            # print(Y_oh[:,i].shape)
            # print(dz.shape)
            assert(dz.shape == (5, 1))
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # 更新参数
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=t, cost=cost))
            print("第{t}轮，准确度为{precise}".format(t=t, precise=correct*1./m))
            loss_y.append(cost)


    x = [i for i in range(len(loss_y))]
    plt.plot(x, loss_y, 'red')
    plt.show()

    return W, b

def test(X, Y, W, b, word_to_vec):

    m = Y.shape[0]

    correct = 0
    for i in range(m):
        avg = sentence_to_avg(X[i], word_to_vec)
        z = np.dot(W, avg) + b
        a = softmax(z)

        pred = np.argmax(a, axis=0)
        if pred == Y[i]:
            correct += 1

    print('test data accuracy is {v}'.format(v=correct*1./m))

def predict(x, W, b, word_to_vec):
    m = x.shape[0]
    pred = []
    for i in range(m):
        avg = sentence_to_avg(x[i], word_to_vec)
        z = np.dot(W, avg)+b
        a = softmax(z)
        val = np.argmax(a, axis=0)
        pred.append(val[0])
    return pred

words, word_to_vec = read_glove_vecs('F:\\deeplearning-data\\word2vec-data\\glove.txt')
train_x, train_y = read_csv()
test_x, test_y = read_csv('F:\\deeplearning-data\\word2vec-data\\test.csv')
W, b = model(train_x, train_y, word_to_vec, 0.01, 2000)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
pred = predict(X_my_sentences, W, b, word_to_vec)

for i in range(len(pred)):
    print(X_my_sentences[i]+" "+label_to_emoji(pred[i]))