from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

imdb = load_dataset("stanfordnlp/imdb")
train_texts, train_labels = imdb["train"]["text"], imdb["train"]["label"]
test_texts, test_labels = imdb["test"]["text"], imdb["test"]["label"]

import re
from nltk.stem import SnowballStemmer
def preprocess_text(text):
    # 清除HTML标签和特殊字符
    clean_text = re.sub('<.*?>', '', text)
    clean_text = re.sub('[^a-zA-Z]', ' ', clean_text)
    
    # 将文本转换为小写
    clean_text = clean_text.lower()
    
    # 切割成单词
    words = clean_text.split()
    
    # 去除停用词
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    # 对文本进行词干化处理
    stemmer = SnowballStemmer('english')
    words = [stemmer.stem(word) for word in words]
    
    # 将单词重新组合成文本
    clean_text = ' '.join(words)
    
    return clean_text

train_texts = [preprocess_text(text) for text in train_texts]
test_texts = [preprocess_text(text) for text in test_texts]
# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer()
# 在训练集上进行特征提取
train_features = vectorizer.fit_transform(train_texts)
# 在测试集上进行特征提取
test_features = vectorizer.transform(test_texts)

# 创建朴素贝叶斯分类器
model = LogisticRegression(max_iter=1000, random_state=42)
# 在训练集上训练模型
model.fit(train_features, train_labels)

# 在测试集上进行预测
pred_labels = model.predict(test_features)
# 计算准确率和其他评估指标
accuracy = accuracy_score(test_labels, pred_labels)
report = classification_report(test_labels, pred_labels)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
