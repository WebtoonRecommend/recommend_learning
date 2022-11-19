from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

model = KeyedVectors.load_word2vec_format("descrip_w2v") # 저장한 Word2Vec 모델 가져오기
Description = []

with open('Descrip.pkl', 'rb') as f: # pickle 파일에서 전처리한 Description 불러오기
    while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
        Description.append(data)

Title = []
with open('Title.pkl', 'rb') as f: # pickle 파일에서 전처리한 Title 불러오기
    while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
        Title.append(data)


Title = pd.DataFrame(Title, columns=['Title']) # 리스트를 데이터프레임으로 변환

# 단어 벡터 평균 구하기

# 코드에 대한 설명은 README.md의 링크 참조
def get_doc2vec(document):
    document_embed = []

    for line in document:
        doc2vec = None
        count = 0
        for word in line:
            if word in list(model.index_to_key):
                count += 1
                if doc2vec is None:
                    doc2vec = model[word]
                else:
                    doc2vec = doc2vec + model[word]

        if doc2vec is not None:
            doc2vec = doc2vec / count
            document_embed.append(doc2vec)
    return document_embed

document = get_doc2vec(Description) # 각 웹툰의 줄거리를 각 단어의 임베딩 값들의 평균으로 나타냄

with open('document.pkl', 'wb') as f: # 전처리된 Title을 pkl 파일로 저장
    for a in document:
        pickle.dump(a , f)

# 웹툰 추천 시스템

# 웹툰 줄거리의 유사도 계산
story_cosine = cosine_similarity(document) #코사인 유사도를 2차원의 형태로 생성

with open('Story_Cosine.pkl', 'wb') as f: # 전처리된 Title을 pkl 파일로 저장
    for a in story_cosine:
        pickle.dump(a , f)