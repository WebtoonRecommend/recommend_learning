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


# 웹툰 추천 시스템

# 웹툰 줄거리의 유사도 계산
story_cosine = cosine_similarity(document) #코사인 유사도를 2차원의 형태로 생성

def Recommendations10(title): # https://wikidocs.net/102705 참고
    WebToon = Title[['Title']]
    indices = list(WebToon.index)
    idx = WebToon.index[WebToon['Title']==title].to_list()[0]
    idx = indices[idx]
    
    sim_scores = list(enumerate(story_cosine[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    WebToon_indices = [i[0] for i in sim_scores]

    recommend = WebToon.iloc[WebToon_indices].reset_index(drop=True)

    return recommend


def FirstRecommendations10(words): # https://wikidocs.net/102705 참고, 처음 입력 받은 단어를 기반으로 추천, 처음 받은 단어의 word2vec값을 평균내어 입력
    
    WebToon = Title[['Title']] # 웹툰 제목 목록
    doc2vec = 0 # word2vec의 평균을 저장할 변수
    
    for i in words: # 추천받은 단어의 word2vec들을 모두 합하여 평균을 냄
        doc2vec += model[i]
    
    doc2vec = doc2vec / len(words) 

    sim_scores = list(enumerate(doc2vec))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    WebToon_indices = [i[0] for i in sim_scores]

    recommend = WebToon.iloc[WebToon_indices].reset_index(drop=True)

    return recommend


print(Recommendations10('대학일기'))
