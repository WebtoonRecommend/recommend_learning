import sqlite3
from konlpy.tag import Okt
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

# 사용할 db 연결

con = sqlite3.connect('test.db') 

cur = con.cursor()

# 데이터 가져오기

desp = cur.execute('select 웹툰ID, 이름, 설명, 장르 from webtoon_info') # 데이터를 쿼리하여 가져오기

# 데이터 파싱

WebToonId = []
Title = []
Description = []
Genre = []

for i in desp: # 데이터를 각 변수별로 분리하기
    WebToonId.append(i[0])
    Title.append(i[1])
    Description.append(i[2])
    Genre.append(i[3])

# 정규표현식으로 한글이 아닌 것 제거
Description = pd.DataFrame(Description)

Description[0] = Description[0].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

Description = Description[0].to_list()

# 웹툰 줄거리 토큰화(어근 단위로 문장을 분할)
okt = Okt() 
for j in range(len(Description)):
    Description[j] = okt.morphs(Description[j]) # 어근 단위로 문장을 분할

# 웹툰 줄거리 불용어 제거(불필요한 단어)

f = open('stop_word.txt', 'r') # 불용어 목록 파일 읽어오기

stop_words_file = f.readlines() # 불용어 목록 리스트로 저장
for i in range(len(stop_words_file)):
    stop_words_file[i] = stop_words_file[i].strip()
    

# 불용어 삭제 함수 정의
def del_stop(a, stop_words): 
    for j in stop_words:
        if j in a:
            a.remove(j)
    return a

# 불용어 삭제
for i in range(len(Description)):
    Description[i] = del_stop(Description[i], stop_words_file)

# 단어 토큰화

# num = 1000 # 단어의 최소 사용빈도수

# tokenizer = Tokenizer(num_words=num) # 사용빈도수 설정

# tokenizer.fit_on_texts(Description) # 단어를 토큰화하는 함수

# print(tokenizer.texts_to_sequences(Description)) # 문장을 토큰화하고 정수 인코딩하는 함수


# 단어 임베딩

from gensim.models import Word2Vec

model = Word2Vec(sentences=Description, vector_size=100, window=5, min_count=5, workers=4, sg=0) # 단어 임베딩

model.wv.save_word2vec_format('descrip_w2v')