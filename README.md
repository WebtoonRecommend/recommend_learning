# recommend_learning
prepro_story.py는 데이터 전처리를 위한 파일입니다.



stop_word.txt는 데이터 전처리 과정에서 불필요한 어휘들, 즉 삭제해도 되는 어휘들을 기술한 파일입니다.




## Word2Vec
각 웹툰의 줄거리를 Word2Vec를 사용하여 임베딩합니다. 해당 코드는 prepro_story.py에서 확인할 수 있습니다.



해당 임베딩 과정은 https://wikidocs.net/22644를 참고하였습니다.



## 추천 모델 학습
각 웹툰의 줄거리의 word2vec 값을 평균을 내어 저장하고 해당 값을 바탕으로 모델을 학습합니다. 해당 코드는 train_recommend.py에서 확인할 수 있습니다.



해당 과정은 https://wikidocs.net/102705를 참고하였습니다.