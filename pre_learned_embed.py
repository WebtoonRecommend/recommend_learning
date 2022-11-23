from gensim.models import FastText
import pickle

Description = []

with open('Descrip.pkl', 'rb') as f: # pickle 파일에서 전처리한 Description 불러오기
    while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
        Description.append(data)


fast_text_model = FastText(vector_size=200, window=5, min_count=2, workers=-1)
fast_text_model.build_vocab(Description)
fast_text_model.load_fasttext_format('ko/ko.vec')
fast_text_model.train(Description, total_examples= fast_text_model.corpus_count, epochs=15)

