# 한글 폰트 설치 및 설정
!apt-get update -qq
!apt-get install -y fonts-nanum

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Nanum 폰트 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)
plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

from googletrans import Translator
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

# Google Translator 초기화
translator = Translator()

# IMDb 데이터셋 로드
vocab_size = 10000
max_length = 100
(x_train, _), (_, _) = imdb.load_data(num_words=vocab_size)

# 단어 인덱스 로드 및 복원
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# IMDb 데이터셋 텍스트 복원 함수
def decode_review(sequence):
    return ' '.join([reverse_word_index.get(i, '?') for i in sequence])

# 샘플 데이터
sample_data = x_train[:10]
padded_data = pad_sequences(sample_data, maxlen=max_length, padding='post')

# 모델 생성 및 임베딩 벡터 생성
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D()
])
embeddings = model.predict(padded_data)

# 추천 시스템 함수
def recommend_reviews(input_text, top_n=3):
    translated_input = translator.translate(input_text, src='ko', dest='en').text
    print(f"입력 리뷰의 영어 번역: {translated_input}")

    input_sequence = [word_index.get(word, 2) for word in translated_input.lower().split()]
    input_padded = pad_sequences([input_sequence], maxlen=max_length, padding='post')

    input_embedding = model.predict(input_padded)
    sim_scores = cosine_similarity(input_embedding, embeddings).flatten()

    similar_indices = sim_scores.argsort()[-top_n:][::-1]
    return similar_indices, sim_scores

# 추천 리뷰 출력 및 시각화 함수
def visualize_recommendations(input_review, similar_indices, sim_scores):
    recommended_reviews = [decode_review(sample_data[idx]) for idx in similar_indices]
    recommended_scores = [sim_scores[idx] for idx in similar_indices]

    print("\n추천된 리뷰:")
    for idx, review in enumerate(recommended_reviews):
        translated_review = translator.translate(review, src='en', dest='ko').text
        print(f"- {translated_review} (유사도: {recommended_scores[idx]:.2f})")

    plt.figure(figsize=(10, 6))
    plt.barh(recommended_reviews[::-1], recommended_scores[::-1], color='skyblue', edgecolor='black')
    plt.xlabel('유사도 점수', fontproperties=fontprop)
    plt.title('추천된 리뷰', fontproperties=fontprop)
    plt.tight_layout()
    plt.show()

# 사용자 입력 테스트
input_review = input("리뷰를 입력하세요 (한국어로 입력): ")
similar_indices, sim_scores = recommend_reviews(input_review)
visualize_recommendations(input_review, similar_indices, sim_scores)
