import numpy as np
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import fasttext
import fasttext.util
import torch
import math

 # 1. 모델 로드 (한글도 꽤 잘 됩니다)
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# KeyBERT, FastText 모델 로드
kw_model = KeyBERT(model)
fasttext.util.download_model('ko', if_exists='ignore')
ft_model = fasttext.load_model('cc.ko.300.bin')

def salary_similarity(preferred_salary, company_salary, max_diff=2000):
    diff = abs(preferred_salary - company_salary)
    return max(0.0, 1 - diff / max_diff)

def get_keyword_embedding(text, top_n=3):
    keywords = [kw[0] for kw in kw_model.extract_keywords(text, top_n=top_n)]
    vectors = [ft_model.get_word_vector(word) for word in keywords if word in ft_model.words]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

def arcface_similarity(vec1, vec2, margin=0.3):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    cos_sim = np.dot(vec1, vec2)
    theta = math.acos(np.clip(cos_sim, -1.0, 1.0))
    return math.cos(theta + margin)

def text_similarity(text1, text2):
    # 문맥 임베딩
    context_emb1 = model.encode(text1, convert_to_numpy=True, normalize_embeddings=True)
    context_emb2 = model.encode(text2, convert_to_numpy=True, normalize_embeddings=True)

    # 키워드 임베딩
    keyword_emb1 = get_keyword_embedding(text1)
    keyword_emb2 = get_keyword_embedding(text2)

    # 차원 맞추기 (384 → 300으로 축소, 여기선 간단히 300 기준으로 자름)
    context_emb1 = context_emb1[:300]
    context_emb2 = context_emb2[:300]

    # 가중 평균
    combined1 = 0.6 * context_emb1 + 0.4 * keyword_emb1
    combined2 = 0.6 * context_emb2 + 0.4 * keyword_emb2

    # ArcFace 보정 유사도 계산
    return arcface_similarity(combined1, combined2)

def normalize_cosine(score):
    return (score + 1) / 2  # cosine -1~1 → 0~1

def vector_similarity(user_vec, company_vec):
    dist = np.linalg.norm(np.array(user_vec) - np.array(company_vec))
    max_dist = np.sqrt(len(user_vec))
    return max(0.0, 1 - dist / max_dist)

def categorical_similarity(user_value, company_value, similar_pairs=None):
    if user_value == company_value:
        return 1.0
    elif similar_pairs and (user_value, company_value) in similar_pairs:
        return 0.5
    else:
        return 0.0

def calculate_preference_similarity(user_inputs, company_info, weights):
    salary_score = salary_similarity(user_inputs["salary"], company_info["salary"])
    text_sim = text_similarity(company_info["vision"], user_inputs["introduction"])
    normalized_text_sim = normalize_cosine(text_sim)
    vector_sim = vector_similarity(user_inputs["traits"], company_info["traits"])
    scale_score = categorical_similarity(user_inputs["scale"], company_info["scale"])

    final_score = (
        weights["salary"] * salary_score +
        weights["vision"] * normalized_text_sim +
        weights["traits"] * vector_sim +
        weights["scale"] * scale_score
    ) / (weights["salary"] + weights["vision"] + weights["traits"] + weights["scale"])
    return round(final_score, 4)

def run_preference_match_api(user_input: dict, company_input: dict) -> float:
    """
    FastAPI에서 받는 입력값을 기반으로 취향 유사도를 계산해주는 함수
    :param user_input: 사용자의 입력 정보 (salary, self_intro, traits(float), scale)
    :param company_input: 회사 정보 (salary, vision, job, traits[list], scale)
    :return: 유사도 점수 (0.0~1.0)
    """
    scale_map = {
        "대기업": 1.0,
        "중견기업": 0.7,
        "공기업": 0.7,
        "중소기업": 0.5,
        "스타트업": 0.3
    }
    processed_user = {
        "salary": user_input["salary"],
        "introduction": user_input["introduction"],
        "traits": [user_input["traits"], 1 - user_input["traits"]],
        "scale": scale_map.get(user_input["scale"], 0.5)  # default 중소기업
    }

    processed_company = {
        "salary": company_input["salary"],
        "vision": company_input["vision"],
        "traits": company_input["traits"],
        "scale": company_input["scale"]
    }

    return calculate_preference_similarity(processed_user, processed_company)
