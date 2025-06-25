import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. 모델 로드 (한글도 꽤 잘 됩니다)
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def salary_similarity(preferred_salary, company_salary, max_diff=2000):
    diff = abs(preferred_salary - company_salary)
    return max(0.0, 1 - diff / max_diff)

def text_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

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
