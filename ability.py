import json
import numpy as np

def normalize_column(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.full_like(arr, 0.5) + np.random.normal(0, 1e-4, size=arr.shape)
    return (arr - min_val) / (max_val - min_val)

def compute_similarity(user_vector, company_matrix, weights=None):
    # 정규화된 company_matrix 생성
    normalized_matrix = np.zeros_like(company_matrix)
    for i in range(company_matrix.shape[1]):
        normalized_matrix[:, i] = normalize_column(company_matrix[:, i])

    # 사용자 입력도 동일하게 정규화
    normalized_user = np.zeros_like(user_vector)
    for i in range(len(user_vector)):
        col = company_matrix[:, i]
        min_val, max_val = np.min(col), np.max(col)
        if max_val - min_val == 0:
            normalized_user[i] = 0.5
        else:
            normalized_user[i] = (user_vector[i] - min_val) / (max_val - min_val)

    # 항목별 거리 계산 후 유사도 변환
    diffs = np.abs(normalized_matrix - normalized_user)  # (N, features)
    sims = 1 / (1 + diffs)

    if weights:
        weights_array = np.array([weights.get(k, 1.0) for k in ['schoolName', 'gpa', 'certificationCount', 'awardsCount', 'internshipCount', 'clubActivityCount', 'englishScore']])
        sims = sims * weights_array
        final_scores = np.sum(sims, axis=1) / np.sum(weights_array)
    else:
        final_scores = np.mean(sims, axis=1)

    return final_scores

def get_company_similarity(user_university: str, user_scores: dict, weights: dict = None):
    # JSON 파일 불러오기
    with open('data/company_ability.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    company_names = [entry['기업명'] for entry in data if entry['기업명'] != '전체평균']

    # Set values for keys using Korean field names
    for entry in data:
        entry["gpa"] = entry.get("학점") or 0.0
        entry["certificationCount"] = entry.get("자격증") or 0.0
        entry["awardsCount"] = entry.get("수상") or 0.0
        entry["internshipCount"] = entry.get("인턴") or 0.0
        entry["clubActivityCount"] = entry.get("동아리") or 0.0
        entry["englishScore"] = entry.get("어학") or 875.0
        entry["schoolName"] = entry.get("대학점수") or 0.0

    university_score_map = {
        "대학원": 1.0,
        "해외대학": 0.9,
        "서울4년": 0.8,
        "수도권4년": 0.6,
        "지방4년": 0.4,
        "초대졸": 0.4,
        "대졸4년": 0.4,
        "고졸": 0.2
    }

    for entry in data:
        university_type = entry.get("schoolName", 0.0)
        entry["schoolName"] = university_score_map.get(university_type, 0.0)

    company_matrix = np.array([
        [entry["schoolName"], entry["gpa"], entry["certificationCount"], entry["awardsCount"], entry["internshipCount"], entry["clubActivityCount"], entry["englishScore"]]
        for entry in data if entry["기업명"] != "전체평균"
    ])

    user_input = [university_score_map.get(user_university, 0.0)] + [
        user_scores.get("gpa", 0.0),
        user_scores.get("certificationCount", 0.0),
        user_scores.get("awardsCount", 0.0),
        user_scores.get("internshipCount", 0.0),
        user_scores.get("clubActivityCount", 0.0),
        user_scores.get("englishScore", 875.0)
    ]

    similarities = compute_similarity(user_input, company_matrix, weights=weights)
    sorted_result = sorted(zip(company_names, similarities), key=lambda x: x[1], reverse=True)
    return sorted_result