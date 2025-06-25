from fastapi import APIRouter
from pydantic import BaseModel
from preference import calculate_preference_similarity
from ability import get_company_similarity
import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

api_router = APIRouter()

# 모델
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# 데이터 로드
with open("data/company_preference.json", "r", encoding="utf-8") as f:
    companies = json.load(f)

company_df = pd.read_json("data/company_ability.json")

# 입력 모델
class PreferenceInput(BaseModel):
    salary: int
    introduction: str
    traits: float
    scale: str

class AbilityInput(BaseModel):
    gpa: float
    certificationCount: float
    awardsCount: float
    internshipCount: float
    clubActivityCount: float
    englishScores: float
    schoolName: str
 

# 취향 기반 매칭
@api_router.post("/preference")
def preference_match(input: PreferenceInput):
    user_inputs = {
        "salary": input.salary,
        "introduction": input.introduction,
        "traits": [input.traits, 1 - input.traits],
        "scale": input.scale
    }

    weights = {
        "salary": 2.0,
        "traits": 1.0,
        "scale": 2.0,
        "vision": 1.0
    }

    results = []
    for company in companies:
        company_info = {
            "salary": company["salary"],
            "vision": company["vision"],
            "traits": [company["traits"], 1 - company["traits"]],
            "scale": company["scale"]
        }
        score = calculate_preference_similarity(user_inputs, company_info, weights)
        results.append({
            "company": company["기업명"],
            "score": score
        })

    return {"top_matches": sorted(results, key=lambda x: x["score"], reverse=True)}

# 능력 기반 매칭
@api_router.post("/ability_match")
def ability_match(input: AbilityInput):
    user_scores = {
        "gpa": input.gpa,
        "certificationCount": input.certificationCount,
        "awardsCount": input.awardsCount,
        "internshipCount": input.internshipCount,
        "clubActivityCount": input.clubActivityCount,
        "englishScores": input.englishScores
    }

    weights = {
        "schoolName": 2.0,
        "gpa": 1.0,
        "certificationCount": 1.0,
        "awardsCount": 1.0,
        "internshipCount": 1.5,
        "clubActivityCount": 1.0,
        "englishScores": 1.5
    }

    result = get_company_similarity(user_university=input.schoolName, user_scores=user_scores, weights=weights)
    return {
        "top_matches": [
            {"company": name, "similarity": round(score, 4)}
            for name, score in result
        ]
    }   