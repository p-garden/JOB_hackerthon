import pandas as pd
import ast

# 어학 점수 파싱 함수
def parse_language_scores(lang_str):
    try:
        lang_dict = ast.literal_eval(lang_str)
        toeic = lang_dict.get("토익", None)

        return pd.Series({
            "토익": float(toeic) if toeic else None,
        })
    except:
        return pd.Series({"토익": None})

# 1. 데이터 로드
df = pd.read_csv("data/resume_DB.csv")

# 2. 어학 컬럼 파싱 및 병합
lang_scores_df = df["어학"].apply(parse_language_scores)
df = pd.concat([df, lang_scores_df], axis=1)

# Remove 오픽 and 토스 columns if they exist
if "오픽" in df.columns:
    df = df.drop(columns=["오픽"])
if "토스" in df.columns:
    df = df.drop(columns=["토스"])

# 3. 기업별 평균 계산
grouped = df.groupby("기업명")[["학점", "자격증", "수상", "인턴", "동아리"]].mean()

# 3-1. 기업별 어학 평균 계산 (토익만)
lang_grouped = df.groupby("기업명")[["토익"]].mean()

# 3-2. 대학 점수 매핑
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
df["대학점수"] = df["대학"].map(university_score_map)
univ_grouped = df.groupby("기업명")[["대학점수"]].mean()

def format_lang_avg(toeic):
    toeic_score = round(toeic) if not pd.isna(toeic) else 'None'
    return f"{{'토익': {toeic_score}}}"

# 4. 전체 평균 계산 (학점 0 제외)
overall_avg = {
    "학점": df[df["학점"] > 0]["학점"].mean(),
    "자격증": df["자격증"].mean(),
    "수상": df["수상"].mean(),
    "인턴": df["인턴"].mean(),
    "동아리": df["동아리"].mean()
}

# 5. 어학 평균 묶어서 문자열로 (토익만)
toeic_avg = df["토익"].mean()

lang_avg_str = format_lang_avg(toeic_avg)

# 5-1. 대학 점수 평균 포함
grouped["대학점수"] = univ_grouped["대학점수"].round(2)

# 6. 전체 평균 추가
grouped["어학"] = lang_grouped["토익"].round(0)
grouped.loc["어학"] = {
    "학점": overall_avg["학점"],
    "자격증": overall_avg["자격증"],
    "수상": overall_avg["수상"],
    "인턴": overall_avg["인턴"],
    "동아리": overall_avg["동아리"],
    "어학": round(toeic_avg),
    "대학점수": df["대학점수"].mean()
}

# 7. 정리 및 저장
grouped = grouped.round(2)
grouped.reset_index().to_json("data/company_ability.json", orient="records", force_ascii=False, indent=2)