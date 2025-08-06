# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ----------------------------------------
# 1. 모델 로드 (가장 최근 .keras 파일)
# ----------------------------------------
MODEL_DIR = "saved_models"

def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)  # 최신 모델이 위에 오도록 정렬
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
model = load_model(latest_model_path) if latest_model_path else None

# ----------------------------------------
# 2. 데이터 로딩 및 스케일링 학습
# ----------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

scaler = StandardScaler()
scaler.fit(X)  # 스케일러는 학습 데이터 기준으로 fit

# ----------------------------------------
# 3. Streamlit UI
# ----------------------------------------
st.title("유방암 진단 예측기 (Breast Cancer Predictor)")
if model:
    st.markdown(f"불러온 모델: `{os.path.basename(latest_model_path)}`")
else:
    st.error("저장된 모델을 찾을 수 없습니다. 학습을 먼저 진행하세요.")

st.sidebar.header("입력값을 설정하세요")

# 사용자 입력값 (슬라이더로 30개 특성)
user_input = []
for i, feature in enumerate(feature_names):
    val = st.sidebar.slider(
        label=feature,
        min_value=float(X[:, i].min()),
        max_value=float(X[:, i].max()),
        value=float(X[:, i].mean()),
        format="%.2f"
    )
    user_input.append(val)

user_array = np.array(user_input).reshape(1, -1)
scaled_input = scaler.transform(user_array)

# ----------------------------------------
# 4. 예측 수행
# ----------------------------------------
if st.button("예측 실행") and model:
    pred_prob = model.predict(scaled_input)[0][0]
    pred_class = int(pred_prob > 0.5)

    st.subheader("예측 결과")
    st.write(f"예측 확률 (악성일 확률): **{pred_prob * 100:.2f}%**")
    st.write(f"예측 결과: **{'악성(Malignant)' if pred_class == 1 else '양성(Benign)'}**")

    if pred_class == 1:
        st.error("악성(Malignant) 가능성이 높습니다.")
    else:
        st.success("양성(Benign) 가능성이 높습니다.")
