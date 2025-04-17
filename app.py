import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# アプリ終了フラグの管理 
if "app_ended" not in st.session_state:
    st.session_state.app_ended = False

# 入力欄（文字列入力として扱う） 
st.title("Temperature Correlation: 9AM vs 9PM")

# 文字入力欄
input_text = st.text_input("Enter 9AM temperature (℃) or type 'end' to quit:", value="")
prediction_output = st.empty()  #出力用の表示エリアを確保

# 「end」と入力されたら終了フラグを立てる
if input_text.strip().lower() == "end":
    st.session_state.app_ended = True

# 終了処理画面
if st.session_state.app_ended:
    st.empty()  # 画面をクリア（厳密には全部再描画しないだけ）
    st.markdown(
        "<h2 style='text-align:center; color:red;'>Application Ended</h2>",
        unsafe_allow_html=True
    )
    st.stop()



# CSV読み込み
df = pd.read_csv("data2021kuma.csv", encoding="utf-8")
df["年月日時"] = pd.to_datetime(df["年月日時"], format="%Y/%m/%d %H:%M:%S")
df["hour"] = df["年月日時"].dt.hour
df["date"] = df["年月日時"].dt.date

# データ抽出
df_morning = df[df["hour"] == 9].copy()
df_night = df[df["hour"] == 21].copy()

# 日付選択UI
date_min = df["date"].min()
date_max = df["date"].max()

start_date = st.date_input("Start date", value=date_min, min_value=date_min, max_value=date_max)
end_date = st.date_input("End date", value=min(date_min + pd.Timedelta(days=14), date_max), min_value=date_min, max_value=date_max)

if (end_date - start_date).days < 13:
    st.warning("Please select a range of at least 14 days.")
else:
    df_morning_filtered = df_morning[(df_morning["date"] >= start_date) & (df_morning["date"] <= end_date)]
    df_night_filtered = df_night[(df_night["date"] >= start_date) & (df_night["date"] <= end_date)]

    merged = pd.merge(
        df_morning_filtered[["date", "気温(℃)"]].rename(columns={"気温(℃)": "temp_9am"}),
        df_night_filtered[["date", "気温(℃)"]].rename(columns={"気温(℃)": "temp_9pm"}),
        on="date"
    )

    X = merged["temp_9am"].values.reshape(-1, 1)
    y = merged["temp_9pm"].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # グラフ
    st.subheader("Scatter Plot with Linear Regression")
    fig, ax = plt.subplots()
    ax.scatter(merged["temp_9am"], merged["temp_9pm"], label="Data points")
    ax.plot(merged["temp_9am"], y_pred, color="red", label="Regression line")
    ax.set_xlabel("Temperature at 9AM (℃)")
    ax.set_ylabel("Temperature at 9PM (℃)")
    ax.set_title("9AM vs 9PM Temperature Correlation")
    ax.legend()
    st.pyplot(fig)

    # 相関係数
    corr = np.corrcoef(merged["temp_9am"], merged["temp_9pm"])[0, 1]
    st.write(f"Pearson correlation coefficient: {corr:.2f}")

    # 入力処理（予測値を出す）
    try:
        input_temp = float(input_text)
        predicted_temp = model.predict([[input_temp]])[0]
        prediction_output.success(f"Predicted 9PM temperature: {predicted_temp:.2f} ℃")  # ← ここに出力
    except ValueError:
        if input_text:
            prediction_output.error("Please enter a valid number or 'end' to quit.")