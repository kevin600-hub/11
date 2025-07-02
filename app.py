
import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# 页面配置
try:
    if not os.path.exists("movie.csv"):
        raise FileNotFoundError("movie.csv 不存在！")

    df = pd.read_csv("movie.csv")
    if "genres" not in df.columns or "title" not in df.columns:
        raise ValueError("movie.csv 缺少 'title' 或 'genres' 列")

except Exception as e:
    st.error(f"❌ 程序出错：{e}")
    import traceback
    st.text("🔍 错误详情：")
    st.text(traceback.format_exc())
    st.stop()

# 特征处理：对 genres 做 One-Hot 编码
genre_features = df["genres"].str.get_dummies(sep='|')
df_features = pd.concat([df[["title"]], genre_features], axis=1)

# 计算余弦相似度矩阵
cosine_sim = cosine_similarity(df_features.drop("title", axis=1).fillna(0))
cosine_sim_df = pd.DataFrame(cosine_sim, index=df_features["title"], columns=df_features["title"])

# 页面标题
st.title("🎬 电影类型相似推荐系统")
st.markdown("通过电影的类型（Genres）为你推荐类似风格的影片 🎯")

# 选择框
selected_movie = st.selectbox("请选择你喜欢的一部电影：", df["title"].sort_values().unique())

# 推荐按钮
if st.button("📽 推荐相似电影"):
    st.subheader("推荐结果：")
    if selected_movie in cosine_sim_df.index:
        recommendations = cosine_sim_df[selected_movie].sort_values(ascending=False)
        recommendations = recommendations.drop(index=selected_movie)  # 不推荐自己
        top_movies = recommendations.head(5)

        for i, movie in enumerate(top_movies.index, 1):
            st.markdown(f"**{i}. {movie}**")
    else:
        st.warning("⚠️ 无法找到推荐结果，请检查电影名称是否正确。")
