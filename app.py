
import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

# 页面配置
st.set_page_config(page_title="🎬 电影推荐系统", layout="centered")

# 检查文件是否存在
if not os.path.exists("movie.csv"):
    st.error("❌ 找不到 movie.csv 文件，请确认它上传到了仓库根目录！")
    st.stop()

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv("movie.csv")
    if "genres" not in df.columns or "title" not in df.columns:
        raise ValueError("❌ movie.csv 缺少必要的 'title' 或 'genres' 列")
    return df

df = load_data()
if df is None:
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
