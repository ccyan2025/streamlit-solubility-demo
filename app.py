# 1. 导入需要的库（Streamlit核心+数据分析+可视化库）
import streamlit as st  # 导入Streamlit，简写为st（官方推荐）
import pandas as pd     # 导入Pandas，处理数据
import plotly.express as px  # 导入Plotly，画交互式图表

# 2. 设置页面标题和副标题（网页上显示的标题）
st.title("我的第一个Streamlit应用")  # 一级标题
st.subheader("鸢尾花数据集可视化工具")  # 二级标题
st.text("可以通过下拉框选择特征，实时查看散点图")  # 普通文本说明

# 3. 加载数据（从网络读取公开数据集，无需本地下载）
# 这里用的是鸢尾花数据集（经典分类数据集，包含花瓣/花萼特征+品种）
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# 4. 添加交互组件（让用户操作的控件）
# 下拉框：让用户选择X轴特征（选项是数据集中除了最后一列“species”的所有列）
x_feature = st.selectbox(
    label="请选择X轴特征",  # 下拉框的提示文字
    options=df.columns[:-1]  # 可选值（花萼长度、花萼宽度、花瓣长度、花瓣宽度）
)

# 下拉框：选择Y轴特征（和X轴逻辑一致）
y_feature = st.selectbox(
    label="请选择Y轴特征",
    options=df.columns[:-1]
)

# 5. 显示原始数据（可选，让用户看到数据本身）
st.subheader("原始数据预览")
st.dataframe(df.head(10))  # 显示前10行数据，用dataframe()函数生成表格

# 6. 绘制交互式散点图（按花的品种着色，方便区分）
st.subheader("特征散点图")
fig = px.scatter(
    df,  # 数据源
    x=x_feature,  # X轴（用户选择的特征）
    y=y_feature,  # Y轴（用户选择的特征）
    color="species",  # 按品种着色（三种鸢尾花不同颜色）
    title=f"{x_feature} vs {y_feature} 散点图",  # 图表标题
    width=800,  # 图表宽度
    height=500  # 图表高度
)

# 把图表显示在网页上
st.plotly_chart(fig)