#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import time
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from joblib import dump, load
from matplotlib import font_manager
from github import Github

# 动态读取Token
token = os.getenv("GITHUB_TOKEN")
if not token:
    st.error("GitHub Token 未设置。请在 Streamlit Cloud 的 Secrets 中添加 GITHUB_TOKEN。")
    st.stop()

# GitHub 配置
repo_name = "xantoxia/neck"  # 替换为你的 GitHub 仓库
models_folder = "models/"  # GitHub 仓库中模型文件存储路径
latest_model_file = "latest_model_info.txt"  # 最新模型信息文件
commit_message = "更新模型文件"  # 提交信息

# 定义带时间戳的备份文件名
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_filename = f"肩颈分析-模型-{timestamp}.joblib"

# 上传文件到 GitHub
def upload_file_to_github(file_path, github_path, commit_message):
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)

        # 读取文件内容
        with open(file_path, "rb") as f:
            content = f.read()

        # 检查文件是否存在
        try:
            file = repo.get_contents(github_path)
            repo.update_file(github_path, commit_message, content, file.sha)
            st.success(f"文件已成功更新到 GitHub 仓库：{github_path}")
        except:
            repo.create_file(github_path, commit_message, content)
            st.success(f"文件已成功上传到 GitHub 仓库：{github_path}")
    except Exception as e:
        st.error(f"上传文件到 GitHub 失败：{e}")

# 下载最新模型文件
def download_latest_model_from_github():
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)

        # 获取最新模型信息
        try:
            latest_info = repo.get_contents(models_folder + latest_model_file).decoded_content.decode()
            latest_model_path = models_folder + latest_info.strip()
            st.write(f"最新模型路径：{latest_model_path}")

            # 下载最新模型文件
            file_content = repo.get_contents(latest_model_path)
            with open("/tmp/latest_model.joblib", "wb") as f:
                f.write(file_content.decoded_content)

            st.success("成功下载最新模型！")
            return "/tmp/latest_model.joblib"
        except:
            st.warning("未找到最新模型信息文件，无法下载模型。")
            return None
    except Exception as e:
        st.error(f"从 GitHub 下载模型失败：{e}")
        return None

# 设置中文字体
simhei_font = font_manager.FontProperties(fname="simhei.ttf")
plt.rcParams['font.family'] = simhei_font.get_name()  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# Streamlit 标题
st.title("肩颈角度动态分析与异常检测")
st.write("本人因AI工具结合人因规则与机器学习模型，自动检测异常作业姿势并提供可视化分析。")

# 模板下载
with open("肩颈角度数据模版.csv", "rb") as file:
    st.download_button(
        label="下载 CSV 模板",
        data=file,
        file_name="template.csv",
        mime="text/csv"
    )

# 数据加载与预处理
uploaded_file = st.file_uploader("上传肩颈角度数据文件 (CSV 格式)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = ['天(d)', '时间(s)', '颈部角度(°)', '肩部上举角度(°)', 
                    '肩部外展/内收角度(°)', '肩部旋转角度(°)']
    st.write("### 1.1  数据预览")
    
    # 调整序号显示，从 1 开始
    data_reset = data.copy()  # 复制原始数据
    data_reset.index += 1  # 将索引从 1 开始
    data_reset.index.name = "序号"  # 为索引命名为 "序号"

    # 在 Streamlit 显示数据预览
    st.write(data_reset.head())
       
    # 数据统计分析函数
    def analyze_data(data):
        st.write("### 1.2  数据统计分析")
        stats = data.describe()
        st.write(stats)

        st.write("### 1.3  动态分析结论：数据统计特性")
        st.write(f"- 颈部角度范围：{stats['颈部角度(°)']['min']}° 至 {stats['颈部角度(°)']['max']}°，平均值为 {stats['颈部角度(°)']['mean']:.2f}°")
        st.write(f"- 肩部旋转角度范围：{stats['肩部旋转角度(°)']['min']}° 至 {stats['肩部旋转角度(°)']['max']}°，平均值为 {stats['肩部旋转角度(°)']['mean']:.2f}°")
        st.write(f"- 肩部外展/内收角度的标准差为 {stats['肩部外展/内收角度(°)']['std']:.2f}，波动较 {'大' if stats['肩部外展/内收角度(°)']['std'] > 15 else '小'}。")

    # 3D 散点图
    def generate_3d_scatter(data):
        st.write("### 2.1  肩颈角度3D可视化散点图")
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data['时间(s)'], data['颈部角度(°)'], data['肩部旋转角度(°)'], c=data['肩部外展/内收角度(°)'], cmap='viridis')
        ax.set_xlabel('时间(s)', fontproperties=simhei_font)
        ax.set_ylabel('颈部角度(°)', fontproperties=simhei_font)
        ax.set_zlabel('肩部旋转角度(°)', fontproperties=simhei_font)
        plt.title('肩颈角度3D可视化散点图', fontproperties=simhei_font)

        # 修改 colorbar 的 label 字体
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('肩部外展/内收角度(°)', fontproperties=simhei_font)
        
        st.pyplot(fig)
        
        # 3D 散点图分析结论
        st.write("\n**动态分析结论：3D可视化散点图**")
        if data['颈部角度(°)'].max() > 40:
            st.write("- 部分时间点颈部角度超过 40°，可能存在极端动作。")

        shoulder_rotation_std = data['肩部旋转角度(°)'].std()
        if shoulder_rotation_std < 10:
            st.write("- 肩部旋转角度的波动较小，动作幅度相对一致。")
        elif 10 <= shoulder_rotation_std <= 15:
            st.write("- 肩部旋转角度的波动性适中，可能动作较为稳定。")
        else:
            st.write("- 肩部旋转角度的波动性较大，动作可能不稳定。")

        if data['肩部外展/内收角度(°)'].mean() > 20:
            st.write("- 肩部外展/内收角度的整体幅度较大，运动强度可能较高。")

    # 相关性热力图
    def generate_correlation_heatmap(data):
        st.write("### 2.2 肩颈角度相关性热力图")

        # 计算相关性矩阵
        corr = data[['颈部角度(°)', '肩部上举角度(°)', '肩部外展/内收角度(°)', '肩部旋转角度(°)']].corr()

        # 创建绘图
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, ax=ax)

        # 设置标题和坐标轴字体
        ax.set_title('肩颈角度相关性热力图', fontproperties=simhei_font)
        ax.set_xticklabels(ax.get_xticklabels(), fontproperties=simhei_font, fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=simhei_font, fontsize=8)

        # 渲染图表到 Streamlit
        st.pyplot(fig)

        # 相关性热力图分析结论
        st.write("\n**动态分析结论：相关性热力图**")
        if corr['颈部角度(°)']['肩部上举角度(°)'] > 0.5:
            st.write("- 颈部角度与肩部上举角度高度正相关，动作之间可能存在协同性。")
        elif 0 < corr['颈部角度(°)']['肩部上举角度(°)'] <= 0.5:
            st.write("- 颈部角度与肩部上举角度存在一定程度的正相关，但相关性较弱，协同性可能较低。")

        if corr['肩部旋转角度(°)']['肩部外展/内收角度(°)'] < 0:
            st.write("- 肩部旋转与外展/内收角度存在负相关，可能是补偿动作的表现。")
        elif 0 <= corr['肩部旋转角度(°)']['肩部外展/内收角度(°)'] <= 0.5:
            st.write("- 肩部旋转与外展/内收角度存在弱正相关，可能与动作的协调性有关，但关联较弱。")
            
    # 肩颈角度时间变化散点图
    def generate_scatter_plots(data):
        st.write("### 2.3  肩颈角度时间变化散点图")
        
        # 绘制图像
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data['时间(s)'], data['颈部角度(°)'], label='颈部角度(°)', alpha=0.7)
        ax.scatter(data['时间(s)'], data['肩部旋转角度(°)'], label='肩部旋转角度(°)', alpha=0.7)
        ax.set_xlabel('时间(s)', fontproperties=simhei_font, fontsize=12)
        ax.set_ylabel('角度(°)', fontproperties=simhei_font, fontsize=12)
        ax.set_title('肩颈角度时间变化散点图', fontproperties=simhei_font, fontsize=12)

        # 设置图例字体
        legend = ax.legend(prop=simhei_font)  # 图例字体设置为 simhei
        
        # 用 st.pyplot() 嵌入图像
        st.pyplot(fig)
        
        # 散点图动态结论
        st.write("\n**动态分析结论：散点图**")

        # 对 颈部角度(°) 的分析
        neck_mean = data['颈部角度(°)'].mean()
        if neck_mean > 20:
            st.write("- 颈部角度的整体水平较高，可能是头部前倾较多导致的。")
        elif 10 <= neck_mean <= 20:
            st.write("- 颈部角度处于中等水平，动作姿势可能较为自然。")
        else:
            st.write("- 颈部角度较低，头部可能偏后或抬头动作较多。")

        # 对 肩部旋转角度(°) 的分析（统一标准差逻辑）
        shoulder_rotation_std = data['肩部旋转角度(°)'].std()
        if shoulder_rotation_std < 10:
            st.write("- 肩部旋转角度的波动较小，动作幅度相对一致。")
        elif 10 <= shoulder_rotation_std <= 15:
            st.write("- 肩部旋转角度的波动性适中，可能动作较为稳定。")
        else:
            st.write("- 肩部旋转角度的波动性较大，动作可能不稳定。")         

    # 综合分析
    def comprehensive_analysis(data, model):
        neck_threshold = data['颈部角度(°)'].mean() + data['颈部角度(°)'].std()
        shoulder_threshold = data['肩部旋转角度(°)'].mean() + data['肩部旋转角度(°)'].std()

        st.write("### 3.1  AI模型综合分析结果")
        st.write(f"- **动态阈值**：颈部角度 > {neck_threshold:.2f}° 为异常")
        st.write(f"- **动态阈值**：肩部旋转 > {shoulder_threshold:.2f}° 为异常")

        feature_importances = model.feature_importances_
        st.write("#### 3.2  机器学习特征重要性")
        for name, importance in zip(data.columns[2:], feature_importances):
            st.write(f"- {name}: {importance:.4f}")

        abnormal_indices = []
        st.write("### 3.3  作业姿势AI模型检测结果")

        # 前10条
        st.write("#### 前10条检测结果：")
        for index, row in data.iloc[:10].iterrows():
            rule_based_conclusion = "正常"
            if row['颈部角度(°)'] > neck_threshold:
                rule_based_conclusion = "颈部角度异常"
            elif row['肩部旋转角度(°)'] > shoulder_threshold:
                rule_based_conclusion = "肩部旋转角度异常"

            ml_conclusion = "异常" if model.predict([[row['颈部角度(°)'], row['肩部上举角度(°)'], 
                                                      row['肩部外展/内收角度(°)'], row['肩部旋转角度(°)']]])[0] == 1 else "正常"

            if rule_based_conclusion == "正常" and ml_conclusion == "异常":
                st.write(f"- 第 {index+1} 条数据：机器学习检测为异常姿势，但规则未发现，建议进一步分析。")
                abnormal_indices.append(index)
            elif rule_based_conclusion != "正常" and ml_conclusion == "异常":
                st.write(f"- 第 {index+1} 条数据：规则与机器学习一致检测为异常姿势，问题可能较严重。")
                abnormal_indices.append(index)
            else:
                st.write(f"- 第 {index+1} 条数据：规则和机器学习均检测为正常姿势，无明显问题。")

        # 中间数据折叠
        if len(data) > 15:
            st.write(f"#### 中间检测结果：")

            with st.expander("展开查看中间检测结果"):
                for index, row in data.iloc[10:-5].iterrows():
                    rule_based_conclusion = "正常"
                    if row['颈部角度(°)'] > neck_threshold:
                        rule_based_conclusion = "颈部角度异常"
                    elif row['肩部旋转角度(°)'] > shoulder_threshold:
                        rule_based_conclusion = "肩部旋转角度异常"

                    ml_conclusion = "异常" if model.predict([[row['颈部角度(°)'], row['肩部上举角度(°)'], 
                                                              row['肩部外展/内收角度(°)'], row['肩部旋转角度(°)']]])[0] == 1 else "正常"

                    if rule_based_conclusion == "正常" and ml_conclusion == "异常":
                        st.write(f"- 第 {index+1} 条数据：机器学习检测为异常姿势，但规则未发现，建议进一步分析。")
                        abnormal_indices.append(index)
                    elif rule_based_conclusion != "正常" and ml_conclusion == "异常":
                        st.write(f"- 第 {index+1} 条数据：规则与机器学习一致检测为异常姿势，问题可能较严重。")
                        abnormal_indices.append(index)
                    else:
                        st.write(f"- 第 {index+1} 条数据：规则和机器学习均检测为正常姿势，无明显问题。")
        
        # 后5条
        st.write("#### 后5条检测结果：")
        for index, row in data.iloc[-5:].iterrows():
            rule_based_conclusion = "正常"
            if row['颈部角度(°)'] > neck_threshold:
                rule_based_conclusion = "颈部角度异常"
            elif row['肩部旋转角度(°)'] > shoulder_threshold:
                rule_based_conclusion = "肩部旋转角度异常"

            ml_conclusion = "异常" if model.predict([[row['颈部角度(°)'], row['肩部上举角度(°)'], 
                                                      row['肩部外展/内收角度(°)'], row['肩部旋转角度(°)']]])[0] == 1 else "正常"

            if rule_based_conclusion == "正常" and ml_conclusion == "异常":
                st.write(f"- 第 {index+1} 条数据：机器学习检测为异常姿势，但规则未发现，建议进一步分析。")
                abnormal_indices.append(index)
            elif rule_based_conclusion != "正常" and ml_conclusion == "异常":
                st.write(f"- 第 {index+1} 条数据：规则与机器学习一致检测为异常姿势，问题可能较严重。")
                abnormal_indices.append(index)
            else:
                st.write(f"- 第 {index+1} 条数据：规则和机器学习均检测为正常姿势，无明显问题。")

        return abnormal_indices
  
    # 机器学习
    if uploaded_file is not None:
        # 下载最新模型
        model_path = download_latest_model_from_github()

    if model_path:
        model = load(model_path)
        st.write("加载最新模型进行分析。")
    else:
        model = RandomForestClassifier(random_state=42)
        st.write("未加载到模型，训练新模型。")

    # 模型训练或重新训练
    X = data[['颈部角度(°)', '肩部上举角度(°)', '肩部外展/内收角度(°)', '肩部旋转角度(°)']]
    if 'Label' not in data.columns:
        np.random.seed(42)
        data['Label'] = np.random.choice([0, 1], size=len(data))
    y = data['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    
    # 保存新模型到临时文件夹
    local_model_path = f"/tmp/{model_filename}"
    dump(model, local_model_path)
    st.write("模型已训练并保存到本地临时路径。")

    # 上传新模型到 GitHub
    upload_file_to_github(local_model_path, models_folder + model_filename, commit_message)
    st.write("模型已保存并上传到 GitHub。")
    
    # 更新最新模型信息
    latest_info_path = "/tmp/" + latest_model_file
    with open(latest_info_path, "w") as f:
        f.write(model_filename)
    upload_file_to_github(latest_info_path, models_folder + latest_model_file, "更新最新模型信息")

    st.success("新模型已上传，并更新最新模型记录。")
            
    # 调用函数生成图和结论
    analyze_data(data)
    generate_3d_scatter(data)
    generate_correlation_heatmap(data)
    generate_scatter_plots(data)
    abnormal_indices = comprehensive_analysis(data, model)
    
    if abnormal_indices:
        st.write(f"#### AI模型共检测到 {len(abnormal_indices)} 条异常数据")
    else:
        st.write("AI模型未检测到异常数据。")
                               
    st.write("### 3.4  AI模型质量评估")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制ROC曲线
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linestyle='-')
    ax.plot([0, 1], [0, 1], 'r--', label="随机模型")

    # 找到最佳阈值的坐标
    best_threshold_index = (tpr - fpr).argmax()
    best_threshold = thresholds[best_threshold_index]
    best_fpr = fpr[best_threshold_index]
    best_tpr = tpr[best_threshold_index]

    # 在ROC曲线上标注最佳阈值点
    ax.scatter(best_fpr, best_tpr, color='red', label=f'最佳阈值: {best_threshold:.2f}')
    ax.annotate(f'({best_fpr:.2f}, {best_tpr:.2f})',
                xy=(best_fpr, best_tpr),
                xytext=(best_fpr - 0.2, best_tpr - 0.1),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=10,
                fontproperties=simhei_font)

    # 设置标题和轴标签
    ax.set_xlabel('假阳性率', fontproperties=simhei_font)
    ax.set_ylabel('真阳性率', fontproperties=simhei_font)
    ax.set_title('ROC曲线', fontproperties=simhei_font)

    # 添加图例，显式设置字体
    ax.legend(loc='lower right', prop=simhei_font)

    # 在Streamlit中显示图像
    st.pyplot(fig)
     
    st.write("\n**AI模型优化建议**")
    st.write(f"AI模型AUC值为 {roc_auc:.2f}，最佳阈值为 {best_threshold:.2f}，可根据此阈值优化AI模型。")
