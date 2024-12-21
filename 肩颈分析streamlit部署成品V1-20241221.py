#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from joblib import dump, load
import os

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 标题
st.title("肩颈角度动态分析与异常检测")
st.write("本人因AI工具结合人因规则与机器学习模型，自动检测异常作业姿势并提供可视化分析。")

# 数据加载与预处理
uploaded_file = st.file_uploader("上传肩颈角度数据文件 (CSV 格式)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = ['天(d)', '时间(s)', '颈部角度(°)', '肩部上举角度(°)', 
                    '肩部外展/内收角度(°)', '肩部旋转角度(°)']
    st.write("### 1.1  数据预览")
    st.write(data.head())
    
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
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('颈部角度(°)')
        ax.set_zlabel('肩部旋转角度(°)')
        plt.title('肩颈角度3D可视化散点图')
        fig.colorbar(scatter, ax=ax, label='肩部外展/内收角度(°)')
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
        st.write("### 2.2  肩颈角度相关性热力图")
        corr = data[['颈部角度(°)', '肩部上举角度(°)', '肩部外展/内收角度(°)', '肩部旋转角度(°)']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title('肩颈角度相关性热力图')
        st.pyplot(plt)
        
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
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('角度(°)')
        ax.legend()
        ax.set_title('肩颈角度时间变化散点图')
    
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
        st.write("#### 3.3  作业姿势AI模型检测结果")
        for index, row in data.iterrows():
            rule_based_conclusion = "正常"
            if row['颈部角度(°)'] > neck_threshold:
                rule_based_conclusion = "颈部角度异常"
            elif row['肩部旋转角度(°)'] > shoulder_threshold:
                rule_based_conclusion = "肩部旋转角度异常"

            ml_conclusion = "异常" if model.predict([[row['颈部角度(°)'], row['肩部上举角度(°)'], 
                                                      row['肩部外展/内收角度(°)'], row['肩部旋转角度(°)']]])[0] == 1 else "正常"

            if rule_based_conclusion == "正常" and ml_conclusion == "异常":
                st.write(f"- 第 {index} 条数据：机器学习检测为异常姿势，但规则未发现，建议进一步分析。")
                abnormal_indices.append(index)
            elif rule_based_conclusion != "正常" and ml_conclusion == "异常":
                st.write(f"- 第 {index} 条数据：规则与机器学习一致检测为异常姿势，问题可能较严重。")
                abnormal_indices.append(index)
            else:
                st.write(f"- 第 {index} 条数据：规则和机器学习均检测为正常姿势，无明显问题。")

        return abnormal_indices
  
    # 机器学习
    model_file = '肩颈分析-机器学习版模型.joblib'

    if os.path.exists(model_file):
        model = load(model_file)
        st.write("加载已有模型。")
    else:
        model = RandomForestClassifier(random_state=42)

    X = data[['颈部角度(°)', '肩部上举角度(°)', '肩部外展/内收角度(°)', '肩部旋转角度(°)']]
    if 'Label' not in data.columns:
        np.random.seed(42)
        data['Label'] = np.random.choice([0, 1], size=len(data))
    y = data['Label']

    if not os.path.exists(model_file):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        dump(model, model_file)
        st.write(f"模型已保存：{model_file}")
        
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
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel('假阳性率')
    ax.set_ylabel('真阳性率')
    ax.set_title('ROC曲线')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    best_threshold_index = (tpr - fpr).argmax()
    best_threshold = thresholds[best_threshold_index]
    
    st.write("\n**AI模型优化建议**")
    st.write(f"AI模型AUC值为 {roc_auc:.2f}，最佳阈值为 {best_threshold:.2f}，可根据此阈值优化AI模型。")
     

