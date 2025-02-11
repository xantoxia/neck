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
commit_message = "从Streamlit更新模型文件"  # 提交信息

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
simhei_font = font_manager.FontProperties(fname="SimHei.ttf")
plt.rcParams['font.family'] = simhei_font.get_name()  # 使用 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# Streamlit 标题
st.title("肩颈角度分析与异常检测")
st.write("本人因AI工具结合规则与机器学习模型，可以自动检测异常作业姿势并提供可视化分析。")

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
    # 提取文件名并去掉扩展名
    csv_file_name = os.path.splitext(uploaded_file.name)[0]
     # 使用 HTML 格式设置字体颜色为蓝色
    st.markdown(f"<h3 style='color:blue;'>{csv_file_name} 肩颈作业姿势分析</h3>", unsafe_allow_html=True)

    # 读取数据
    data = pd.read_csv(uploaded_file)
    data.columns = ['工站(w)', '时间(s)', '颈部角度(°)', '肩部前屈角度(°)', 
                    '肩部外展角度(°)']

    # 显示数据预览
    st.write("### 1.1  数据预览")
    data_reset = data.copy()
    data_reset.index += 1
    data_reset.index.name = "序号"
    st.write(data_reset.head())

    # 按工站汇总计算
    def summarize_by_station(data):
        st.write("### 1.2  数据统计分析")
    
        # 按 '工站(w)' 分组并计算统计特性
        station_summary = data.groupby('工站(w)').agg({
            '时间(s)': ['count'],
            '颈部角度(°)': ['mean', 'min', 'max', 'std'],
            '肩部前屈角度(°)': ['mean', 'min', 'max', 'std'],
            '肩部外展角度(°)': ['mean', 'min', 'max', 'std']
        })

        # 调整列名格式
        station_summary.columns = ['_'.join(col).strip() for col in station_summary.columns.values]
        station_summary.reset_index(inplace=True)

        # 限制小数点位数为最多2位
        station_summary = station_summary.round(2)
  
        # 显示汇总统计结果
        st.write(station_summary)

    # 调用函数
    summarize_by_station(data)

    # 3D 散点图
    def generate_3d_scatter(data):
        st.write("### 2.1  肩颈角度3D可视化散点图")
    
    # 按 '工站(w)' 分组
        grouped = data.groupby('工站(w)')
    
    # 遍历每个工站的数据
        for station, group_data in grouped:
            st.write(f"#### 工站 {station} 的3D散点图")
        
        # 创建图形
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点图
            scatter = ax.scatter(
                group_data['时间(s)'], 
                group_data['颈部角度(°)'], 
                group_data['肩部前屈角度(°)'], 
                c=group_data['肩部外展角度(°)'], 
                cmap='viridis'
            )
        
        # 设置坐标轴标签
            ax.set_xlabel('时间(s)', fontproperties=simhei_font)
            ax.set_ylabel('颈部角度(°)', fontproperties=simhei_font)
            ax.set_zlabel('肩部前屈角度(°)', fontproperties=simhei_font)
        
        # 设置标题
            plt.title(f'工站 {station} 肩颈角度3D可视化散点图', fontproperties=simhei_font)
        
        # 添加 colorbar
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('肩部外展角度(°)', fontproperties=simhei_font)
        
        # 显示图形
            st.pyplot(fig)
        
        # 动态分析结论
            st.write(f"**工站 {station} 的动态分析结论：**")
            neck_Flexion_max = group_data['颈部角度(°)'].max()
            if neck_Flexion_max < 20:
                st.write("- 作业时颈部角度处于20°之内，MSD风险较低。")
            elif 20 <= neck_Flexion_max <= 40:
                st.write("- 部分时间点颈部角度超过20°，存在一定的MSD风险。")
            else:
                st.write("- 部分时间点颈部角度超过40°，请注意可能存在极端低头动作。")
        
            shoulder_Flexion_max = group_data['肩部前屈角度(°)'].max()
            if shoulder_Flexion_max < 15:
                st.write("- 肩部前屈角度的波动较小，动作幅度相对一致。")
            elif shoulder_Flexion_max >= 45:
                st.write("- 部分时间点肩部前屈角度大于45°，请注意作业时是否有手部支撑。")

            if group_data['肩部外展角度(°)'].mean() > 20:
                st.write("- 肩部外展角度的整体幅度较大，上臂作业时运动强度可能较高。")

    # 调用函数
    generate_3d_scatter(data)

    # 相关性热力图
    def generate_correlation_heatmap(data):
        st.write("### 2.2 肩颈角度相关性热力图")
    
        # 按 '工站(w)' 分组
        grouped = data.groupby('工站(w)')
    
        # 遍历每个工站的数据
        for station, group_data in grouped:
            st.write(f"#### 工站 {station} 的相关性热力图")
        
            # 计算相关性矩阵（移除肩部旋转角度）
            corr = group_data[['颈部角度(°)', '肩部前屈角度(°)', '肩部外展角度(°)']].corr()
 
            # 创建绘图
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, ax=ax)

            # 设置标题和坐标轴字体
            ax.set_title(f'工站 {station} 的肩颈角度相关性热力图', fontproperties=simhei_font)
            ax.set_xticklabels(ax.get_xticklabels(), fontproperties=simhei_font, fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontproperties=simhei_font, fontsize=8)

            # 渲染图表到 Streamlit
            st.pyplot(fig)
        
            # 分析动态结论
            st.write(f"**工站 {station} 的动态分析结论：**")
        
            # 分析颈部角度和肩部前屈角度
            if corr.loc['颈部角度(°)', '肩部前屈角度(°)'] > 0.5:
                st.write("- 颈部角度与肩部前屈角度高度正相关，动作之间可能存在协同性。")
            elif 0 < corr.loc['颈部角度(°)', '肩部前屈角度(°)'] <= 0.5:
                st.write("- 颈部角度与肩部前屈角度存在一定程度的正相关，但相关性较弱，协同性可能较低。")

            # 分析颈部角度和肩部外展角度
            if corr.loc['颈部角度(°)', '肩部外展角度(°)'] > 0.5:
                st.write("- 颈部角度与肩部外展角度高度正相关，表明两者可能存在较强的协同动作趋势。")
            elif 0 < corr.loc['颈部角度(°)', '肩部外展角度(°)'] <= 0.5:
                st.write("- 颈部角度与肩部外展角度存在一定程度的正相关，但相关性较弱，协同性可能较低。")
            elif corr.loc['颈部角度(°)', '肩部外展角度(°)'] < 0:
                st.write("- 颈部角度与肩部外展角度呈负相关，可能表现为动作补偿或反向趋势。")

    # 调用函数生成图和结论
    generate_correlation_heatmap(data)
            
    # 肩颈角度时间变化折线图（带水平预警线）
    def generate_line_plots_with_threshold(data):
        st.write("### 2.3 肩颈角度时间变化折线图")
    
        # 按 '工站(w)' 分组
        grouped = data.groupby('工站(w)')
    
        # 遍历每个工站的数据
        for station, group_data in grouped:
            st.write(f"#### 工站 {station} 的肩颈角度时间变化折线图")
        
            # 绘制图像
            fig, ax = plt.subplots(figsize=(10, 6))
        
            # 绘制折线图
            ax.plot(group_data['时间(s)'], group_data['颈部角度(°)'], label='颈部角度(°)', color='blue', linewidth=2)
            ax.plot(group_data['时间(s)'], group_data['肩部前屈角度(°)'], label='肩部前屈角度(°)', color='green', linewidth=2)
        
            # 添加水平预警线
            ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='颈部角度预警线 (20°)')
            ax.axhline(y=45, color='orange', linestyle='--', linewidth=1.5, label='肩部前屈角度预警线 (45°)')
        
            # 设置坐标轴和标题
            ax.set_xlabel('时间(s)', fontproperties=simhei_font, fontsize=12)
            ax.set_ylabel('角度(°)', fontproperties=simhei_font, fontsize=12)
            ax.set_title(f'工站 {station} 的肩颈角度时间变化折线图', fontproperties=simhei_font, fontsize=12)
        
            # 设置图例
            ax.legend(prop=simhei_font, fontsize=10)  # 图例字体设置为 simhei
        
            # 渲染图像
            st.pyplot(fig)
        
            # 动态分析结论
            st.write(f"**工站 {station} 的动态分析结论：**")
        
            # 颈部角度分析
            neck_exceed_count = (group_data['颈部角度(°)'] > 20).sum()
            total_time_points = len(group_data)  # 总时间点数
            neck_exceed_ratio = neck_exceed_count / total_time_points  # 超过 20° 的时间点占比
            
            if neck_exceed_count > 0:
                # 初始化变量，确保在所有条件下都有值
                neck_risk_level = "轻度"
                neck_color = "black"
                
                # 根据占比分类 MSD 风险等级
                if neck_exceed_ratio > 0.5:
                    risk_level = "较高"
                    neck_color = "red"
                elif neck_exceed_ratio >= 0.25:
                    risk_level = "中等"
                    neck_color = "orange"
            
                st.markdown(f"<span style='color:{neck_color};'>- 有 {neck_exceed_count} 个时间点颈部角度超过 20°，占比 {neck_exceed_ratio:.2%}，颈部存在 {neck_risk_level} MSD 风险。</span>", unsafe_allow_html=True)
            else:
                st.write("-作业时颈部角度未超过20°，MSD风险较低。")
        
            # 肩部前屈角度分析
            shoulder_exceed_count = (group_data['肩部前屈角度(°)'] > 45).sum()
            shoulder_exceed_ratio = shoulder_exceed_count / total_time_points  # 超过 45° 的时间点占比

            if shoulder_exceed_count > 0:
                # 初始化变量，确保在所有条件下都有值
                shoulder_risk_level = "轻度"
                shoulder_color = "black"
                
                # 根据占比分类 MSD 风险等级
                if shoulder_exceed_ratio > 0.5:
                    shoulder_risk_level = "较高"
                    shoulder_color = "red"
                elif shoulder_exceed_ratio >= 0.25:
                    shoulder_risk_level = "中等"
                    shoulder_color = "orange"
            
                st.markdown(f"<span style='color:{shoulder_color};'>- 有 {shoulder_exceed_count} 个时间点肩部前屈角度超过 45°，占比 {shoulder_exceed_ratio:.2%}，肩部存在 {shoulder_risk_level} MSD 风险。</span>", unsafe_allow_html=True)
            else:
                st.write("-作业时肩部前屈角度未超过45°，动作幅度较为自然，MSD风险较低。")
                
    # 调用函数生成图和结论
    generate_line_plots_with_threshold(data)

     # 综合分析
    def comprehensive_analysis_by_workstation(data, model):

        st.write("### 3.1  机器学习AI模型分析结果")
        
        # 按 '工站(w)' 分组
        grouped = data.groupby('工站(w)')

        # 用于记录所有工站的异常索引
        total_abnormal_indices = []
    
        # 遍历每个工站的数据
        for station, group_data in grouped:
            st.write(f"#### 工站{station}的AI模型分析结果")
        
           # 动态阈值计算
            neck_threshold = group_data['颈部角度(°)'].mean() + group_data['颈部角度(°)'].std()
            shoulder_threshold = group_data['肩部前屈角度(°)'].mean() + group_data['肩部前屈角度(°)'].std()

            # 输出动态阈值
            st.write(f"- **动态阈值**：颈部角度 > {neck_threshold:.2f}° 为异常")
            st.write(f"- **动态阈值**：肩部前屈 > {shoulder_threshold:.2f}° 为异常")

            # 特征重要性
            st.write("##### 机器学习特征重要性")
            feature_importances = model.feature_importances_
            for name, importance in zip(group_data.columns[2:], feature_importances):
                st.write(f"- {name}: {importance:.4f}")

            # 重置序号
            group_data = group_data.reset_index(drop=True)     
            
            # AI模型检测结果
            abnormal_indices = []
            st.write(f"##### 工站{station}的逐条数据AI分析检测结果")
        
            # 前5条
            st.write(f"###### 工站{station}的前5条数据检测结果：")
            for i, row in group_data.iloc[:5].iterrows():
                rule_based_conclusion = "正常"
                if row['颈部角度(°)'] > neck_threshold:
                    rule_based_conclusion = "颈部角度异常"
                elif row['肩部前屈角度(°)'] > shoulder_threshold:
                    rule_based_conclusion = "肩部前屈角度异常"

                ml_conclusion = "异常" if model.predict([[row['颈部角度(°)'], row['肩部前屈角度(°)'], 
                                                          row['肩部外展角度(°)']]])[0] == 1 else "正常"

                if rule_based_conclusion == "正常" and ml_conclusion == "异常":
                    st.write(f"- 第 {i+1} 条数据：机器学习检测为异常姿势，但规则未发现，建议进一步分析。")
                    abnormal_indices.append(i)
                elif rule_based_conclusion != "正常" and ml_conclusion == "异常":
                    st.write(f"- 第 {i+1} 条数据：规则与机器学习一致检测为异常姿势，问题可能较严重。")
                    abnormal_indices.append(i)
                elif rule_based_conclusion != "正常" and ml_conclusion == "正常":
                    st.write(f"- 第 {i+1} 条数据：规则检测为异常姿势，但机器学习未检测为异常，建议评估规则的适用性。")
                    abnormal_indices.append(i)
                else:
                    st.write(f"- 第 {i+1} 条数据：规则和机器学习均检测为正常姿势，无明显问题。")
        
            # 中间数据折叠
            if len(group_data) > 10:
                st.write(f"###### 工站{station}的中间数据检测结果：")
                with st.expander(f"展开查看工站{station}的中间数据检测结果"):
                    for i, row in group_data.iloc[5:-5].iterrows():
                        rule_based_conclusion = "正常"
                        if row['颈部角度(°)'] > neck_threshold:
                            rule_based_conclusion = "颈部角度异常"
                        elif row['肩部前屈角度(°)'] > shoulder_threshold:
                            rule_based_conclusion = "肩部前屈角度异常"

                        ml_conclusion = "异常" if model.predict([[row['颈部角度(°)'], row['肩部前屈角度(°)'], 
                                                              row['肩部外展角度(°)']]])[0] == 1 else "正常"

                        if rule_based_conclusion == "正常" and ml_conclusion == "异常":
                            st.write(f"- 第 {i+1} 条数据：机器学习检测为异常姿势，但规则未发现，建议进一步分析。")
                            abnormal_indices.append(i)
                        elif rule_based_conclusion != "正常" and ml_conclusion == "异常":
                            st.write(f"- 第 {i+1} 条数据：规则与机器学习一致检测为异常姿势，问题可能较严重。")
                            abnormal_indices.append(i)
                        elif rule_based_conclusion != "正常" and ml_conclusion == "正常":
                            st.write(f"- 第 {i+1} 条数据：规则检测为异常姿势，但机器学习未检测为异常，建议评估规则的适用性。")
                            abnormal_indices.append(i)
                        else:
                            st.write(f"- 第 {i+1} 条数据：规则和机器学习均检测为正常姿势，无明显问题。")
        
            # 后5条
            st.write(f"###### 工站{station}的后5条数据检测结果：")
            for i, row in group_data.iloc[-5:].iterrows():
                rule_based_conclusion = "正常"
                if row['颈部角度(°)'] > neck_threshold:
                    rule_based_conclusion = "颈部角度异常"
                elif row['肩部前屈角度(°)'] > shoulder_threshold:
                    rule_based_conclusion = "肩部前屈角度异常"

                ml_conclusion = "异常" if model.predict([[row['颈部角度(°)'], row['肩部前屈角度(°)'], 
                                                          row['肩部外展角度(°)']]])[0] == 1 else "正常"

                if rule_based_conclusion == "正常" and ml_conclusion == "异常":
                    st.write(f"- 第 {i+1} 条数据：机器学习检测为异常姿势，但规则未发现，建议进一步分析。")
                    abnormal_indices.append(i)
                elif rule_based_conclusion != "正常" and ml_conclusion == "异常":
                    st.write(f"- 第 {i+1} 条数据：规则与机器学习一致检测为异常姿势，问题可能较严重。")
                    abnormal_indices.append(i)
                elif rule_based_conclusion != "正常" and ml_conclusion == "正常":
                    st.write(f"- 第 {i+1} 条数据：规则检测为异常姿势，但机器学习未检测为异常，建议评估规则的适用性。")
                    abnormal_indices.append(i)
                else:
                    st.write(f"- 第 {i+1} 条数据：规则和机器学习均检测为正常姿势，无明显问题。")
    
            # 总结性描述
            if abnormal_indices:
                st.write(f"##### 工站{station}总结：AI模型共检测到 {len(abnormal_indices)} 条异常数据。")
            else:
                st.write(f"##### 工站{station}总结：AI模型未检测到异常数据。")
        
             # 记录工站异常数据索引
            total_abnormal_indices.extend(abnormal_indices)
        
        # 返回所有工站的异常数据索引
        return total_abnormal_indices
  
    # 机器学习
    if uploaded_file is not None:
        # 增加延迟
        time.sleep(1)
        
        # 下载最新模型
        model_path = download_latest_model_from_github()

    if model_path:
        model = load(model_path)
        st.write("加载最新模型进行分析...")
    else:
        model = RandomForestClassifier(random_state=42)
        st.write("未加载到模型，训练新模型...")
    
    # 模型训练或重新训练
    X = data[['颈部角度(°)', '肩部前屈角度(°)', '肩部外展角度(°)']]
    if 'Label' not in data.columns:
        np.random.seed(42)
        data['Label'] = np.random.choice([0, 1], size=len(data))
    y = data['Label']

    # 数据预处理：重新定义标签
    data['Label'] = ((data['颈部角度(°)'] > 20) | (data['Label'] == 1)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)   
    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.4).astype(int)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 调用函数生成图和结论
    total_abnormal_indices = comprehensive_analysis_by_workstation(data, model)
    
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
    
    st.write("#### 页面导出")
    st.info("如需导出页面为 html 文件，请在浏览器中按 `Ctrl+S`，然后进行保存。")
