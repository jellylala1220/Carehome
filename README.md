# Care Home Analysis Dashboard

本项目为护理院数据分析的可视化Web仪表板，支持上传原始数据文件后，自动生成多种统计分析和交互图表。适用于数据分析师、护理院管理者等零基础用户。

---

## 功能简介

- **文件上传**：支持通过Web页面上传观察数据和生理参数定义文件
- **Overview统计**：自动识别有效/无效Care Home，并展示观测量降序表格
- **Care Home级分析**：可下拉选择Care Home，查看其使用情况和健康洞察（多折线/柱状交互图）
- **时间粒度切换**：支持日/周/月/年视角统计与对比
- **高风险与临床判断指标**：一键查看NEWS2高分比例、触发参数等
- **图表导出**：所有图表可直接导出为图片

---

## 安装环境

建议使用Python 3.8或以上版本。

1. **克隆或下载本项目文件**
2. **在命令行进入项目目录**
3. **安装依赖库**

   ```sh
   pip install -r requirements.txt


## 如果遇到kaleido相关报错，单独运行：
pip install kaleido

## 启动Web应用
streamlit run app.py

## 浏览器自动打开页面，如未自动打开，请访问
http://localhost:8501


## Q: 提示缺少某个库怎么办？
## A: 按提示用 pip install 库名 安装。例如 pip install streamlit。

## Q: Kaleido报错（导出图片相关）怎么办？
## A: 运行 pip install kaleido。


