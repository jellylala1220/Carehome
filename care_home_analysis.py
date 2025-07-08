# 另外一个包来制作care home daashboard
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from datetime import datetime
import numpy as np
from data_processor import DataProcessor
import os
from tkhtmlview import HTMLLabel
import plotly.io as pio
from plotly_helper import *

class CareHomeAnalysis:
    def __init__(self, root):
        self.root = root
        self.root.title("Care Home Analysis Dashboard")
        self.root.geometry("1200x800")
        
        # 初始化数据处理器
        self.data_processor = DataProcessor()
        
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 显示主页
        self.show_home_page()
        
    def show_home_page(self):
        # 清空主框架
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # 创建欢迎信息
        welcome_frame = ttk.Frame(self.main_frame)
        welcome_frame.pack(pady=20)
        
        welcome_label = tk.Label(
            welcome_frame,
            text="Welcome to Care Home Analysis Dashboard",
            font=("Arial", 24, "bold"),
            fg="#6C3483"
        )
        welcome_label.pack(pady=10)
        
        # 创建文件上传区域
        upload_frame = ttk.LabelFrame(self.main_frame, text="Data Upload", padding=20)
        upload_frame.pack(pady=20, padx=50, fill=tk.X)
        
        # 主数据文件上传
        main_data_frame = ttk.Frame(upload_frame)
        main_data_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(
            main_data_frame,
            text="Main Data File (Excel):",
            font=("Arial", 12)
        ).pack(side=tk.LEFT, padx=5)
        
        self.main_data_path_var = tk.StringVar()
        ttk.Entry(
            main_data_frame,
            textvariable=self.main_data_path_var,
            width=50,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            main_data_frame,
            text="Browse",
            command=self.browse_file
        ).pack(side=tk.LEFT, padx=5)
        
        # 上传按钮
        upload_btn = tk.Button(
            upload_frame,
            text="Upload Data",
            font=("Arial", 12, "bold"),
            bg="#B39DDB",
            fg="white",
            height=2,
            width=20,
            command=self.upload_data
        )
        upload_btn.pack(pady=20)
        
        # 创建分析按钮区域（初始隐藏）
        self.analysis_btn_frame = ttk.Frame(self.main_frame)
        self.analysis_btn_frame.pack(pady=20)
        self.analysis_btn_frame.pack_forget()  # 初始隐藏
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Main Data File",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if file_path:
            self.main_data_path_var.set(file_path)
                
    def upload_data(self):
        main_data_path = self.main_data_path_var.get()
        
        if not main_data_path:
            messagebox.showerror("Error", "Please select main data file")
            return
            
        try:
            # 加载主数据
            self.data_processor.load_data(main_data_path)
            
            # 自动加载固定的生理参数文件
            physio_path = os.path.join("Data", "Physiological parameters and algorithm table.xlsx")
            if os.path.exists(physio_path):
                self.data_processor.load_physio_params(physio_path)
            else:
                messagebox.showwarning("Warning", "Physiological parameters file not found")
                
            # 显示分析按钮
            self.show_analysis_buttons()
            
            messagebox.showinfo("Success", "Data uploaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
            
    def show_analysis_buttons(self):
        # 显示分析按钮区域
        self.analysis_btn_frame.pack(pady=20)
        
        # 清空现有按钮
        for widget in self.analysis_btn_frame.winfo_children():
            widget.destroy()
            
        # Care Home Level Analysis按钮
        care_home_btn = tk.Button(
            self.analysis_btn_frame,
            text="Care Home Level Analysis",
            font=("Arial", 16, "bold"),
            bg="#B39DDB",
            fg="white",
            height=2,
            width=25,
            command=self.show_care_home_analysis
        )
        care_home_btn.pack(pady=20)
        
        # Regional Analysis按钮（暂时禁用）
        regional_btn = tk.Button(
            self.analysis_btn_frame,
            text="Regional Analysis (Coming Soon)",
            font=("Arial", 16, "bold"),
            bg="#E0E0E0",
            fg="gray",
            height=2,
            width=25,
            state="disabled"
        )
        regional_btn.pack(pady=10)
        
    def show_care_home_analysis(self):
        # 清空主框架
        for widget in self.main_frame.winfo_children():
            widget.destroy()
            
        # 创建顶部控制栏
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # 返回主页按钮
        back_btn = tk.Button(
            control_frame,
            text="← Back to Home",
            font=("Arial", 12),
            command=self.show_home_page
        )
        back_btn.pack(side=tk.LEFT, padx=10)
        
        # Care Home选择
        ttk.Label(control_frame, text="Care Home:").pack(side=tk.LEFT, padx=5)
        self.care_home_var = tk.StringVar()
        self.care_home_combo = ttk.Combobox(
            control_frame,
            textvariable=self.care_home_var,
            state="readonly",
            width=20
        )
        self.care_home_combo.pack(side=tk.LEFT, padx=5)
        
        # Info按钮
        info_btn = tk.Button(
            control_frame,
            text="Care Home Info",
            command=self.show_care_home_info
        )
        info_btn.pack(side=tk.LEFT, padx=5)
        
        # 数据加载按钮
        load_btn = tk.Button(
            control_frame,
            text="Load Data",
            command=self.load_data
        )
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建分析模块按钮区
        analysis_btn_frame = ttk.Frame(self.main_frame)
        analysis_btn_frame.pack(pady=20)
        
        # Usage Analysis按钮
        usage_btn = tk.Button(
            analysis_btn_frame,
            text="Usage Analysis",
            font=("Arial", 14, "bold"),
            bg="#B39DDB",
            fg="white",
            height=2,
            width=20,
            command=lambda: self.show_analysis_page("Usage Analysis")
        )
        usage_btn.pack(side=tk.LEFT, padx=10)
        
        # Health Insights按钮
        health_btn = tk.Button(
            analysis_btn_frame,
            text="Health Insights",
            font=("Arial", 14, "bold"),
            bg="#B39DDB",
            fg="white",
            height=2,
            width=20,
            command=lambda: self.show_analysis_page("Health Insights")
        )
        health_btn.pack(side=tk.LEFT, padx=10)
        
        # 默认显示Usage Analysis
        self.show_analysis_page("Usage Analysis")
        
    def show_care_home_info(self):
        care_home_id = self.care_home_var.get()
        if not care_home_id:
            return
            
        info = self.data_processor.get_care_home_info(care_home_id)
        
        # 创建信息窗口
        info_window = tk.Toplevel(self.root)
        info_window.title(f"Care Home {care_home_id} Information")
        info_window.geometry("400x300")
        
        # 显示信息
        info_text = f"Care Home Name: {info.get('Care Home Name', '-')}\n" \
                   f"Number of Beds: {info.get('No of Beds', '-')}\n" \
                   f"Area: {info.get('Area', '-')}\n" \
                   f"Type: {info.get('Type', '-')}\n" \
                   f"Amount of Asset: {info.get('Amount of Asset', '-')}\n" \
                   f"Provider company: {info.get('Provider company', '-')}"
        
        info_label = tk.Label(
            info_window,
            text=info_text,
            font=("Arial", 12),
            justify="left",
            padx=20,
            pady=20
        )
        info_label.pack(expand=True)
        
    def load_data(self):
        data_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("Excel files", "*.xlsx")]
        )
        if data_path:
            try:
                self.data_processor.load_data(data_path)
                care_homes = self.data_processor.get_care_homes()
                if care_homes:
                    self.care_home_combo['values'] = care_homes
                    self.care_home_var.set(care_homes[0])
                else:
                    print("Warning: No Care Home data found")
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                
    def show_analysis_page(self, analysis_type):
        # 清空分析区域
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.Frame) and widget.winfo_children():
                widget.destroy()
                
        # 创建分析区域框架
        analysis_frame = ttk.Frame(self.main_frame)
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 时间粒度选择（右上角）
        time_frame = ttk.Frame(analysis_frame)
        time_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(time_frame, text="Time Granularity:").pack(side=tk.RIGHT, padx=5)
        self.time_granularity_var = tk.StringVar(value="Monthly")
        time_granularity_combo = ttk.Combobox(
            time_frame,
            textvariable=self.time_granularity_var,
            values=["Daily", "Weekly", "Monthly", "Yearly"],
            state="readonly",
            width=10
        )
        time_granularity_combo.pack(side=tk.RIGHT)
        
        # 更新图表按钮
        update_btn = ttk.Button(
            time_frame,
            text="Update Charts",
            command=lambda: self.update_charts(analysis_type)
        )
        update_btn.pack(side=tk.RIGHT, padx=10)
        
        # 创建图表区域
        self.charts_frame = ttk.Frame(analysis_frame)
        self.charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 默认显示图表
        self.update_charts(analysis_type)
        
    def update_charts(self, analysis_type):
        if self.data_processor.data is None:
            print("Please load data first")
            return
            
        care_home_id = self.care_home_var.get()
        if not care_home_id:
            print("Please select a Care Home")
            return
            
        time_granularity = self.time_granularity_var.get()
        filtered_data = self.data_processor.filter_by_care_home(care_home_id)
        
        # 根据粒度选择tickformat
        if time_granularity == 'Daily':
            tickformat = '%Y-%m-%d'
        elif time_granularity == 'Weekly':
            tickformat = '%Y-%m-%d'
        elif time_granularity == 'Monthly':
            tickformat = '%Y-%m'
        else:  # Yearly
            tickformat = '%Y'
            
        # 清空图表区域
        for widget in self.charts_frame.winfo_children():
            widget.destroy()
            
        if analysis_type == "Usage Analysis":
            # Usage Count
            result = self.data_processor.calculate_usage(filtered_data, time_granularity)
            fig1 = plot_usage_volume_over_time(result, care_home_id, tickformat)
            for trace in fig1.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig1.update_layout(
                xaxis_title='Time',
                yaxis_title='Usage Count',
                yaxis=dict(tickformat='.0f'),
                height=600
            )
            self.show_plotly_figure(fig1)
            
            # Usage per Bed
            beds = filtered_data['No of Beds'].iloc[0] if 'No of Beds' in filtered_data.columns else 10
            result = self.data_processor.calculate_usage_per_bed(filtered_data, time_granularity, beds)
            fig2 = plot_usage_per_bed(result, care_home_id, tickformat)
            for trace in fig2.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig2.update_layout(
                xaxis_title='Time',
                yaxis_title='Usage per Bed',
                yaxis=dict(tickformat='.0f'),
                height=600
            )
            self.show_plotly_figure(fig2)
            
            # Coverage Percentage
            result = self.data_processor.calculate_coverage_percentage(filtered_data)
            fig3 = plot_coverage_percentage(result, care_home_id, tickformat)
            for trace in fig3.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig3.update_layout(
                xaxis_title='Month',
                yaxis_title='Coverage %',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=600
            )
            fig3.update_traces(hovertemplate='%{y:.1%}')
            self.show_plotly_figure(fig3)
            
        else:  # Health Insights
            # NEWS2 Counts
            result = self.data_processor.calculate_news2_counts(filtered_data)
            fig1 = plot_news2_score_category_counts(result, care_home_id, tickformat)
            for trace in fig1.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig1.update_layout(
                xaxis_title='Time',
                yaxis_title='News2 Counts',
                yaxis=dict(tickformat='.0f'),
                height=600
            )
            self.show_plotly_figure(fig1)
            
            # High Risk Proportion
            result = self.data_processor.calculate_high_risk_proportion(filtered_data)
            fig2 = plot_high_risk_prop(result, time_granularity)
            for trace in fig2.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig2.update_layout(
                xaxis_title='Time',
                yaxis_title='High Risk %',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=600
            )
            self.show_plotly_figure(fig2)
            
            # Concern Proportion
            result = self.data_processor.calculate_concern_proportion(filtered_data)
            fig3 = plot_high_risk_prop(result, time_granularity)
            for trace in fig3.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig3.update_layout(
                xaxis_title='Time',
                yaxis_title='Concern %',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=600
            )
            self.show_plotly_figure(fig3)
            
            # Clinical Judgement Accuracy
            result = self.data_processor.calculate_clinical_judgement_accuracy(filtered_data)
            fig4 = plot_staff_judgement_accuracy(result, care_home_id, tickformat)
            for trace in fig4.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig4.update_layout(
                xaxis_title='Time',
                yaxis_title='Clinical Judgement Accuracy %',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=600
            )
            self.show_plotly_figure(fig4)
            
            # Physio Parameters Analysis
            result = self.data_processor.analyze_all_physio_parameters(filtered_data)
            fig5 = plot_all_physio_parameters(result, care_home_id, tickformat)
            for trace in fig5.data:
                if hasattr(trace, 'mode'):
                    trace.mode = 'lines+markers'
            fig5.update_layout(
                xaxis_title='Time',
                yaxis_title='Trigger Rate %',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=600
            )
            self.show_plotly_figure(fig5)
            
    def show_plotly_figure(self, fig):
        from plotly_helper import show_plotly_in_webview
        show_plotly_in_webview(fig)

def plot_high_risk_prop(hi_data, period):
    s = hi_data['high_risk_prop']
    if s is None or s.empty:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        fig.update_layout(height=500)
        return fig
    fig = px.line(x=s.index, y=s.values, title=f'High Risk (NEWS2≥6) Proportion ({period})')
    fig.update_traces(mode='lines+markers')
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='High Risk (%)',
        yaxis=dict(range=[0, 100], tickformat='.0f'),
        height=600
    )
    return fig

def plot_coverage_percentage(df, care_home_id, tickformat='%Y-%m'):
    if df is None or df.empty or 'coverage' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="无数据", xref="paper", yref="paper", showarrow=False, font=dict(size=20))
        return fig
    # 构造标注文本
    text_labels = [
        f"{int(row['days_with_obs'])}/{int(row['total_days'])}={row['coverage']:.1f}%"
        for _, row in df.iterrows()
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['coverage'],
        mode='lines+markers+text',
        name='Coverage',
        text=text_labels,
        textposition='top center',
        hovertemplate=(
            "Month: %{x|%Y-%m}<br>"
            "Coverage: %{y:.1f}%<br>"
            "Days: %{customdata[0]}/%{customdata[1]}"
        ),
        customdata=np.stack([df['days_with_obs'], df['total_days']], axis=-1)
    ))
    fig.update_layout(
        title=f'Coverage Percentage (Monthly) - {care_home_id}',
        xaxis_title='Month',
        yaxis_title='Coverage %',
        yaxis=dict(
            range=[0, 100],
            tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ticktext=['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
            constrain='range',
            automargin=True
        ),
        height=600,
        margin=dict(t=80, b=60, l=60, r=40)
    )
    return fig

if __name__ == "__main__":
    root = tk.Tk()
    app = CareHomeAnalysis(root)
    root.mainloop()
