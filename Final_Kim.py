import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO, StringIO
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân Tích Tuyển Dụng Việt Nam", page_icon="💼", layout="wide", initial_sidebar_state="expanded")

# Thiết lập bảng màu
sns.set_palette("colorblind")
PALETTE = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

# --- Load Environment Variables ---
load_dotenv()  # Load variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configure AI ---
MODEL_NAME = "gemini-2.0-flash"

        
# --- Hàm tạo PDF từ figure ---
def save_fig_to_pdf(fig):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    buffer.seek(0)
    return buffer

# --- Hàm tải và tiền xử lý dữ liệu ---
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file):
    try:
        df = pd.read_csv(file, encoding='utf-8', quoting=csv.QUOTE_ALL, na_values=["None", " ", "UNKNOWN", -1, 999, "NA", "N/A", "NULL"])
        numeric_cols = ['min_salary_mil_vnd', 'max_salary_mil_vnd', 'min_experience_years', 'max_experience_years']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        categorical_cols = ['primary_location', 'primary_category', 'position', 'order']
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        # Đảm bảo cột job_requirements là chuỗi
        if 'job_requirements' in df.columns:
            df['job_requirements'] = df['job_requirements'].astype(str).fillna('')
        return df
    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        st.stop()

# --- Hàm dự đoán lương ---
def predict_salary(df):
    # Chọn các đặc trưng và mục tiêu
    features = ['min_experience_years', 'primary_category', 'primary_location']
    target = 'min_salary_mil_vnd'
    
    # Lọc dữ liệu
    df_model = df[features + [target]].dropna()
    if df_model.empty:
        return None, None, None
    
    X = df_model[features]
    y = df_model[target]
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tạo pipeline với mã hóa one-hot cho các cột phân loại
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['primary_category', 'primary_location']),
            ('num', 'passthrough', ['min_experience_years'])
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Tạo DataFrame cho kết quả
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    return model, results, X_test

# --- Hàm trích xuất kỹ năng ---
def extract_skills(df):
    # Danh sách kỹ năng phổ biến (có thể mở rộng)
    skills_list = [
        'python', 'sql', 'java', 'javascript', 'aws', 'docker', 'kubernetes',
        'excel', 'power bi', 'tableau', 'management', 'communication', 'leadership',
        'marketing', 'sales', 'finance', 'accounting', 'design', 'ui/ux'
    ]
    
    # Gộp tất cả job_requirements thành một văn bản
    text = ' '.join(df['job_requirements'].str.lower())
    
    # Tìm các kỹ năng trong văn bản
    skill_counts = Counter()
    for skill in skills_list:
        count = len(re.findall(r'\b' + re.escape(skill) + r'\b', text))
        if count > 0:
            skill_counts[skill] = count
    
    # Lấy top 10 kỹ năng
    top_skills = skill_counts.most_common(10)
    return pd.DataFrame(top_skills, columns=['Skill', 'Count'])

# --- Hàm vẽ phân phối mức lương theo danh mục ---
def plot_salary_by_category(df, chart_type='Box'):
    filtered_df = df[['primary_category', 'min_salary_mil_vnd']].dropna()
    top_categories = filtered_df['primary_category'].value_counts().index[:10]
    filtered_df = filtered_df[filtered_df['primary_category'].isin(top_categories)]
    
    if chart_type == 'Box':
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(x='min_salary_mil_vnd', y='primary_category', data=filtered_df, palette=PALETTE, ax=ax)
        ax.set_title('Phân phối Mức lương Tối thiểu theo Danh mục Công việc', fontsize=14, pad=15)
        ax.set_xlabel('Mức lương Tối thiểu (Triệu VND)', fontsize=12)
        ax.set_ylabel('Danh mục Công việc', fontsize=12)
        plt.tight_layout()
    else:
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()
        for idx, category in enumerate(top_categories):
            if idx < len(axes):
                sns.histplot(filtered_df[filtered_df['primary_category'] == category]['min_salary_mil_vnd'], 
                             kde=True, ax=axes[idx], color=PALETTE[idx % len(PALETTE)])
                axes[idx].set_title(f'{category}', fontsize=10)
                axes[idx].set_xlabel('Mức lương (Triệu VND)')
                axes[idx].set_ylabel('Số lượng')
        for idx in range(len(top_categories), len(axes)):
            axes[idx].axis('off')
        plt.suptitle('Phân phối Mức lương Tối thiểu theo Danh mục', fontsize=14, y=1.05)
        plt.tight_layout()
    
    return fig

# --- Hàm vẽ ảnh hưởng của kinh nghiệm đến mức lương ---
def plot_experience_salary(df, chart_type='Box'):
    filtered_df = df[['min_experience_years', 'min_salary_mil_vnd']].dropna()
    filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()
    filtered_df = filtered_df[filtered_df['min_experience_years'] <= 5]
    
    if chart_type == 'Box':
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='min_experience_years', y='min_salary_mil_vnd', data=filtered_df, palette="coolwarm", ax=ax)
        ax.set_title('Phân phối Mức lương Tối thiểu theo Kinh nghiệm (0-5 Năm)', fontsize=14, pad=15)
        ax.set_xlabel('Kinh nghiệm Tối thiểu (Năm)', fontsize=12)
        ax.set_ylabel('Mức lương Tối thiểu (Triệu VND)', fontsize=12)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='min_experience_years', y='min_salary_mil_vnd', data=filtered_df, 
                        color=PALETTE[0], alpha=0.5, ax=ax)
        ax.set_title('Mức lương Tối thiểu theo Kinh nghiệm (0-5 Năm)', fontsize=14, pad=15)
        ax.set_xlabel('Kinh nghiệm Tối thiểu (Năm)', fontsize=12)
        ax.set_ylabel('Mức lương Tối thiểu (Triệu VND)', fontsize=12)
        plt.tight_layout()
    
    return fig

# --- Tải dữ liệu cố định ---
DATA_FILE = "Data/cleaned_vietnamese_job_posting.csv"
try:
    df = load_and_preprocess_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"🚨 Không tìm thấy file dữ liệu: {DATA_FILE}. Vui lòng kiểm tra đường dẫn và thử lại.")
    st.stop()

# --- CSS ---
st.markdown("""
    <style>
    div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        font-size: 18px;
        font-weight: bold;
        color: #333;
        padding: 10px 15px;
        border-radius: 8px;
        transition: background-color 0.3s ease, transform 0.2s ease;
        cursor: pointer;
    }
    div[role="radiogroup"] label span {
        visibility: hidden;
        width: 0;
        margin: 0;
        padding: 0;
    }
    div[role="radiogroup"] label:hover {
        background-color: #FF851B;
        color: white !important;
        transform: scale(1.02);
    }
    div[role="radiogroup"] label[data-selected="true"] {
        background-color: #FF851B !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💼 Bảng Điều Khiển Phân Tích Tuyển Dụng Việt Nam")
page = st.sidebar.radio("Chọn Trang", [
    "1. Giới Thiệu Dữ Liệu",
    "2. Thống Kê Mô Tả",
    "3. Phân Tích Chuyên Sâu",
    "4. Nhận Xét Chung",
], index=0)

# --- Trang 1: Giới Thiệu Dữ Liệu ---
if page == "1. Giới Thiệu Dữ Liệu":
    st.header("📊 1. Giới Thiệu Dữ Liệu")

    st.markdown("""
        <style>
        .highlight {
            color: #FF851B;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📜 Nguồn Gốc Dữ Liệu", "📋 Mô Tả Dữ Liệu"])

    with tab1:
        st.subheader("Nguồn Gốc Dữ Liệu")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Số Bản Ghi Ban Đầu", value="2056", delta=None)
        with col2:
            st.metric(label="Số Bản Ghi Sau Xử Lý", value="1945", delta="-111", delta_color="inverse")
        with col3:
            st.metric(label="Thời Gian Thu Thập", value="02/2023 - 03/2023", delta=None)

        st.markdown("""
        - Dữ liệu được thu thập từ các bài đăng tuyển dụng trên trang web <span class="highlight">CareerBuilder.vn</span>.
        - Dữ liệu đã được làm sạch để loại bỏ HTML, các cụm từ không mong muốn, và xử lý các cột như mức lương, kinh nghiệm, và địa điểm.
        """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Mô Tả Dữ Liệu")

        with st.expander("📝 Xem Chi Tiết Các Cột Dữ Liệu"):
            st.markdown("""
            - **job_title:** Tên vị trí tuyển dụng.
            - **job_id:** Mã định danh duy nhất cho bài đăng.
            - **company_title:** Tên công ty tuyển dụng.
            - **salary:** Mức lương (đã xử lý thành min_salary_mil_vnd và max_salary_mil_vnd, đơn vị triệu VND).
            - **location:** Địa điểm làm việc (đã xử lý thành primary_location).
            - **outstanding_welfare:** Quyền lợi nổi bật.
            - **category:** Danh mục công việc (đã xử lý thành primary_category).
            - **position:** Cấp bậc vị trí.
            - **exp:** Kinh nghiệm yêu cầu (đã xử lý thành min_experience_years và max_experience_years).
            - **order:** Loại hợp đồng.
            - **detailed_welfare:** Quyền lợi chi tiết.
            - **job_description:** Mô tả công việc.
            - **job_requirements:** Yêu cầu công việc.
            - **job_tags:** Từ khóa liên quan đến công việc.
            - **primary_location:** Địa điểm làm việc chính.
            - **primary_category:** Danh mục công việc chính.
            """)

        buffer = BytesIO()
        with PdfPages(buffer) as pdf:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Cấu trúc dữ liệu:\n\n" + 
                    "job_title: Tên vị trí\n" +
                    "job_id: Mã định danh\n" +
                    "company_title: Tên công ty\n" +
                    "... (xem chi tiết trong dashboard)", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        buffer.seek(0)
        st.download_button(
            label="📥 Tải Xuống Cấu Trúc Dữ Liệu (PDF)",
            data=buffer,
            file_name="cau_truc_du_lieu.pdf",
            mime="application/pdf"
        )

    st.subheader("🗂 Xem Dữ Liệu Đã Xử Lý")
    st.dataframe(df.style.set_properties(**{
        'background-color': '#f9f9f9',
        'border': '1px solid #e0e0e0',
        'padding': '5px',
        'text-align': 'left'
    }), height=300)

# --- Trang 2: Thống Kê Mô Tả ---
elif page == "2. Thống Kê Mô Tả":
    st.header("📈 2. Thống Kê Mô Tả")

    st.markdown("""
        <style>
        .section-title {
            font-size: 24px;
            font-weight: 500;
            margin-bottom: 15px;
            color: #333333;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            color: #FF851B;
            margin-bottom: 10px;
        }
        .debug-expander {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Thông Tin Dữ Liệu", "📈 Phân Phối Biến Số", "📉 Phân Phối Biến Phân Loại"])

    with tab1:
        st.markdown('<div class="section-title">Thông Tin Dữ Liệu</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include='all').style.set_properties(**{
            'background-color': '#ffffff',
            'border': '1px solid #e0e0e0',
            'padding': '5px',
            'text-align': 'left'
        }), height=300)

    with tab2:
        st.markdown('<div class="section-title">Phân Phối Biến Số</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_select = st.selectbox("Chọn một biến số để xem phân phối:", numeric_cols)
        if num_select:
            chart_type = st.radio("Chọn loại biểu đồ:", ["Histogram (với KDE)", "Boxplot", "KDE Plot"])
            st.markdown('<div class="chart-title">Biểu Đồ Phân Phối</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(14, 7))

            if chart_type == "Histogram (với KDE)":
                sns.histplot(df[num_select].dropna(), kde=True, color=PALETTE[0], ax=ax)
                ax.set_title(f"Phân bố của {num_select} (Histogram với KDE)", fontsize=14, pad=15)
            elif chart_type == "Boxplot":
                sns.boxplot(y=df[num_select].dropna(), color=PALETTE[1], ax=ax)
                ax.set_title(f"Phân bố của {num_select} (Boxplot)", fontsize=14, pad=15)
            elif chart_type == "KDE Plot":
                sns.kdeplot(df[num_select].dropna(), color=PALETTE[2], fill=True, ax=ax)
                ax.set_title(f"Phân bố của {num_select} (KDE Plot)", fontsize=14, pad=15)

            ax.set_xlabel(num_select, fontsize=12)
            ax.set_ylabel('Số lượng' if chart_type != "Boxplot" else '', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name=f"PhanBo_{num_select}_{chart_type}.pdf",
                mime="application/pdf"
            )


    with tab3:
        st.markdown('<div class="section-title">Phân Phối Biến Phân Loại</div>', unsafe_allow_html=True)

        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        unique_locations = df['primary_location'].dropna().unique().tolist()
        selected_locations = st.multiselect(
            "📍 Lọc theo địa điểm:",
            options=unique_locations,
            default=unique_locations,
            key="filter_locations_categorical"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = df.copy()
        if selected_locations:
            filtered_df = filtered_df[filtered_df['primary_location'].isin(selected_locations)]

        if filtered_df.empty:
            st.warning("Không có dữ liệu hợp lệ sau khi lọc. Vui lòng chọn lại địa điểm.")
        else:
            cat_cols = ['order', 'position', 'primary_category']
            cat_select = st.selectbox("Chọn một biến phân loại để xem phân phối:", cat_cols)

            if cat_select:
                grouped = filtered_df.groupby('primary_location')[cat_select].value_counts().unstack(fill_value=0)
                grouped = grouped.reindex(selected_locations)
                grouped = grouped.sort_index(ascending=False)
                
                fig, ax = plt.subplots(figsize=(20, 16))
                grouped.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
                ax.set_title(f"Phân Phối '{cat_select}' Theo Địa Điểm Đã Chọn", fontsize=16, pad=10)
                ax.set_xlabel("Địa điểm", fontsize=14)
                ax.set_ylabel("Số lượng", fontsize=14)
                
                plt.xticks(rotation=90, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

                pdf_buffer = save_fig_to_pdf(fig)
                st.download_button(
                    label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                    data=pdf_buffer,
                    file_name=f"PhanBo_{cat_select}_TheoDiaDiem.pdf",
                    mime="application/pdf"
                )

# --- Trang 3: Phân Tích Chuyên Sâu ---
elif page == "3. Phân Tích Chuyên Sâu":
    st.header("🔍 3. Phân Tích Chuyên Sâu")

    st.markdown("""
        <style>
        .section-title {
            font-size: 24px;
            font-weight: 500;
            margin-bottom: 15px;
            color: #333333;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            color: #FF851B;
            margin-bottom: 10px;
        }
        .filter-box {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Thống Kê Chung",
        "📈 Số lượng tin tuyển dụng theo danh mục",
        "💰 Mức Lương Theo Danh Mục",
        "🕒 Kinh Nghiệm & Mức Lương",
        "📍 Phân Bố Địa Điểm",
        "🔗 Tương Quan",
        "📈 Phân Tích Song Biến",
        "📈 Xu Hướng Theo Thời Gian",
        "🌟 Dự đoán",
        "🤖 Phân Tích AI với Google Gemini"
    ])

    with tab1:
        total_jobs = len(df)
        avg_min_salary = df["min_salary_mil_vnd"].mean()
        avg_max_salary = df["max_salary_mil_vnd"].mean()
        top_location = df["primary_location"].mode()[0] if not df["primary_location"].empty else "N/A"
        top_category = df["primary_category"].mode()[0] if not df["primary_category"].empty else "N/A"
        location_percentage = (df["primary_location"] == top_location).mean() * 100
        total_categories = df["primary_category"].nunique()
        total_locations = df["primary_location"].nunique()
        max_min_salary = df["min_salary_mil_vnd"].max()
        max_max_salary = df["max_salary_mil_vnd"].max()
        highest_salary_category = df.groupby("primary_category")["min_salary_mil_vnd"].mean().idxmax()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng Số Bài Đăng", total_jobs)
        with col2:
            st.metric("Mức Lương Tối Thiểu TB", f"{avg_min_salary:.2f} triệu" if not pd.isna(avg_min_salary) else "N/A")
        with col3:
            st.metric("Mức Lương Tối Đa TB", f"{avg_max_salary:.2f} triệu" if not pd.isna(avg_max_salary) else "N/A")
        with col4:
            st.metric("Địa Điểm Phổ Biến Nhất", top_location)

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Danh Mục Phổ Biến Nhất", top_category)
        with col6:
            st.metric("Tổng Số Danh Mục", total_categories)
        with col7:
            st.metric("Tổng Số Địa Điểm", total_locations)
        with col8:
            st.metric("Địa Điểm Phổ Biến (%)", f"{location_percentage:.2f}%")

        col9, col10 = st.columns(2)
        with col9:
            st.metric("Mức Lương Tối Thiểu Cao Nhất", f"{max_min_salary:.2f} triệu" if not pd.isna(max_min_salary) else "N/A")
        with col10:
            st.metric("Danh Mục Lương Cao Nhất", highest_salary_category)
    
    with tab2:
        st.markdown("##### Số lượng tin tuyển dụng theo ngành")
        # Calculate the count of job postings for each category
        category_counts = df['primary_category'].value_counts().reset_index()
        category_counts.columns = ['Ngành nghề', 'Số lượng tin']

        top_n_cat = st.slider("Chọn Top N ngành nghề:", 5, min(30, len(category_counts)), 15, key='slider_cat_count_market')
        if not category_counts.empty:
            fig_cat_bar = px.bar(category_counts.head(top_n_cat), x='Số lượng tin', y='Ngành nghề', orientation='h', title=f'Top {top_n_cat} Ngành Nghề Nhiều Tin Nhất', text_auto=True, color='Số lượng tin', color_continuous_scale=px.colors.sequential.Blues)
            fig_cat_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(400, top_n_cat*25), showlegend=False, uniformtext_minsize=8, uniformtext_mode='hide')
            fig_cat_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_cat_bar, use_container_width=True)
        else:
            st.info("Không có dữ liệu ngành nghề.")
            
        # Biểu đồ Boxplot: Mối quan hệ giữa số lượng bài đăng và kinh nghiệm
        st.markdown('<div class="chart-title">Mối Quan Hệ Giữa Số Lượng Bài Đăng và Kinh Nghiệm</div>', unsafe_allow_html=True)
        job_counts = df.groupby('min_experience_years')['job_title'].count().reset_index()
        job_counts.columns = ['min_experience_years', 'job_count']
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            x="min_experience_years",
            y="job_count",
            data=job_counts,
            palette="viridis",
            ax=ax
        )
        ax.set_ylabel("Số Lượng Bài Đăng", fontsize=12)
        ax.set_xlabel("Số Năm Kinh Nghiệm Tối Thiểu", fontsize=12)
        ax.set_title("Mối Quan Hệ Giữa Số Lượng Bài Đăng và Kinh Nghiệm", fontsize=14, pad=15)
        plt.tight_layout()
        st.pyplot(fig)

        pdf_buffer = save_fig_to_pdf(fig)
        st.download_button(
            label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
            data=pdf_buffer,
            file_name="SoLuongBaiDang_vs_KinhNghiem_Boxplot.pdf",
            mime="application/pdf"
        )
        
    with tab3:
        st.markdown('<div class="section-title">Mức Lương Theo Danh Mục Công Việc</div>', unsafe_allow_html=True)

        # Bộ lọc danh mục công việc
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        all_categories = df['primary_category'].dropna().unique().tolist()
        selected_categories = st.multiselect(
            "Chọn các danh mục công việc (bỏ trống để chọn tất cả):",
            options=all_categories,
            default=all_categories,
            key="filter_categories_salary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Lọc dữ liệu theo danh mục công việc
        if selected_categories:
            filtered_df = df[df['primary_category'].isin(selected_categories)]
        else:
            filtered_df = df.copy()  # Nếu không chọn danh mục, sử dụng toàn bộ dữ liệu

        if filtered_df.empty:
            st.warning("Không có dữ liệu hợp lệ sau khi lọc. Vui lòng chọn lại danh mục.")
        else:
            st.markdown('<div class="chart-title">Mức Lương Tối Thiểu Theo Danh Mục Công Việc (Bar Chart)</div>', unsafe_allow_html=True)
            grouped_min_salary = (
                filtered_df
                .groupby("primary_category", as_index=False)["min_salary_mil_vnd"]
                .mean()
                .sort_values("min_salary_mil_vnd", ascending=False)
            )

            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(
                x="primary_category",
                y="min_salary_mil_vnd",
                data=grouped_min_salary,
                palette="viridis",
                ax=ax,
                order=selected_categories if selected_categories else grouped_min_salary['primary_category'].tolist()
            )
            ax.set_ylabel("Mức Lương Tối Thiểu Trung Bình (triệu VND)", fontsize=12)
            ax.set_xlabel("Danh Mục Công Việc", fontsize=12)
            ax.set_title(f"Mức Lương Tối Thiểu", fontsize=14, pad=15)
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig)

            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name="Luong_ToiThieu_TheoDanhMuc.pdf",
                mime="application/pdf"
            )

            st.markdown('<div class="chart-title">Mức Lương Tối Đa Theo Danh Mục Công Việc (Bar Chart)</div>', unsafe_allow_html=True)
            grouped_max_salary = (
                filtered_df
                .groupby("primary_category", as_index=False)["max_salary_mil_vnd"]
                .mean()
                .sort_values("max_salary_mil_vnd", ascending=False)
            )

            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(
                x="primary_category",
                y="max_salary_mil_vnd",
                data=grouped_max_salary,
                palette="viridis",
                ax=ax,
                order=selected_categories if selected_categories else grouped_max_salary['primary_category'].tolist()
            )
            ax.set_ylabel("Mức Lương Tối Đa Trung Bình (triệu VND)", fontsize=12)
            ax.set_xlabel("Danh Mục Công Việc", fontsize=12)
            ax.set_title(f"Mức Lương Tối Đa", fontsize=14, pad=15)
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig)

            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name="Luong_ToiDa_TheoDanhMuc.pdf",
                mime="application/pdf"
            )
    with tab4:
        st.markdown('<div class="section-title">Kinh Nghiệm & Mức Lương</div>', unsafe_allow_html=True)

        filtered_df = df[['min_experience_years', 'min_salary_mil_vnd', 'max_salary_mil_vnd']].dropna()
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()
        filtered_df = filtered_df[filtered_df['min_experience_years'] <= 5]

        if filtered_df.empty:
            st.warning("Không có dữ liệu hợp lệ cho kinh nghiệm từ 0-5 năm.")
        else:
            st.markdown('<div class="chart-title">Mức Lương Tối Thiểu Theo Kinh Nghiệm</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.boxplot(
                x="min_experience_years",
                y="min_salary_mil_vnd",
                data=filtered_df,
                palette="coolwarm",
                ax=ax
            )
            ax.set_title("Phân phối Mức lương Tối thiểu theo Kinh nghiệm Tối thiểu (0-5 Năm)", fontsize=16)
            ax.set_xlabel("Kinh nghiệm Tối thiểu (Năm)", fontsize=12)
            ax.set_ylabel("Mức lương Tối thiểu (Triệu VND)", fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)

            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name="Luong_ToiThieu_TheoKinhNghiem.pdf",
                mime="application/pdf"
            )
            
            st.markdown('<div class="chart-title">Mức Lương Tối Đa Theo Kinh Nghiệm</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.boxplot(
                x="min_experience_years",
                y="max_salary_mil_vnd",
                data=filtered_df,
                palette="coolwarm",
                ax=ax
            )
            ax.set_title("Phân phối Mức lương Tối đa theo Kinh nghiệm Tối thiểu (0-5 Năm)", fontsize=16)
            ax.set_xlabel("Kinh nghiệm Tối thiểu (Năm)", fontsize=12)
            ax.set_ylabel("Mức lương Tối đa (Triệu VND)", fontsize=12)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            plt.tight_layout()
            st.pyplot(fig)

            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name="Luong_ToiDa_TheoKinhNghiem.pdf",
                mime="application/pdf"
            )


    with tab5:
        st.markdown('<div class="section-title">Phân Bố Địa Điểm Tuyển Dụng</div>', unsafe_allow_html=True)

        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        locations = df['primary_location'].dropna().unique().tolist()
        selected_locations = st.multiselect(
            "Chọn các địa điểm (bỏ trống để chọn tất cả):",
            options=locations,
            default=locations,
            key="filter_locations"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = df.copy()
        if selected_locations:
            filtered_df = filtered_df[filtered_df['primary_location'].isin(selected_locations)]

        if filtered_df.empty:
            st.warning("Không có dữ liệu hợp lệ sau khi lọc. Vui lòng chọn lại địa điểm.")
        else:
            st.markdown('<div class="chart-title">Biểu Đồ Phân Bố</div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(20, 16))
            y_axis_value = st.selectbox(
                "Chọn giá trị trên trục X:",
                ["Số lượng bài đăng", "Mức lương tối thiểu trung bình", "Mức lương tối đa trung bình"],
                key="y_axis_location"
            )

            if y_axis_value == "Số lượng bài đăng":
                counts = (
                    filtered_df['primary_location']
                    .value_counts()
                    .rename_axis('primary_location')
                    .reset_index(name='count')
                )
                counts = counts[counts['primary_location'].isin(selected_locations)]
                # Sắp xếp theo số lượng bài đăng từ cao đến thấp
                counts = counts.sort_values('count', ascending=False)
                # Sử dụng order để đảm bảo thứ tự đúng
                sns.barplot(x='count', y='primary_location', data=counts, palette="viridis", ax=ax, order=counts['primary_location'])
                ax.set_xlabel('Số Lượng Bài Đăng', fontsize=12)
                ax.set_ylabel('Địa Điểm', fontsize=12)
            else:
                if y_axis_value == "Mức lương tối thiểu trung bình":
                    grouped = filtered_df.groupby("primary_location", as_index=False)["min_salary_mil_vnd"].mean()
                    x_col = "min_salary_mil_vnd"
                    x_label = "Mức Lương Tối Thiểu Trung Bình (triệu VND)"
                else:
                    grouped = filtered_df.groupby("primary_location", as_index=False)["max_salary_mil_vnd"].mean()
                    x_col = "max_salary_mil_vnd"
                    x_label = "Mức Lương Tối Đa Trung Bình (triệu VND)"

                grouped = grouped[grouped['primary_location'].isin(selected_locations)]
                # Sắp xếp theo mức lương từ cao đến thấp
                grouped = grouped.sort_values(x_col, ascending=False)
                # Sử dụng order để đảm bảo thứ tự đúng
                sns.barplot(x=x_col, y='primary_location', data=grouped, palette="viridis", ax=ax, order=grouped['primary_location'])
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel('Địa Điểm', fontsize=12)

            ax.set_title("Phân Bố Theo Địa Điểm Tuyển Dụng", fontsize=14, pad=15)
            plt.tight_layout()
            st.pyplot(fig)

            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name="PhanBo_DiaDiem.pdf",
                mime="application/pdf"
            )

    with tab6:
        st.markdown('<div class="section-title">Biểu Đồ Tương Quan (Heatmap)</div>', unsafe_allow_html=True)
        
    
        st.markdown('<div class="chart-title">Biểu Đồ Nhiệt</div>', unsafe_allow_html=True)
        
        corr_cols = ['min_salary_mil_vnd', 'max_salary_mil_vnd', 'min_experience_years', 'max_experience_years']
        corr_df = filtered_df[corr_cols].dropna()
        
        if corr_df.empty:
            st.warning("Không có dữ liệu hợp lệ để tính tương quan. Vui lòng kiểm tra dữ liệu sau khi lọc danh mục.")
        else:
            corr_matrix = corr_df.corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                vmin=-1, vmax=1,
                ax=ax,
                square=True
            )
            ax.set_title("Tương Quan Giữa Các Biến Số", fontsize=14, pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            
            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name="TuongQuan.pdf",
                mime="application/pdf"
            )


    with tab7:
        st.markdown('<div class="section-title">Phân Tích Song Biến</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        categories = df['primary_category'].unique().tolist()
        selected_categories = st.multiselect(
            "Chọn các danh mục công việc (bỏ trống để chọn tất cả):",
            options=categories,
            default=categories,
            key="filter_categories_bivariate"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = df.copy()
        if selected_categories:
            filtered_df = filtered_df[filtered_df['primary_category'].isin(selected_categories)]

        if filtered_df.empty:
            st.warning("Không có dữ liệu hợp lệ sau khi lọc. Vui lòng chọn lại danh mục.")
        else:
            num_col = filtered_df.select_dtypes(include=['float64']).columns.tolist()
            feature_x = st.selectbox("Chọn biến X:", num_col)
            feature_y = st.selectbox("Chọn biến Y:", num_col)
            plot_type = st.radio("Chọn loại biểu đồ:", ["Scatter", "2D KDE"])

            if feature_x != feature_y:
                st.markdown('<div class="chart-title">Biểu Đồ Phân Tích</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(12, 7))
                
                if plot_type == "Scatter":
                    sns.scatterplot(
                        x=filtered_df[feature_x], 
                        y=filtered_df[feature_y], 
                        hue=filtered_df["primary_category"], 
                        palette=PALETTE[:len(selected_categories)],
                        ax=ax
                    )
                elif plot_type == "2D KDE":
                    sns.kdeplot(
                        x=filtered_df[feature_x], 
                        y=filtered_df[feature_y], 
                        cmap="Blues", 
                        fill=True, 
                        ax=ax
                    )
                
                ax.set_title(f"{feature_x} vs {feature_y} ({plot_type})", fontsize=14, pad=15)
                ax.set_xlabel(feature_x, fontsize=12)
                ax.set_ylabel(feature_y, fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                
                pdf_buffer = save_fig_to_pdf(fig)
                st.download_button(
                    label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                    data=pdf_buffer,
                    file_name=f"{feature_x}_vs_{feature_y}_{plot_type}.pdf",
                    mime="application/pdf"
                )

    with tab8:
        st.markdown('<div class="section-title">Biểu Đồ Xu Hướng</div>', unsafe_allow_html=True)

        grouped = df.groupby("min_experience_years")[["min_salary_mil_vnd", "max_salary_mil_vnd"]].mean().reset_index()

        if grouped.empty:
            st.warning("Không có dữ liệu hợp lệ để hiển thị biểu đồ xu hướng.")
        else:
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.lineplot(x="min_experience_years", y="min_salary_mil_vnd", data=grouped, label="Mức Lương Tối Thiểu", ax=ax, color=PALETTE[0])
            sns.lineplot(x="min_experience_years", y="max_salary_mil_vnd", data=grouped, label="Mức Lương Tối Đa", ax=ax, color=PALETTE[1])
            ax.set_title("Mức Lương Theo Kinh Nghiệm (Biểu Đồ Đường)", fontsize=14, pad=15)
            ax.set_xlabel("Số Năm Kinh Nghiệm Tối Thiểu", fontsize=12)
            ax.set_ylabel("Mức Lương (Triệu VND)", fontsize=12)
            ax.legend(title="Loại Lương")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name="LineChart_MucLuong_KinhNghiem.pdf",
                mime="application/pdf"
            )

    with tab9:
        st.markdown('<div class="section-title">Phân Tích AI</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        categories = df['primary_category'].unique().tolist()
        selected_categories = st.multiselect(
            "Chọn các danh mục công việc (bỏ trống để chọn tất cả):",
            options=categories,
            default=categories,
            key="filter_categories_ai"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = df.copy()
        if selected_categories:
            filtered_df = filtered_df[filtered_df['primary_category'].isin(selected_categories)]

        if filtered_df.empty:
            st.warning("Không có dữ liệu hợp lệ sau khi lọc. Vui lòng chọn lại danh mục.")
        else:
            # Phân tích dự đoán lương
            st.markdown('<div class="chart-title">Dự Đoán Mức Lương</div>', unsafe_allow_html=True)
            salary_model, results, X_test = predict_salary(filtered_df)
            
            if salary_model is None:
                st.warning("Không đủ dữ liệu để huấn luyện mô hình dự đoán lương.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x='Actual', y='Predicted', data=results, ax=ax, color=PALETTE[0], alpha=0.5)
                ax.plot([results['Actual'].min(), results['Actual'].max()], 
                        [results['Actual'].min(), results['Actual'].max()], 
                        'r--', lw=2, label='Đường lý tưởng')
                ax.set_title('Dự Đoán Mức Lương Tối Thiểu (Hồi Quy Tuyến Tính)', fontsize=14, pad=15)
                ax.set_xlabel('Mức Lương Thực Tế (Triệu VND)', fontsize=12)
                ax.set_ylabel('Mức Lương Dự Đoán (Triệu VND)', fontsize=12)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                
                pdf_buffer = save_fig_to_pdf(fig)
                st.download_button(
                    label="📥 Lưu Biểu Đồ Dự Đoán Lương (PDF)",
                    data=pdf_buffer,
                    file_name="Salary_Prediction.pdf",
                    mime="application/pdf"
                )

                # Tùy chọn nhập liệu để dự đoán lương
                st.subheader("Dự Đoán Lương Cá Nhân")
                exp_years = st.number_input("Số năm kinh nghiệm:", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
                category = st.selectbox("Danh mục công việc:", categories)
                location = st.selectbox("Địa điểm:", df['primary_location'].unique().tolist())
                
                if st.button("Dự đoán"):
                    input_data = pd.DataFrame({
                        'min_experience_years': [exp_years],
                        'primary_category': [category],
                        'primary_location': [location]
                    })
                    pred_salary = salary_model.predict(input_data)[0]
                    st.success(f"Mức lương tối thiểu dự đoán: {pred_salary:.2f} triệu VND")

            # Phân tích kỹ năng
            st.markdown('<div class="chart-title">Top 10 Kỹ Năng Phổ Biến</div>', unsafe_allow_html=True)
            skills_df = extract_skills(filtered_df)
            
            if skills_df.empty:
                st.warning("Không tìm thấy kỹ năng nào trong yêu cầu công việc.")
            else:
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.barplot(x='Count', y='Skill', data=skills_df, palette="viridis", ax=ax)
                ax.set_title('Top 10 Kỹ Năng Phổ Biến Trong Yêu Cầu Công Việc', fontsize=14, pad=15)
                ax.set_xlabel('Số Lượng Xuất Hiện', fontsize=12)
                ax.set_ylabel('Kỹ Năng', fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                
                pdf_buffer = save_fig_to_pdf(fig)
                st.download_button(
                    label="📥 Lưu Biểu Đồ Kỹ Năng (PDF)",
                    data=pdf_buffer,
                    file_name="Top_Skills.pdf",
                    mime="application/pdf"
                )

    with tab10:
        if not GEMINI_API_KEY:
            st.error("🚨 GEMINI_API_KEY environment variable not found. Please set it in your .env file.")
            st.stop()
        else:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                gemini_model = genai.GenerativeModel(MODEL_NAME)
            except Exception as e:
                st.error(f"Error configuring Google AI or creating model: {e}")
                st.stop()
        st.markdown('<div class="section-title">Phân Tích AI với Google Gemini</div>', unsafe_allow_html=True)
        
        # Hiển thị tóm tắt dữ liệu
        st.subheader("Tóm tắt dữ liệu")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Xem trước dữ liệu**")
            st.dataframe(df.head())
        with col2:
            st.markdown("**Thông tin dữ liệu**")
            st.write(f"**Số dòng:** {df.shape[0]}")
            st.write(f"**Số cột:** {df.shape[1]}")
            st.write("**Các cột:** " + ", ".join(df.columns.tolist()))
            # Chuẩn bị dữ liệu cho AI
            buffer = StringIO()
            filtered_df.info(buf=buffer)
            info_str = buffer.getvalue()

            # Lấy mẫu dữ liệu (giới hạn 100 dòng để tránh vượt giới hạn token)
            max_rows_to_feed = 100
            data_summary = []
            for i, row in filtered_df.head(max_rows_to_feed).iterrows():
                row_data = [f"{col}: {row[col]}" for col in filtered_df.columns]
                formatted_row = f"Row {i+1}: {', '.join(row_data)}"
                data_summary.append(formatted_row)
            data_summary = "\n".join(data_summary)
            if len(filtered_df) > max_rows_to_feed:
                data_summary += f"\n... (Truncated to {max_rows_to_feed} rows out of {len(filtered_df)} total rows)"

            data_to_feed = f"""
            Dataset Info:
            {info_str}
            
            Dataset Rows (Sample):
            {data_summary}
            """

            # Giao diện nhập câu hỏi
    
            st.subheader("🤖 Hỏi Google Gemini về dữ liệu")
            user_query = st.text_area(
                "Đặt câu hỏi cho AI:",
                height=100,
                placeholder="Ví dụ: Các danh mục công việc phổ biến nhất là gì? Mức lương trung bình của ngành IT ở TP.HCM?"
            )

            if st.button("Tạo phản hồi", type="primary"):
                if user_query:
                    with st.spinner("Đang tạo phản hồi từ Google Gemini..."):
                        # Hàm generate_feedback
                        safety_settings = [
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        ]
                        generation_config = genai.types.GenerationConfig(
                            temperature=0.9,
                            max_output_tokens=1000,
                            top_p=0.9,
                            top_k=40,
                        )
                        prompt = f"""
                        Bạn là một chuyên gia phân tích dữ liệu và bạn có khả năng phân tích dữ liệu CSV.
                        Dưới đây là một tóm tắt về dữ liệu mà bạn sẽ phân tích:
                        {data_to_feed}
                        
                        Query của người dùng: {user_query}
                        
                        Bạn hãy phân tích dữ liệu và trả lời câu hỏi của người dùng một cách chi tiết và dễ hiểu.
                        Hãy cung cấp các thông tin hữu ích và có thể bao gồm các gợi ý biểu đồ nếu cần thiết.
                        """
                        try:
                            response = gemini_model.generate_content(
                                prompt,
                                generation_config=generation_config,
                                safety_settings=safety_settings
                                )
                            st.subheader("Phản hồi từ Google Gemini")
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Lỗi khi tạo phản hồi từ AI: {e}")
                else:
                    st.warning("Vui lòng nhập câu hỏi hoặc yêu cầu.")
                
        

# --- Trang 4: Nhận Xét Chung ---
elif page == "4. Nhận Xét Chung":
    st.header("4. Nhận Xét Chung")
    avg_min_salary = df["min_salary_mil_vnd"].mean()
    avg_experience = df["min_experience_years"].mean()
    top_location = df["primary_location"].mode()[0]
    top_category = df["primary_category"].mode()[0]
    location_percentage = (df["primary_location"] == top_location).mean() * 100
    st.markdown(f"""
    - **Tổng Quan về Dữ Liệu và Kết Quả Phân Tích:**
        - **Mức lương và Kinh nghiệm:** Mức lương tối thiểu trung bình là {avg_min_salary:.2f} triệu VND, với kinh nghiệm yêu cầu trung bình {avg_experience:.2f} năm.
        - **Phân bố Địa điểm:** {top_location} chiếm {location_percentage:.2f}% số bài đăng, cho thấy sự tập trung cơ hội việc làm ở các thành phố lớn.
        - **Danh mục Phổ biến:** {top_category} là danh mục công việc phổ biến nhất, phản ánh nhu cầu cao trong lĩnh vực này.
        - **Kinh nghiệm và Lương:** Có xu hướng mức lương tăng theo kinh nghiệm, nhưng sự biến động lớn do ảnh hưởng của danh mục công việc và địa điểm.
        - **Phân tích AI:** Mô hình dự đoán lương và phân tích kỹ năng giúp xác định các yếu tố ảnh hưởng đến lương và nhu cầu kỹ năng thị trường.
    """)
