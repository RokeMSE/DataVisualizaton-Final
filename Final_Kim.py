import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from collections import Counter
import re

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân Tích Tuyển Dụng Việt Nam", page_icon="💼", layout="wide", initial_sidebar_state="expanded")

# Thiết lập bảng màu
sns.set_palette("colorblind")
PALETTE = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

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

# --- Tải dữ liệu từ người dùng ---
st.sidebar.header("Tải Dữ Liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV", type=["csv"])
if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
else:
    st.warning("Vui lòng tải lên tệp CSV để tiếp tục.")
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

        with st.expander("💡 Nhận Xét Về Thống Kê Mô Tả"):
            st.markdown("""
            - **Mức lương (min_salary_mil_vnd, max_salary_mil_vnd):** Mức lương tối thiểu và tối đa trung bình, phạm vi lương, và phân phối (triệu VND).
            - **Kinh nghiệm (min_experience_years, max_experience_years):** Số năm kinh nghiệm yêu cầu trung bình và phân phối.
            - **Địa điểm (primary_location):** Các địa điểm phổ biến nhất (ví dụ: TP.HCM, Hà Nội).
            - **Danh mục (primary_category):** Các danh mục công việc phổ biến (ví dụ: IT, Kinh doanh).
            """)

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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Thống Kê Chung",
        "💰 Mức Lương Theo Danh Mục",
        "🕒 Kinh Nghiệm & Mức Lương",
        "📍 Phân Bố Địa Điểm",
        "🔗 Tương Quan",
        "📈 Phân Tích Song Biến",
        "📈 Xu Hướng Theo Thời Gian",
        "🤖 Phân Tích AI"
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
        st.markdown('<div class="section-title">Mức Lương Theo Danh Mục Công Việc</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        categories = df['primary_category'].unique().tolist()
        selected_categories = st.multiselect(
            "Chọn các danh mục công việc (bỏ trống để chọn tất cả):",
            options=categories,
            default=categories,
            key="filter_categories_salary"
        )
        min_salary_range = st.slider(
            "Chọn khoảng mức lương tối thiểu (triệu VND):",
            min_value=float(df['min_salary_mil_vnd'].min()),
            max_value=float(df['min_salary_mil_vnd'].max()),
            value=(float(df['min_salary_mil_vnd'].min()), float(df['min_salary_mil_vnd'].max())),
            step=1.0
        )

        y_axis_value = st.selectbox(
            "Chọn giá trị trên trục Y:",
            ["Số lượng bài đăng", "Mức lương tối thiểu trung bình", "Mức lương tối đa trung bình"],
            key="y_axis_salary"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = df.copy()
        if selected_categories:
            filtered_df = filtered_df[filtered_df['primary_category'].isin(selected_categories)]
        filtered_df = filtered_df[
            (filtered_df['min_salary_mil_vnd'] >= min_salary_range[0]) &
            (filtered_df['min_salary_mil_vnd'] <= min_salary_range[1])
        ]

        chart_type = st.radio("Chọn loại biểu đồ:", ["Box", "Histogram"])
        st.markdown('<div class="chart-title">Biểu Đồ Phân Tích</div>', unsafe_allow_html=True)
        
        if y_axis_value == "Số lượng bài đăng":
            if chart_type == "Box":
                fig, ax = plt.subplots(figsize=(14, 8))
                sns.boxplot(x="primary_category", y="min_salary_mil_vnd", data=filtered_df, ax=ax, order=selected_categories if selected_categories else categories)
                ax.set_ylabel("Mức Lương Tối Thiểu (triệu VND)", fontsize=12)
                ax.set_title("Mức Lương Theo Danh Mục Công Việc (Boxplot)", fontsize=14, pad=15)
                ax.set_xlabel("Danh Mục Công Việc", fontsize=12)
                plt.xticks(rotation=90)
                plt.tight_layout()
                st.pyplot(fig)
                pdf_buffer = save_fig_to_pdf(fig)
                st.download_button(
                    label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                    data=pdf_buffer,
                    file_name="Luong_Theo_DanhMuc_Box.pdf",
                    mime="application/pdf"
                )
            elif chart_type == "Histogram":
                unique_categories = filtered_df["primary_category"].unique()
                num_categories = len(unique_categories)
                if num_categories == 0:
                    st.warning("Không có danh mục nào để hiển thị sau khi lọc. Vui lòng kiểm tra bộ lọc.")
                else:
                    for i, category in enumerate(unique_categories):
                        st.markdown(f"**Histogram cho danh mục: {category}**", unsafe_allow_html=True)
                        fig, ax = plt.subplots(figsize=(14, 8))
                        sns.histplot(filtered_df[filtered_df["primary_category"] == category]["min_salary_mil_vnd"], kde=True, ax=ax, color=PALETTE[i % len(PALETTE)])
                        ax.set_title(f"Phân Bố Mức Lương Tối Thiểu - {category}", fontsize=14, pad=15)
                        ax.set_xlabel("Mức Lương Tối Thiểu (triệu VND)", fontsize=12)
                        ax.set_ylabel("Số Lượng Bài Đăng", fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig)
                        pdf_buffer = save_fig_to_pdf(fig)
                        st.download_button(
                            label=f"📥 Lưu Biểu Đồ Dưới Dạng PDF ({category})",
                            data=pdf_buffer,
                            file_name=f"Luong_Theo_DanhMuc_Histogram_{category}.pdf",
                            mime="application/pdf"
                        )
        else:
            if y_axis_value == "Mức lương tối thiểu trung bình":
                grouped = filtered_df.groupby("primary_category")["min_salary_mil_vnd"].mean().reset_index()
                y_col = "min_salary_mil_vnd"
                y_label = "Mức Lương Tối Thiểu Trung Bình (triệu VND)"
            else:
                grouped = filtered_df.groupby("primary_category")["max_salary_mil_vnd"].mean().reset_index()
                y_col = "max_salary_mil_vnd"
                y_label = "Mức Lương Tối Đa Trung Bình (triệu VND)"
            
            fig, ax = plt.subplots(figsize=(14, 8))
            if chart_type == "Box":
                sns.barplot(x="primary_category", y=y_col, data=grouped, ax=ax, order=selected_categories if selected_categories else categories)
            elif chart_type == "Histogram":
                sns.histplot(grouped[y_col], kde=True, ax=ax)
            ax.set_title("Mức Lương Theo Danh Mục Công Việc", fontsize=14, pad=15)
            ax.set_xlabel("Danh Mục Công Việc", fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            plt.xticks(rotation=90)
            plt.tight_layout()
            st.pyplot(fig)
            pdf_buffer = save_fig_to_pdf(fig)
            st.download_button(
                label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
                data=pdf_buffer,
                file_name=f"Luong_Theo_DanhMuc_{chart_type}.pdf",
                mime="application/pdf"
            )

        with st.expander("💡 Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            - **Biểu đồ Boxplot/Histogram:**
              - **Boxplot:** Thể hiện phân phối mức lương tối thiểu theo danh mục công việc. Trung vị (đường giữa hộp) cho thấy mức lương trung bình, IQR (độ cao hộp) cho thấy sự biến động, và các điểm ngoại lai cho thấy các mức lương bất thường.
              - **Histogram:** Thể hiện phân bố chi tiết của mức lương trong mỗi danh mục, với đường KDE để thấy xu hướng.
            - **Nhận xét:**
              - Các danh mục như IT, Tài chính có xu hướng có mức lương cao hơn.
              - Sự biến động lương (IQR) khác nhau giữa các danh mục, cho thấy mức độ đa dạng trong cơ hội lương.
              - Các điểm ngoại lai ở một số danh mục có thể là các vị trí cấp cao hoặc đặc thù.
            """)

    with tab3:
        st.markdown('<div class="section-title">Kinh Nghiệm & Mức Lương</div>', unsafe_allow_html=True)

        y_axis_value = st.selectbox(
            "Chọn giá trị trên trục Y:",
            ["Số lượng bài đăng", "Mức lương tối thiểu trung bình", "Mức lương tối đa trung bình"],
            key="y_axis_exp"
        )

        chart_type = st.radio("Chọn loại biểu đồ:", ["Box", "Scatter"])
        st.markdown('<div class="chart-title">Biểu Đồ Phân Tích</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        if y_axis_value == "Số lượng bài đăng":
            if chart_type == "Box":
                sns.boxplot(x="min_experience_years", y="min_salary_mil_vnd", data=df, ax=ax)
                ax.set_ylabel("Số lượng bài đăng", fontsize=12)
            elif chart_type == "Scatter":
                sns.scatterplot(x="min_experience_years", y="min_salary_mil_vnd", data=df, ax=ax)
                ax.set_ylabel("Số lượng bài đăng", fontsize=12)
        else:
            if y_axis_value == "Mức lương tối thiểu trung bình":
                grouped = df.groupby("min_experience_years")["min_salary_mil_vnd"].mean().reset_index()
                y_col = "min_salary_mil_vnd"
                y_label = "Mức Lương Tối Thiểu Trung Bình (triệu VND)"
            else:
                grouped = df.groupby("min_experience_years")["max_salary_mil_vnd"].mean().reset_index()
                y_col = "max_salary_mil_vnd"
                y_label = "Mức Lương Tối Đa Trung Bình (triệu VND)"

            if chart_type == "Box":
                sns.barplot(x="min_experience_years", y=y_col, data=grouped, ax=ax)
            elif chart_type == "Scatter":
                sns.scatterplot(x="min_experience_years", y=y_col, data=grouped, ax=ax)
            ax.set_ylabel(y_label, fontsize=12)

        ax.set_title("Kinh Nghiệm và Mức Lương", fontsize=14, pad=15)
        ax.set_xlabel("Số Năm Kinh Nghiệm Tối Thiểu", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        pdf_buffer = save_fig_to_pdf(fig)
        st.download_button(
            label="📥 Lưu Biểu Đồ Dưới Dạng PDF",
            data=pdf_buffer,
            file_name=f"KinhNghiem_Luong_{chart_type}.pdf",
            mime="application/pdf"
        )

        with st.expander("💡 Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            - **Biểu đồ Boxplot/Scatter:**
              - **Boxplot:** Thể hiện phân phối mức lương tối thiểu theo số năm kinh nghiệm (0-5 năm). Trung vị tăng theo kinh nghiệm, cho thấy kinh nghiệm cao hơn thường có lương cao hơn.
              - **Scatter:** Hiển thị mối quan hệ giữa kinh nghiệm và lương, với các điểm phân tán cho thấy sự đa dạng trong mức lương cho cùng mức kinh nghiệm.
            - **Nhận xét:**
              - Kinh nghiệm từ 0-1 năm có mức lương thấp nhất, với ít biến động.
              - Từ 2-5 năm, mức lương tăng đáng kể, nhưng cũng có nhiều điểm ngoại lai (các vị trí lương cao bất thường).
              - Mối quan hệ giữa kinh nghiệm và lương không hoàn toàn tuyến tính, do các yếu tố khác như danh mục công việc hoặc địa điểm.
            """)

    with tab4:
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
            st.markdown(f"Đang hiển thị dữ liệu cho: **{', '.join(selected_locations)}**")
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
                counts['primary_location'] = pd.Categorical(counts['primary_location'], categories=selected_locations, ordered=True)
                counts = counts.sort_values('primary_location')
                sns.barplot(x='count', y='primary_location', data=counts, palette="viridis", ax=ax)
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
                grouped['primary_location'] = pd.Categorical(grouped['primary_location'], categories=selected_locations, ordered=True)
                grouped = grouped.sort_values('primary_location')

                sns.barplot(x=x_col, y='primary_location', data=grouped, palette="viridis", ax=ax)
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

    with tab5:
        st.markdown('<div class="section-title">Biểu Đồ Tương Quan (Heatmap)</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        categories = df['primary_category'].unique().tolist()
        selected_categories = st.multiselect(
            "Chọn các danh mục công việc (bỏ trống để chọn tất cả):",
            options=categories,
            default=categories,
            key="filter_categories_corr"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        filtered_df = df.copy()
        if selected_categories:
            filtered_df = filtered_df[filtered_df['primary_category'].isin(selected_categories)]

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

        with st.expander("💡 Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            - **Biểu đồ Nhiệt Tương Quan:**
              - Thể hiện mức độ tương quan tuyến tính giữa các biến số: mức lương tối thiểu, tối đa, kinh nghiệm tối thiểu, tối đa.
              - Giá trị gần 1 hoặc -1 cho thấy tương quan mạnh; gần 0 cho thấy ít hoặc không tương quan.
            - **Nhận xét:**
              - **Mức lương tối thiểu và tối đa:** Thường có tương quan cao, vì các bài đăng có xu hướng xác định một khoảng lương rõ ràng.
              - **Kinh nghiệm và mức lương:** Có thể có tương quan dương nhẹ, nhưng không mạnh, do các yếu tố khác như danh mục công việc ảnh hưởng đến lương.
              - **Kinh nghiệm tối thiểu và tối đa:** Tương quan cao, vì nhiều bài đăng yêu cầu một khoảng kinh nghiệm cụ thể.
            """)

    with tab6:
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

    with tab7:
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

        with st.expander("💡 Xem Nhận Xét Chi Tiết"):
            st.markdown("""
            - **Biểu Đồ Đường:**
              - Thể hiện xu hướng mức lương tối thiểu và tối đa theo số năm kinh nghiệm.
              - Đường biểu diễn giúp dễ dàng nhận thấy sự thay đổi mức lương theo kinh nghiệm.
            - **Nhận xét:**
              - Mức lương tối thiểu và tối đa thường tăng theo số năm kinh nghiệm.
              - Sự khác biệt giữa mức lương tối thiểu và tối đa có thể phản ánh sự đa dạng trong các vị trí công việc.
            """)

    with tab8:
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
            model, results, X_test = predict_salary(filtered_df)
            
            if model is None:
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
                    pred_salary = model.predict(input_data)[0]
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

        with st.expander("💡 Nhận Xét Về Phân Tích AI"):
            st.markdown("""
            - **Dự Đoán Mức Lương:**
              - Mô hình hồi quy tuyến tính dự đoán mức lương tối thiểu dựa trên kinh nghiệm, danh mục công việc, và địa điểm.
              - Biểu đồ phân tán so sánh giá trị thực tế và dự đoán, với đường lý tưởng (y=x) để đánh giá độ chính xác.
              - Nhận xét: Mô hình có thể dự đoán gần đúng mức lương, nhưng độ chính xác phụ thuộc vào chất lượng dữ liệu và các yếu tố khác (như mô tả công việc).
            - **Phân Tích Kỹ Năng:**
              - Trích xuất các kỹ năng phổ biến từ yêu cầu công việc, hiển thị tần suất xuất hiện.
              - Nhận xét: Các kỹ năng như Python, SQL, hoặc quản lý thường xuất hiện nhiều trong các danh mục công nghệ và kinh doanh, phản ánh nhu cầu thị trường.
            """)

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