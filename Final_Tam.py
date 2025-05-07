# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from collections import Counter
import csv
from io import BytesIO, StringIO
import statsmodels.api as sm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- AI Config ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_MODEL_NAME = "gemini-2.0-flash"

genai_model = None
ai_configured_successfully = False

if not GEMINI_API_KEY:
    st.error("🚨 Không tìm thấy biến môi trường GEMINI_API_KEY. Vui lòng thiết lập trong file .env.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_model = genai.GenerativeModel(AI_MODEL_NAME)
        ai_configured_successfully = True
    except Exception as e:
        ai_configured_successfully = False

# --- Page Config ---
st.set_page_config(
    page_title="Dashboard Phân Tích Tuyển Dụng VN",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_palette("colorblind")
PALETTE = ["#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]

def save_fig_to_pdf(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='pdf', bbox_inches='tight')
    buffer.seek(0)
    return buffer

# --- Helper Functions ---
@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data(uploaded_file):
    """Tải và tiền xử lý dữ liệu từ file CSV được tải lên."""
    try:
        df = pd.read_csv(
            uploaded_file,
            encoding='utf-8',
            na_values=["None", " ", "UNKNOWN", -1, 999, "NA", "N/A", "NULL", ""],
        )
        st.session_state.original_filename = uploaded_file.name

        # Xử lý job_title
        if 'job_title' in df.columns:
            unique_types = df['job_title'].apply(type).unique()
            if len(unique_types) > 1:
                st.warning(f"Cột job_title chứa nhiều kiểu dữ liệu: {unique_types}. Đang chuẩn hóa thành chuỗi.")
            df['job_title'] = df['job_title'].astype(str).replace('nan', 'Không xác định').str.strip()
        else:
            df['job_title'] = 'Không xác định'

        # Xử lý các cột số
        salary_cols = ['min_salary_mil_vnd', 'max_salary_mil_vnd']
        for col in salary_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan

        exp_cols = ['min_experience_years', 'max_experience_years']
        for col in exp_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').clip(0, 50)
            else:
                df[col] = np.nan

        # Xử lý các cột phân loại
        categorical_cols = ['primary_location', 'primary_category', 'position', 'order']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Không xác định').str.strip()
                df[col] = df[col].astype('category')
            else:
                df[col] = pd.Categorical(['Không xác định'] * len(df))

        if 'job_requirements' in df.columns:
            df['job_requirements'] = df['job_requirements'].astype(str).fillna('')
        else:
            df['job_requirements'] = ''

        if 'company_title' not in df.columns:
            df['company_title'] = 'Không xác định'
        else:
            df['company_title'] = df['company_title'].astype(str).str.strip()

        # Chuyển các cột StringDtype thành object
        for col in df.columns:
            if df[col].dtype == 'string':
                df[col] = df[col].astype('object')

        # Chuyển đổi kiểu dữ liệu tối ưu
        df = df.convert_dtypes()

        # Kiểm tra kiểu dữ liệu
        # st.write("Kiểu dữ liệu của các cột sau khi xử lý:")
        # st.write(df.dtypes)

        if df.empty:
            st.error("Dữ liệu rỗng sau khi đọc file.")
            return None

        return df

    except Exception as e:
        st.error(f"Lỗi khi đọc hoặc xử lý file CSV: {e}")
        st.exception(e)
        return None

@st.cache_data
def extract_skills_from_requirements(text_series, num_skills=30):
    """Trích xuất và đếm tần suất các kỹ năng từ cột job_requirements."""
    skills_keywords = [
        'python', 'java', 'sql', 'javascript', 'react', 'angular', 'vue', 'node.js', 'nodejs',
        'c#', '.net', 'php', 'ruby', 'swift', 'kotlin', 'android', 'ios', 'flutter', 'html', 'css',
        'typescript', 'go', 'rust', 'scala', 'perl', 'c++', ' c ',
        'mysql', 'postgresql', 'sql server', 'mongodb', 'oracle', 'redis', 'cassandra', 'nosql',
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'terraform', 'linux', 'jenkins', 'git', 'ci/cd', 'devops', 'ansible',
        'excel', 'power bi', 'tableau', 'qlik', 'data analysis', 'phân tích dữ liệu', 'machine learning', 'học máy', 'ai', 'trí tuệ nhân tạo', 'deep learning', 'học sâu',
        'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras', 'hadoop', 'spark', 'big data', 'dữ liệu lớn', 'data warehouse', 'data mining',
        'project management', 'quản lý dự án', 'agile', 'scrum', 'communication', 'giao tiếp', 'leadership', 'lãnh đạo', 'teamwork', 'làm việc nhóm',
        'problem solving', 'giải quyết vấn đề', 'critical thinking', 'tư duy phản biện', 'english', 'tiếng anh', 'japanese', 'tiếng nhật', 'korean', 'tiếng hàn', 'chinese', 'tiếng trung',
        'marketing', 'digital marketing', 'sales', 'bán hàng', 'seo', 'sem', 'content', 'social media',
        'finance', 'tài chính', 'accounting', 'kế toán', 'auditing', 'kiểm toán', 'banking', 'ngân hàng',
        'hr', 'nhân sự', 'recruitment', 'tuyển dụng', 'talent acquisition', 'c&b', 'payroll',
        'customer service', 'chăm sóc khách hàng', 'support', 'hỗ trợ',
        'design', 'ui', 'ux', 'photoshop', 'illustrator', 'figma', 'sketch', 'graphic design',
        'autocad', 'revit', 'sap', 'erp', 'logistics', 'supply chain', 'chuỗi cung ứng', 'teaching', 'giảng dạy', 'research', 'nghiên cứu', 'writing', 'viết lách'
    ]
    text = ' '.join(text_series.astype(str).str.lower())
    skill_counts = Counter()
    for skill in skills_keywords:
        pattern = r'(?:^|\s|[.,;!?():\'"])' + re.escape(skill).replace(r'\.', r'[\.\s-]?').replace(r'\-', r'[\s-]?') + r'(?:$|\s|[.,;!?():\'"])'
        try:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            count = sum(1 for _ in matches)
            if count > 0:
                normalized_skill = skill.replace('\\', '').replace('.', '').replace('-', '').replace(' ', '')
                existing_keys = [k for k in skill_counts if k.replace('.', '').replace('-', '').replace(' ', '') == normalized_skill]
                if existing_keys:
                    chosen_key = min(existing_keys + [skill.replace('\\', '')], key=len)
                    skill_counts[chosen_key] += count
                    for other_key in existing_keys:
                        if other_key != chosen_key and other_key in skill_counts:
                            skill_counts[chosen_key] += skill_counts.pop(other_key)
                else:
                    skill_counts[skill.replace('\\', '')] += count
        except re.error:
            pass
    top_skills = skill_counts.most_common(num_skills)
    return pd.DataFrame(top_skills, columns=['Kỹ năng', 'Số lần xuất hiện']), text

@st.cache_resource
def train_salary_model(df_train):
    """Huấn luyện mô hình dự đoán lương (min_salary_mil_vnd)."""
    model_features = ['min_experience_years', 'primary_category', 'primary_location']
    target = 'min_salary_mil_vnd'
    X = df_train[model_features]
    y = df_train[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    categorical_features = ['primary_category', 'primary_location']
    numeric_features = ['min_experience_years']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numeric_features)
        ],
        remainder='drop'
    )
    model_choice = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5, oob_score=True)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model_choice)
    ])
    try:
        pipeline.fit(X_train, y_train)
        oob_score = pipeline.named_steps['regressor'].oob_score_ if hasattr(pipeline.named_steps['regressor'], 'oob_score_') else None
    except Exception as e:
        st.error(f"Lỗi khi huấn luyện mô hình: {e}")
        return None, None, None, None, None
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results_df = pd.DataFrame({'Thực tế': y_test, 'Dự đoán': y_pred})
    st.session_state.model_pipeline = pipeline
    return results_df, rmse, mae, r2, oob_score

def get_dataset_info(df):
    if df is None:
        return "Không có dữ liệu DataFrame để hiển thị thông tin."
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    info_summary = (
        f"Tổng quan Dataset:\n"
        f"--------------------\n"
        f"Số dòng: {df.shape[0]}\n"
        f"Số cột: {df.shape[1]}\n\n"
        f"Thông tin chi tiết các cột (dtypes, non-null counts):\n"
        f"{info_str}\n"
        f"Một vài thống kê mô tả cơ bản cho các cột số:\n"
        f"{df.describe(include=np.number).to_string()}\n\n"
        f"Một vài thống kê mô tả cơ bản cho các cột object/category:\n"
        f"{df.describe(include=['object', 'category']).to_string()}"
    )
    return info_summary

def generate_feedback(data_context, query):
    global genai_model
    if not genai_model:
        return "AI model chưa được cấu hình hoặc khởi tạo thành công."
    try:
        prompt = f"""
        Dựa trên dữ liệu tuyển dụng được cung cấp sau đây:
        <data_context>
        {data_context}
        </data_context>

        Hãy trả lời câu hỏi sau một cách chi tiết và hữu ích (bằng ngôn ngữ của user' input): "{query}"
        """
        response = genai_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Đã xảy ra lỗi khi giao tiếp với AI: {str(e)}"

@st.cache_data
def convert_df_to_csv(df):
    """Chuyển đổi DataFrame thành bytes CSV UTF-8."""
    return df.to_csv(index=False).encode('utf-8')

# --- Session State ---
if 'df_jobs' not in st.session_state:
    st.session_state.df_jobs = None
    st.session_state.df_filtered = None
    st.session_state.original_filename = ""
    st.session_state.filters = {}
    st.session_state.model_results = None
    st.session_state.model_metrics = {}
    st.session_state.model_pipeline = None

# --- Sidebar ---
st.sidebar.title("Bảng điều khiển")
uploaded_file = st.sidebar.file_uploader(
    "📁 Tải lên tệp CSV",
    type=["csv"],
    help="Tải lên file CSV chứa dữ liệu tuyển dụng đã được làm sạch (UTF-8)."
)

if uploaded_file is not None:
    if st.session_state.df_jobs is None or st.session_state.original_filename != uploaded_file.name:
        with st.spinner("Đang xử lý dữ liệu..."):
            st.session_state.df_jobs = load_data(uploaded_file)
            st.session_state.filters = {}
            st.session_state.model_results = None
            st.session_state.model_metrics = {}
            st.session_state.model_pipeline = None

if st.session_state.df_jobs is not None:
    df_jobs = st.session_state.df_jobs
    
    st.sidebar.header("Điều Hướng")
    page_options = [
        "🏠 Trang Chủ & Tổng Quan",
        "📊 Phân Tích Thị Trường",
        "💰 Phân Tích Lương & Kinh Nghiệm",
        "🛠️ Phân Tích Kỹ Năng",
        "🤖 Dự Đoán Lương (ML)",
        "📈 Thống Kê Mô Tả",
        "💡 Cố vấn AI (Gemini)"
    ]
    selected_page = st.sidebar.radio(
        "Chọn trang:",
        page_options,
        key='page_navigation'
    )
    
    st.sidebar.header("Bộ lọc dữ liệu")
    locations = sorted(df_jobs['primary_location'].astype(str).unique())
    categories = sorted(df_jobs['primary_category'].astype(str).unique())
    selected_location = st.sidebar.multiselect(
        '📍 Chọn Địa điểm:', options=locations,
        default=locations,
        key='filter_location'
    )
    selected_category = st.sidebar.multiselect(
        '📂 Chọn Ngành nghề:', options=categories,
        default=categories,
        key='filter_category'
    )
    exp_col = 'min_experience_years'
    if exp_col in df_jobs.columns and df_jobs[exp_col].notna().any():
        min_exp_val = int(df_jobs[exp_col].min(skipna=True))
        max_exp_val = int(df_jobs[exp_col].max(skipna=True))
        exp_default = (min_exp_val, max_exp_val)
        selected_exp_range = st.sidebar.slider(
            '⏳ Kinh nghiệm tối thiểu (năm):', min_value=min_exp_val, max_value=max_exp_val,
            value=exp_default, key='filter_experience'
        )
    else:
        selected_exp_range = None
    salary_col = 'min_salary_mil_vnd'
    if salary_col in df_jobs.columns and df_jobs[salary_col].notna().any():
        min_sal_val = float(df_jobs[salary_col].min(skipna=True))
        max_sal_val = float(df_jobs[salary_col].quantile(0.99, interpolation='nearest'))
        if min_sal_val >= max_sal_val:
            max_sal_val = float(df_jobs[salary_col].max(skipna=True))
            if min_sal_val > max_sal_val:
                min_sal_val = max_sal_val
        sal_default = (min_sal_val, max_sal_val)
        salary_median = df_jobs[salary_col].median()
        median_text = f" ({salary_median:.0f} Tr Median)" if pd.notna(salary_median) else ""
        selected_salary_range = st.sidebar.slider(
            f'💰 Lương tối thiểu{median_text}:',
            min_value=min_sal_val, max_value=max_sal_val,
            value=sal_default, step=0.5, key='filter_salary',
            format="%.1f Triệu VND"
        )
    else:
        selected_salary_range = None
    st.session_state.filters = {
        'location': selected_location,
        'category': selected_category,
        'experience': selected_exp_range,
        'salary': selected_salary_range
    }
    df_filtered = df_jobs.copy()
    current_filters = st.session_state.filters
    if current_filters.get('location'):
        df_filtered = df_filtered[df_filtered['primary_location'].isin(current_filters['location'])]
    if current_filters.get('category'):
        df_filtered = df_filtered[df_filtered['primary_category'].isin(current_filters['category'])]
    if current_filters.get('experience') and exp_col in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered[exp_col].between(current_filters['experience'][0], current_filters['experience'][1], inclusive='both')
        ]
    if current_filters.get('salary') and salary_col in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered[salary_col].between(current_filters['salary'][0], current_filters['salary'][1], inclusive='both')
        ]
    st.session_state.df_filtered = df_filtered

    # --- Main Content ---
    if selected_page == page_options[0]:
        st.title("🏠 Trang Chủ & Tổng Quan")
        st.markdown("Chào mừng bạn đến với dashboard phân tích thị trường tuyển dụng Việt Nam.")
        if st.session_state.original_filename:
            st.markdown(f"Dữ liệu đang phân tích từ file: `{st.session_state.original_filename}`")
        st.subheader("Kiểm tra dữ liệu thiếu (Tổng thể)")
        with st.expander("Xem chi tiết tỷ lệ thiếu của các cột"):
            if 'df_jobs' in st.session_state and st.session_state.df_jobs is not None:
                missing_data = st.session_state.df_jobs.isnull().sum()
                if missing_data.sum() == 0:
                    st.info("Chúc mừng! Dữ liệu gốc không có giá trị thiếu.")
                else:
                    missing_percent = (missing_data / len(st.session_state.df_jobs)) * 100
                    missing_df = pd.DataFrame({
                        'Cột': missing_data.index,
                        'Số lượng thiếu': missing_data.values,
                        'Tỷ lệ thiếu (%)': missing_percent.values
                    })
                    missing_df = missing_df.astype({'Cột': 'object'})  # Chuẩn hóa kiểu dữ liệu
                    st.dataframe(missing_df[missing_df['Số lượng thiếu'] > 0].sort_values('Tỷ lệ thiếu (%)', ascending=False))
                    st.caption("Tỷ lệ thiếu cao ở các cột quan trọng (lương, kinh nghiệm, địa điểm, ngành) sẽ ảnh hưởng đến chất lượng phân tích và khả năng huấn luyện mô hình AI.")
            else:
                st.info("Chưa có dữ liệu gốc để kiểm tra (Vui lòng tải file lên).")
        st.markdown("---")
        df_display = st.session_state.df_filtered
        if df_display is not None and not df_display.empty:
            st.header("📌 Tổng Quan (Theo bộ lọc)")
            total_jobs_filtered = df_display.shape[0]
            unique_companies_filtered = df_display['company_title'].nunique()
            avg_min_salary_filtered = df_display['min_salary_mil_vnd'].mean()
            avg_min_exp_filtered = df_display['min_experience_years'].mean()
            kpi_cols = st.columns(4)
            kpi_cols[0].metric(label="Số Tin Tuyển Dụng", value=f"{total_jobs_filtered:,}")
            kpi_cols[1].metric(label="Số Công Ty Tuyển", value=f"{unique_companies_filtered:,}")
            kpi_cols[2].metric(label="Lương Tối Thiểu TB", value=f"{avg_min_salary_filtered:.1f} Tr" if pd.notna(avg_min_salary_filtered) else "N/A")
            kpi_cols[3].metric(label="Kinh Nghiệm Tối Thiểu TB", value=f"{avg_min_exp_filtered:.1f} Năm" if pd.notna(avg_min_exp_filtered) else "N/A")
            st.markdown("---")
            st.subheader("Xem trước dữ liệu (đã lọc)")
            if 'job_title' in df_display.columns:
                df_display['job_title'] = df_display['job_title'].astype(str).replace('nan', 'Không xác định')
            df_display = df_display.astype({col: 'object' for col in df_display.columns if df_display[col].dtype == 'string'})
            st.dataframe(df_display.head(10), use_container_width=True, height=300)
            csv_download = convert_df_to_csv(df_display)
            st.download_button(
                label="📥 Tải xuống dữ liệu đã lọc (CSV)",
                data=csv_download,
                file_name='filtered_job_data.csv',
                mime='text/csv',
                key='download_filtered_home_v2'
            )
        elif uploaded_file is not None:
            st.warning("⚠️ Không tìm thấy dữ liệu phù hợp với bộ lọc hiện tại. Vui lòng thử điều chỉnh bộ lọc.")
        st.markdown("---")
        with st.expander("ℹ️ Thông tin về dữ liệu và dự án", expanded=False):
            st.markdown(f"""
            * **Nguồn dữ liệu:** Dữ liệu tham khảo được thu thập từ CareerBuilder.vn (02-03/2023) - File: `{st.session_state.get("original_filename", "N/A")}`.
            * **Xử lý:** Dữ liệu đã qua các bước làm sạch cơ bản (xử lý giá trị thiếu, chuẩn hóa định dạng...). Xem thêm ở mục kiểm tra dữ liệu thiếu ở trên.
            * **Mục tiêu đồ án:** Xây dựng dashboard tương tác đáp ứng yêu cầu môn học Trực quan hóa Dữ liệu.
                * *Yêu cầu chính:* Dữ liệu VN, đủ biến/dòng, trực quan phù hợp & rõ ràng, liên kết, tương tác, thiết kế hấp dẫn, phân tích sâu, tích hợp AI.
            * **Điều hướng:** Sử dụng menu bên trái để chọn phần phân tích.
            * **Lưu ý:** Phân tích dựa trên dữ liệu được cung cấp.
            """)
    elif selected_page == page_options[1]:
        st.title("📊 Phân Tích Thị Trường Chung")
        df_display = st.session_state.df_filtered
        if df_display is None or df_display.empty:
            st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc để hiển thị phân tích này.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Phân tích theo Ngành Nghề")
                tab_cat_count, tab_cat_salary = st.tabs(["Số lượng tin", "Phân bố lương"])
                category_counts = df_display['primary_category'].value_counts().reset_index()
                category_counts.columns = ['Ngành nghề', 'Số lượng tin']
                with tab_cat_count:
                    st.markdown("##### Số lượng tin tuyển dụng theo ngành")
                    top_n_cat = st.slider("Chọn Top N ngành nghề:", 5, min(30, len(category_counts)), 15, key='slider_cat_count_market')
                    if not category_counts.empty:
                        fig_cat_bar = px.bar(category_counts.head(top_n_cat), x='Số lượng tin', y='Ngành nghề', orientation='h', title=f'Top {top_n_cat} Ngành Nghề Nhiều Tin Nhất', text_auto=True, color='Số lượng tin', color_continuous_scale=px.colors.sequential.Blues)
                        fig_cat_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(400, top_n_cat*25), showlegend=False, uniformtext_minsize=8, uniformtext_mode='hide')
                        fig_cat_bar.update_traces(textposition='outside')
                        st.plotly_chart(fig_cat_bar, use_container_width=True)
                    else:
                        st.info("Không có dữ liệu ngành nghề.")
                with tab_cat_salary:
                    st.markdown("##### Phân bố lương tối thiểu theo ngành")
                    if 'min_salary_mil_vnd' in df_display.columns and df_display['min_salary_mil_vnd'].notna().any():
                        top_categories_list = category_counts.head(top_n_cat)['Ngành nghề'].tolist()
                        df_plot_cat_salary = df_display[df_display['primary_category'].isin(top_categories_list)].dropna(subset=['min_salary_mil_vnd'])
                        if not df_plot_cat_salary.empty:
                            fig_cat_box = px.box(df_plot_cat_salary, x='min_salary_mil_vnd', y='primary_category', title=f'Phân Bố Lương Tối Thiểu Top {top_n_cat} Ngành', labels={'min_salary_mil_vnd': 'Lương Tối Thiểu (Tr VND)', 'primary_category': 'Ngành Nghề'}, points="outliers", color='primary_category', color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig_cat_box.update_layout(yaxis={'categoryorder':'median ascending'}, height=max(400, top_n_cat*30), showlegend=False)
                            st.plotly_chart(fig_cat_box, use_container_width=True)
                            st.caption("Biểu đồ Boxplot: Đường giữa là trung vị, hộp là khoảng tứ phân vị (IQR), các điểm là outliers.")
                        else:
                            st.info("Không đủ dữ liệu lương cho ngành đã chọn.")
                    else:
                        st.warning("Thiếu dữ liệu lương tối thiểu.")
            with col2:
                st.subheader("📍 Phân tích theo Địa Điểm")
                tab_loc_count, tab_loc_salary = st.tabs(["Số lượng tin", "Phân bố lương"])
                location_counts = df_display['primary_location'].value_counts().reset_index()
                location_counts.columns = ['Địa điểm', 'Số lượng tin']
                with tab_loc_count:
                    st.markdown("##### Số lượng tin tuyển dụng theo địa điểm")
                    top_n_loc = st.slider("Chọn Top N địa điểm:", 5, min(30, len(location_counts)), 15, key='slider_loc_count_market')
                    if not location_counts.empty:
                        fig_loc_bar = px.bar(location_counts.head(top_n_loc), x='Số lượng tin', y='Địa điểm', orientation='h', title=f'Top {top_n_loc} Địa Điểm Nhiều Tin Nhất', text_auto=True, color='Số lượng tin', color_continuous_scale=px.colors.sequential.Greens)
                        fig_loc_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(400, top_n_loc*25), showlegend=False, uniformtext_minsize=8, uniformtext_mode='hide')
                        fig_loc_bar.update_traces(textposition='outside')
                        st.plotly_chart(fig_loc_bar, use_container_width=True)
                    else:
                        st.info("Không có dữ liệu địa điểm.")
                with tab_loc_salary:
                    st.markdown("##### Phân bố lương tối thiểu theo địa điểm")
                    if 'min_salary_mil_vnd' in df_display.columns and df_display['min_salary_mil_vnd'].notna().any():
                        top_locations_list = location_counts.head(top_n_loc)['Địa điểm'].tolist()
                        df_plot_loc_salary = df_display[df_display['primary_location'].isin(top_locations_list)].dropna(subset=['min_salary_mil_vnd'])
                        if not df_plot_loc_salary.empty:
                            fig_loc_box = px.box(df_plot_loc_salary, x='min_salary_mil_vnd', y='primary_location', title=f'Phân Bố Lương Tối Thiểu Top {top_n_loc} Địa Điểm', labels={'min_salary_mil_vnd': 'Lương Tối Thiểu (Tr VND)', 'primary_location': 'Địa Điểm'}, points="outliers", color='primary_location', color_discrete_sequence=px.colors.qualitative.Set2)
                            fig_loc_box.update_layout(yaxis={'categoryorder':'median ascending'}, height=max(400, top_n_loc*30), showlegend=False)
                            st.plotly_chart(fig_loc_box, use_container_width=True)
                            st.caption("Biểu đồ Boxplot: Đường giữa là trung vị, hộp là khoảng tứ phân vị (IQR), các điểm là outliers.")
                        else:
                            st.info("Không đủ dữ liệu lương cho địa điểm đã chọn.")
                    else:
                        st.warning("Thiếu dữ liệu lương tối thiểu.")
            st.markdown("---")
            st.subheader("📊 Phân Bố Ngành Nghề Theo Địa Điểm (Biểu đồ cột chồng)")
            location_counts_agg = df_display['primary_location'].value_counts()
            category_counts_agg = df_display['primary_category'].value_counts()
            top_n_loc_stack = st.slider("Chọn Top N địa điểm cho biểu đồ:", 3, min(20, len(location_counts_agg)), 8, key='slider_loc_stack_v2_market')
            top_n_cat_stack = st.slider("Chọn Top N ngành nghề để hiển thị chi tiết:", 3, min(20, len(category_counts_agg)), 10, key='slider_cat_stack_v2_market')
            top_loc_list_stack = location_counts_agg.head(top_n_loc_stack).index.tolist()
            top_cat_list_stack = category_counts_agg.head(top_n_cat_stack).index.tolist()
            df_stack_data_loc_filtered = df_display[df_display['primary_location'].isin(top_loc_list_stack)].copy()
            df_stack_data_loc_filtered['category_display'] = df_stack_data_loc_filtered['primary_category'].apply(
                lambda x: x if x in top_cat_list_stack else 'Ngành Khác'
            )
            all_display_categories = top_cat_list_stack + ['Ngành Khác']
            df_stack_data_loc_filtered['category_display'] = pd.Categorical(df_stack_data_loc_filtered['category_display'], categories=all_display_categories, ordered=True)
            location_category_counts_stack = df_stack_data_loc_filtered.groupby(['primary_location', 'category_display'], observed=False).size().reset_index(name='count')
            if not location_category_counts_stack.empty:
                loc_order = location_category_counts_stack.groupby('primary_location', observed=False)['count'].sum().sort_values(ascending=False).index
                location_category_counts_stack['primary_location'] = pd.Categorical(location_category_counts_stack['primary_location'], categories=loc_order, ordered=True)
                location_category_counts_stack = location_category_counts_stack.sort_values(['primary_location', 'category_display'])
                color_map = {cat: color for cat, color in zip(top_cat_list_stack, px.colors.qualitative.Pastel)}
                color_map['Ngành Khác'] = '#AAAAAA'
                fig_stacked_bar = px.bar(location_category_counts_stack, x='primary_location', y='count', color='category_display', title=f'Phân Bố Top {top_n_cat_stack} Ngành (+ Ngành Khác) tại Top {top_n_loc_stack} Địa Điểm', labels={'primary_location': 'Địa Điểm', 'count': 'Số Lượng Tin Tuyển Dụng', 'category_display': 'Ngành Nghề'}, height=600, color_discrete_map=color_map, custom_data=['category_display'])
                fig_stacked_bar.update_layout(xaxis={'categoryorder':'total descending'}, legend_title_text='Ngành Nghề')
                fig_stacked_bar.update_traces(hovertemplate="<b>Địa điểm:</b> %{x}<br><b>Ngành:</b> %{customdata[0]}<br><b>Số lượng:</b> %{y}<extra></extra>")
                st.plotly_chart(fig_stacked_bar, use_container_width=True)
                st.caption("Mỗi cột là một địa điểm. Các màu khác nhau thể hiện số lượng tin của Top N ngành và nhóm 'Ngành Khác'.")
            else:
                st.info("Không có dữ liệu giao nhau giữa Top N địa điểm và Top N ngành nghề đã chọn để vẽ biểu đồ cột chồng.")
    elif selected_page == page_options[2]:
        st.title("💰 Phân Tích Lương & Kinh Nghiệm")
        df_display = st.session_state.df_filtered
        if df_display is None or df_display.empty:
            st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc để hiển thị phân tích này.")
        else:
            salary_col = 'min_salary_mil_vnd'
            exp_col = 'min_experience_years'
            if salary_col not in df_display.columns or df_display[salary_col].isnull().all():
                st.error(f"Thiếu dữ liệu cột lương '{salary_col}' để phân tích.")
            elif exp_col not in df_display.columns or df_display[exp_col].isnull().all():
                st.error(f"Thiếu dữ liệu cột kinh nghiệm '{exp_col}' để phân tích.")
            else:
                st.subheader("Phân Bố Mức Lương Tối Thiểu")
                fig_hist_min_sal = px.histogram(df_display.dropna(subset=[salary_col]), x=salary_col, nbins=50, title='Phân Bố Mức Lương Tối Thiểu (Triệu VND)', labels={salary_col: 'Mức Lương Tối Thiểu (Triệu VND)'}, marginal="box")
                fig_hist_min_sal.update_layout(bargap=0.1)
                st.plotly_chart(fig_hist_min_sal, use_container_width=True)
                st.subheader("Mối Quan Hệ Giữa Lương và Kinh Nghiệm")
                col_scatter, col_box = st.columns(2)
                df_analysis = df_display.dropna(subset=[exp_col, salary_col]).copy()
                with col_scatter:
                    st.markdown("##### Scatter Plot Lương vs. Kinh Nghiệm")
                    if not df_analysis.empty:
                        color_options_scatter = {'primary_category': 'Ngành Nghề', 'primary_location': 'Địa Điểm', None: 'Không tô màu'}
                        selected_color_scatter = st.selectbox("Tô màu điểm theo:", list(color_options_scatter.keys()), format_func=lambda x: color_options_scatter[x], key='scatter_color_exp_salary')
                        fig_scatter = px.scatter(df_analysis, x=exp_col, y=salary_col, title='Lương Tối Thiểu theo Kinh Nghiệm', labels={exp_col: 'Kinh Nghiệm Tối Thiểu (Năm)', salary_col: 'Lương Tối Thiểu (Tr VND)'}, color=selected_color_scatter, hover_name='job_title', opacity=0.6, trendline="ols", trendline_scope="overall", trendline_color_override="darkblue", height=500)
                        fig_scatter.update_layout(legend_title_text=color_options_scatter.get(selected_color_scatter, ''))
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        st.caption("Mỗi điểm là một tin tuyển dụng. Đường màu xanh đậm là đường xu hướng tổng thể.")
                    else:
                        st.info("Không đủ dữ liệu cho scatter plot.")
                with col_box:
                    st.markdown("##### Box Plot Lương theo Nhóm Kinh Nghiệm")
                    if not df_analysis.empty:
                        max_exp_val_box = df_analysis[exp_col].max()
                        if max_exp_val_box <= 5:
                            bins = list(range(-1, int(max_exp_val_box) + 1))
                            labels = [f'{i} năm' for i in range(int(max_exp_val_box) + 1)]
                        else:
                            bins = [-1, 0, 1, 2, 3, 4, 5, 10, max_exp_val_box]
                            labels = ['0 năm', '1 năm', '2 năm', '3 năm', '4 năm', '5 năm', '6-10 năm', '10+ năm']
                            if max_exp_val_box <= 10:
                                bins = bins[:-1]
                                labels = labels[:-1]
                                if max_exp_val_box <= 5:
                                    bins = bins[:-1]
                                    labels = labels[:-1]
                        if len(bins) > 1:
                            try:
                                df_analysis['experience_group'] = pd.cut(df_analysis[exp_col], bins=bins, labels=labels, right=True)
                                df_plot_box = df_analysis.dropna(subset=['experience_group'])
                                if not df_plot_box.empty:
                                    fig_box_exp = px.box(df_plot_box, x='experience_group', y=salary_col, title='Phân Bố Lương theo Nhóm Kinh Nghiệm', labels={'experience_group': 'Nhóm Kinh Nghiệm', salary_col: 'Lương Tối Thiểu (Tr VND)'}, points="outliers", color='experience_group', color_discrete_sequence=px.colors.qualitative.Bold, height=500)
                                    fig_box_exp.update_layout(showlegend=False)
                                    st.plotly_chart(fig_box_exp, use_container_width=True)
                                else:
                                    st.info("Không có dữ liệu sau khi phân nhóm kinh nghiệm.")
                            except Exception as e:
                                st.warning(f"Lỗi khi phân nhóm kinh nghiệm: {e}. Hiển thị theo từng năm (<=10):")
                                df_plot_box_fb = df_analysis[df_analysis[exp_col] <= 10]
                                if not df_plot_box_fb.empty:
                                    fig_box_exp_fb = px.box(df_plot_box_fb, x=exp_col, y=salary_col, title='Phân Bố Lương theo Kinh Nghiệm (Từng năm <= 10)', labels={exp_col: 'Kinh Nghiệm (Năm)', salary_col: 'Lương Tối Thiểu (Tr VND)'}, points="outliers")
                                    fig_box_exp_fb.update_xaxes(type='category')
                                    st.plotly_chart(fig_box_exp_fb, use_container_width=True)
                                else:
                                    st.info("Không có dữ liệu kinh nghiệm <= 10 năm.")
                        else:
                            st.info("Không đủ khoảng kinh nghiệm để phân nhóm.")
                    else:
                        st.info("Không đủ dữ liệu cho box plot lương theo kinh nghiệm.")
                st.subheader("Xu Hướng Lương Theo Kinh Nghiệm")
                st.markdown('<div class="section-title">Biểu Đồ Xu Hướng</div>', unsafe_allow_html=True)
                grouped = df_display.groupby("min_experience_years")[["min_salary_mil_vnd", "max_salary_mil_vnd"]].mean().reset_index()
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
                st.markdown("---")
                st.subheader("📝 Nhận Xét")
                st.markdown("""
                * **Phân bố lương:** Biểu đồ histogram thường cho thấy lương tập trung ở mức thấp đến trung bình và có một số giá trị rất cao (lệch phải).
                * **Lương & Kinh nghiệm:** Có xu hướng lương tăng theo kinh nghiệm, nhưng mức độ tăng và sự biến động khác nhau tùy ngành nghề và địa điểm.
                * **Xu hướng lương:** Biểu đồ đường cho thấy mức lương trung bình (tối thiểu và tối đa) có xu hướng tăng theo số năm kinh nghiệm, với mức lương tối đa thường cao hơn đáng kể ở các mức kinh nghiệm cao.
                """)
    elif selected_page == page_options[3]:
        st.title("🛠️ Phân Tích Kỹ Năng Yêu Cầu")
        df_display = st.session_state.df_filtered
        if df_display is None or df_display.empty:
            st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc để hiển thị phân tích này.")
        else:
            req_col = 'job_requirements'
            if req_col not in df_display.columns or df_display[req_col].isnull().all():
                st.error(f"Thiếu dữ liệu cột '{req_col}' để phân tích kỹ năng.")
            else:
                st.subheader("Kỹ Năng Phổ Biến Nhất")
                num_display_skills = st.slider("Chọn số lượng kỹ năng hàng đầu:", 10, 50, 20, key='slider_skills_page')
                requirements_text = df_display[req_col]
                if not requirements_text.empty:
                    with st.spinner("Đang trích xuất và phân tích kỹ năng..."):
                        skills_df, full_text_req = extract_skills_from_requirements(requirements_text, num_display_skills)
                    if not skills_df.empty:
                        col_bar, col_cloud = st.columns([0.6, 0.4])
                        with col_bar:
                            st.markdown(f"##### Top {num_display_skills} Kỹ Năng Phổ Biến")
                            skills_df = skills_df.astype({'Kỹ năng': 'object'})  # Chuẩn hóa kiểu dữ liệu
                            fig_skills = px.bar(skills_df, x='Số lần xuất hiện', y='Kỹ năng', orientation='h', title=f'Top {num_display_skills} Kỹ Năng Phổ Biến', text='Số lần xuất hiện', color='Số lần xuất hiện', color_continuous_scale=px.colors.sequential.Viridis)
                            fig_skills.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(400, num_display_skills*20), showlegend=False, uniformtext_minsize=8, uniformtext_mode='hide')
                            fig_skills.update_traces(textposition='outside')
                            st.plotly_chart(fig_skills, use_container_width=True)
                        with col_cloud:
                            st.markdown("##### Word Cloud Kỹ Năng")
                            try:
                                skill_frequencies = {skill: count for skill, count in skills_df.values}
                                if skill_frequencies:
                                    wordcloud = WordCloud(width=800, height=600, mode="RGBA", background_color=None, max_words=100, colormap='viridis', collocations=False, contour_width=1, contour_color='steelblue').generate_from_frequencies(skill_frequencies)
                                    fig_cloud, ax = plt.subplots(figsize=(10, 7))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig_cloud)
                                    st.caption("Kích thước chữ thể hiện tần suất xuất hiện của kỹ năng.")
                                else:
                                    st.info("Không đủ dữ liệu tạo Word Cloud.")
                            except ImportError:
                                st.warning("Vui lòng cài đặt thư viện `wordcloud` và `matplotlib` để xem Word Cloud.")
                            except Exception as e:
                                st.warning(f"Lỗi tạo Word Cloud: {e}")
                    else:
                        st.info("Không tìm thấy kỹ năng nào (theo danh sách keywords) trong dữ liệu đã lọc.")
                else:
                    st.info("Không có yêu cầu công việc nào trong dữ liệu đã lọc.")
                st.markdown("---")
                st.subheader("📝 Nhận Xét")
                st.markdown("""
                * Biểu đồ cột và Word Cloud giúp xác định các kỹ năng được yêu cầu nhiều nhất.
                * **Quan trọng:** Kết quả phụ thuộc lớn vào danh sách `skills_keywords` trong code. Đã rà soát và bổ sung các kỹ năng liên quan!
                """)
    elif selected_page == page_options[4]:
        st.title("🤖 Dự Đoán Lương Tối Thiểu (ML)")
        st.markdown("Sử dụng mô hình Machine Learning (Random Forest) để dự đoán mức lương tối thiểu dựa trên kinh nghiệm, ngành nghề và địa điểm.")
        df_display = st.session_state.df_filtered
        model_features = ['min_experience_years', 'primary_category', 'primary_location']
        target = 'min_salary_mil_vnd'
        required_model_cols = model_features + [target]
        min_records_threshold = 50
        if df_display is None or df_display.empty:
            st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc để huấn luyện mô hình.")
        else:
            missing_model_cols = [col for col in required_model_cols if col not in df_display.columns]
            if missing_model_cols:
                st.error(f"Thiếu các cột cần thiết cho mô hình AI: {', '.join(missing_model_cols)}")
            else:
                df_model_data = df_display[required_model_cols].copy().dropna()
                num_valid_records = df_model_data.shape[0]
                st.info(f"Số lượng bản ghi hợp lệ (đủ dữ liệu cho các cột: {', '.join(required_model_cols)}) sau khi lọc: **{num_valid_records}**")
                if num_valid_records < min_records_threshold:
                    st.error(f"Số lượng bản ghi hợp lệ ({num_valid_records}) thấp hơn ngưỡng tối thiểu ({min_records_threshold}). Không thể huấn luyện mô hình đáng tin cậy.")
                    st.markdown("**Gợi ý:** Thử nới lỏng các bộ lọc dữ liệu ở sidebar.")
                    st.session_state.model_results = None
                    st.session_state.model_metrics = {}
                    st.session_state.model_pipeline = None
                else:
                    col_train_btn, col_model_status = st.columns([0.3, 0.7])
                    # with col_train_btn:
                    #     if st.button("🔄 Huấn luyện/Cập nhật mô hình"):
                    #         st.session_state.model_results = None
                    #         st.session_state.model_metrics = {}
                    #         st.session_state.model_pipeline = None
                    #         st.info("Đã xóa mô hình cũ. Đang chuẩn bị huấn luyện lại...")
                    if st.session_state.model_pipeline is None:
                        with st.spinner(f"Đang huấn luyện mô hình với {num_valid_records} bản ghi..."):
                            results_df, rmse, mae, r2, oob = train_salary_model(df_model_data)
                            if results_df is not None:
                                st.session_state.model_results = results_df
                                st.session_state.model_metrics = {'RMSE': rmse, 'MAE': mae, 'R2 Score': r2, 'OOB Score': oob}
                                st.success(f"Huấn luyện mô hình thành công trên {num_valid_records} bản ghi!")
                            else:
                                st.session_state.model_pipeline = None
                    else:
                        with col_model_status:
                            if st.session_state.model_metrics:
                                metrics = st.session_state.model_metrics
                                st.success(f"✔️ Mô hình đã huấn luyện (R2: {metrics.get('R2 Score', 0):.3f}).")
                            else:
                                st.success("✔️ Mô hình đã được huấn luyện.")
                    if st.session_state.model_pipeline is not None and st.session_state.model_results is not None:
                        st.subheader("Đánh Giá Mô Hình")
                        metrics = st.session_state.model_metrics
                        oob_score_val = metrics.get('OOB Score')
                        oob_text = f"(OOB: {oob_score_val:.3f})" if oob_score_val else ""
                        m_col1, m_col2, m_col3 = st.columns(3)
                        m_col1.metric("RMSE (Lỗi TB)", f"{metrics.get('RMSE', 0):.2f} Tr")
                        m_col2.metric("MAE (Lỗi Tuyệt đối TB)", f"{metrics.get('MAE', 0):.2f} Tr")
                        m_col3.metric(f"R2 Score {oob_text}", f"{metrics.get('R2 Score', 0):.3f}")
                        st.caption(f"Đánh giá trên tập kiểm tra (20% của {num_valid_records}). OOB Score là độ chính xác ước tính trên dữ liệu chưa thấy khi huấn luyện RF.")
                        results_df = st.session_state.model_results
                        results_df = results_df.astype({'Thực tế': 'float64', 'Dự đoán': 'float64'})
                        fig_pred = px.scatter(results_df, x='Thực tế', y='Dự đoán', title='Kết Quả Dự Đoán vs. Thực Tế', labels={'Thực tế': 'Lương Thực Tế (Tr VND)', 'Dự đoán': 'Lương Dự Đoán (Tr VND)'}, opacity=0.7, height=500)
                        min_val = min(results_df['Thực tế'].min(), results_df['Dự đoán'].min()) * 0.95
                        max_val = max(results_df['Thực tế'].max(), results_df['Dự đoán'].max()) * 1.05
                        fig_pred.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", width=2, dash="dash"), name="Lý tưởng")
                        fig_pred.update_xaxes(range=[min_val, max_val])
                        fig_pred.update_yaxes(range=[min_val, max_val])
                        st.plotly_chart(fig_pred, use_container_width=True)
                        st.subheader("Thử Dự Đoán Lương")
                        all_categories_model = sorted(st.session_state.df_jobs['primary_category'].astype(str).unique())
                        all_locations_model = sorted(st.session_state.df_jobs['primary_location'].astype(str).unique())
                        with st.form("prediction_form"):
                            pred_exp = st.number_input("Số năm kinh nghiệm:", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
                            default_cat_index = all_categories_model.index(st.session_state.filters['category'][0]) if st.session_state.filters.get('category') and st.session_state.filters['category'][0] in all_categories_model else 0
                            default_loc_index = all_locations_model.index(st.session_state.filters['location'][0]) if st.session_state.filters.get('location') and st.session_state.filters['location'][0] in all_locations_model else 0
                            pred_cat = st.selectbox("Ngành nghề:", options=all_categories_model, index=default_cat_index)
                            pred_loc = st.selectbox("Địa điểm:", options=all_locations_model, index=default_loc_index)
                            submitted = st.form_submit_button("Dự đoán mức lương tối thiểu")
                            if submitted:
                                input_data = pd.DataFrame({'min_experience_years': [pred_exp], 'primary_category': [pred_cat], 'primary_location': [pred_loc]})
                                try:
                                    prediction = st.session_state.model_pipeline.predict(input_data)
                                    st.success(f"Mức lương tối thiểu dự đoán: **{prediction[0]:.1f} Triệu VND**")
                                except Exception as e:
                                    st.error(f"Lỗi khi dự đoán: {e}")
                        st.markdown("---")
                        st.subheader("📝 Nhận Xét")
                        st.markdown(f"""
                        * **Mô hình:** Random Forest dự đoán lương dựa trên kinh nghiệm, ngành nghề, địa điểm.
                        * **Đánh giá:** Các chỉ số RMSE ({metrics.get('RMSE', 0):.2f} Tr), MAE ({metrics.get('MAE', 0):.2f} Tr), và R2 Score ({metrics.get('R2 Score', 0):.3f}) cho thấy mức độ chính xác của mô hình trên tập dữ liệu kiểm tra. OOB score ({oob_score_val:.3f} nếu có) là một ước tính khác về hiệu suất.
                        * **Hạn chế:** Kết quả chỉ mang tính tham khảo, phụ thuộc vào chất lượng dữ liệu, bộ lọc hiện tại và sự đơn giản của mô hình (chỉ dùng 3 yếu tố).
                        """)
                    elif st.session_state.model_pipeline is None and num_valid_records >= min_records_threshold:
                        st.info("Nhấn nút 'Huấn luyện/Cập nhật mô hình' để bắt đầu.")
    elif selected_page == page_options[5]:
        st.header("📈 Thống Kê Mô Tả")
        df = st.session_state.df_filtered
        if df is None or df.empty:
            st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc để hiển thị phân tích này.")
        else:
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
                df = df.astype({col: 'object' for col in df.columns if df[col].dtype == 'string'})
                st.dataframe(df.describe(include='all'), height=300)
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
                        grouped = filtered_df.groupby('primary_location', observed=False)[cat_select].value_counts().unstack(fill_value=0)
                        grouped.columns.name = cat_select
                        grouped.index.name = 'primary_location'
                        grouped = grouped.astype('object')
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
    elif selected_page == page_options[6]:
        st.title("💡 Cố vấn AI (Gemini)")
        if not ai_configured_successfully:
            st.error("🚨 Cấu hình AI thất bại hoặc thiếu API Key. Vui lòng kiểm tra GEMINI_API_KEY trong file .env hoặc cấu hình secrets.")
            st.warning("Tab này sẽ không hoạt động cho đến khi AI được cấu hình thành công.")
            st.stop()
        df_for_ai = st.session_state.get('df_jobs')
        if df_for_ai is None:
            st.warning("⚠️ Vui lòng tải lên hoặc chờ dữ liệu được xử lý để sử dụng tính năng này.")
        else:
            st.info("Chức năng này sử dụng Google Gemini AI để cung cấp thông tin chi tiết và trả lời câu hỏi dựa trên dữ liệu tuyển dụng hiện tại.")
            with st.expander("Xem thông tin và dữ liệu mẫu được cung cấp cho AI", expanded=False):
                st.subheader("Thông tin tổng quan về dữ liệu (cho AI)")
                dataset_info_str = get_dataset_info(df_for_ai)
                st.text_area("Dataset Info (Text):", dataset_info_str, height=300)
                st.subheader("Dữ liệu mẫu (cho AI)")
                try:
                    csv_string_data = df_for_ai.to_csv(index=False, header=True, quoting=csv.QUOTE_MINIMAL)
                    data_rows_for_ai = csv_string_data.splitlines()
                    max_rows_to_feed = 500
                    data_summary_for_ai = "\\n".join(data_rows_for_ai[:max_rows_to_feed])
                    if len(data_rows_for_ai) > max_rows_to_feed:
                        data_summary_for_ai += f"\\n... (Dữ liệu được cắt ngắn còn {max_rows_to_feed} dòng trong tổng số {len(data_rows_for_ai)} dòng)"
                    st.text_area("Dữ liệu dạng CSV (một phần) gửi cho AI:", data_summary_for_ai, height=200)
                except Exception as e:
                    st.error(f"Lỗi khi chuẩn bị dữ liệu cho AI: {e}")
                    data_summary_for_ai = "Không có dữ liệu do lỗi."
            data_context_for_ai = f"""
            Thông tin Dataset:
            {dataset_info_str}
            
            Một phần dữ liệu (CSV format):
            {data_summary_for_ai}
            """
            st.divider()
            st.subheader("💬 Đặt câu hỏi cho AI")
            user_query = st.text_area("Câu hỏi của bạn về dữ liệu tuyển dụng:", height=100, 
                                      placeholder="Ví dụ: Các kỹ năng nào đang được yêu cầu nhiều nhất cho vị trí Data Analyst? Mức lương trung bình cho các vị trí Fresher là bao nhiêu?")
            if st.button("Gửi câu hỏi cho AI", type="primary", key="ai_ask_button"):
                if not user_query:
                    st.warning("Vui lòng nhập câu hỏi.")
                elif not ai_configured_successfully or genai_model is None:
                    st.error("AI chưa sẵn sàng. Vui lòng kiểm tra cấu hình.")
                else:
                    with st.spinner("🤖 AI đang suy nghĩ..."):
                        feedback = generate_feedback(data_context_for_ai, user_query)
                        st.subheader("AI trả lời:")
                        st.markdown(feedback)
elif uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp CSV qua thanh bên trái để bắt đầu.")
st.sidebar.markdown("---")
st.sidebar.markdown(f"© {pd.Timestamp.now().year} - [Nhóm 3]")
st.sidebar.info("Dashboard được tạo bằng Streamlit.")