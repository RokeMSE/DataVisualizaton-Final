import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import sklearn.preprocessing as skp 
import plotly.express as px 
import csv
from io import StringIO, BytesIO # BytesIO not used currently
import os
from dotenv import load_dotenv

# AI Implementation
import google.generativeai as genai

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file in the same directory
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configure AI ---
MODEL_NAME = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    st.error("🚨 GEMINI_API_KEY environment variable not found. Please set it in your .env file.")
    st.stop() # Stop execution if API key is missing
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY) # Configure the API key for Google Generative AI
        # Create the model instance once
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        st.error(f"Error configuring Google AI or creating model: {e}")
        st.stop()

# --- Constants ---
# Using the path provided by the user
CSV_FILE_PATH = 'Data/cleaned_vietnamese_job_posting.csv'

# --- Helper Functions ---

@st.cache_data # Cache the data loading to improve performance
def load_data(file_path):
    """Loads the CSV data from the specified path."""
    try:
        # Corrected indentation
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        # Corrected indentation
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory relative to where you run streamlit.")
        return None
    except Exception as e:
        # Corrected indentation
        st.error(f"Error loading CSV file: {e}")
        return None

def generate_feedback(data, user_query):
    """Generates feedback using the Google Generative AI model."""
    # For wholesome content only
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE",
        },
    ]

    # Generation config
    generation_config = genai.types.GenerationConfig(
        temperature=0.9, # Randomize the output
        max_output_tokens=1000, 
        top_p=0.9,
        top_k=40,
    )

    # Construct the prompt (same as before)
    prompt = f"""
    Bạn là một chuyên gia phân tích dữ liệu và bạn có khả năng phân tích dữ liệu CSV.
    Dưới đây là một tóm tắt về dữ liệu mà bạn sẽ phân tích:
    {data}
    
    Query của người dùng: {user_query}
    
    Bạn hãy phân tích dữ liệu và trả lời câu hỏi của người dùng một cách chi tiết và dễ hiểu.
    Hãy cung cấp các thông tin hữu ích và có thể bao gồm các biểu đồ hoặc hình ảnh nếu cần thiết.
    """

    try:
        # Generate content using the configured model, prompt, config, and safety settings
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
            )
        return response.text
    except Exception as e:
        st.error(f"Error generating feedback from AI: {e}")
        print(response.prompt_feedback) # Uncomment to debug potential blocks
        return "Sorry, I encountered an error while generating the feedback."

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wide layout
st.title("📊 Phân tích thị trường công việc Việt Nam cùng Google AI!")
st.write(f"Data được lấy từ: `{CSV_FILE_PATH}`")

# --- Load Data ---
df = load_data(CSV_FILE_PATH)

def read_csv_file(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding for special characters
            reader = csv.reader(file)
            header = next(reader)  # Read the header row
            for i, row in enumerate(reader, 1):  # Start row numbering at 1
                # Create a formatted string for each row
                row_data = [f"{header[j]}: {row[j]}" for j in range(len(header)) if j < len(row)]
                formatted_row = f"Row {i}: {', '.join(row_data)}"
                data.append(formatted_row)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it exists in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

data_to_feed = None  # Initialize data_to_feed to None
if df is not None:
    st.success("CSV data loaded successfully!")

    # --- Display Data Summary & Options ---
    col1, col2 = st.columns(2)

    data = read_csv_file(CSV_FILE_PATH) # Read the CSV file into a string
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head()) # Show the first few rows

    with col2:
        st.subheader("Data Overview")
        st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.write("**Columns:**")
        st.write(", ".join(df.columns.tolist()))

        # Generate a simple text summary for the AI
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        # Use the read_csv_file output to create a full data summary
        if data is not None:
            max_rows_to_feed = 1000  # Adjust based on token limits or performance needs
            data_summary = "\n".join(data[:max_rows_to_feed])  # Limit to avoid overwhelming the AI
            if len(data) > max_rows_to_feed:
                data_summary += f"\n... (Truncated to {max_rows_to_feed} rows out of {len(data)} total rows)"
        else:
            data_summary = "No data available due to an error loading the CSV."

        # Combine dataset info and row data for the AI
        data_to_feed = f"""
        Dataset Info:
        {info_str}
        
        Dataset Rows (Sample):
        {data_summary}
        """

    st.divider()

    # --- User Input and AI Feedback ---
    st.subheader("🤖 Hỏi AI thêm về các job!")
    user_query = st.text_area("Đặt câu hỏi cho AI trả lời:", height=100, placeholder="Vd: Các job phổ biến nhất là gì? Tóm tắt các loại công ty.")

    if st.button("Hỏi AI", type="primary"):
        if user_query:
            with st.spinner("Generating feedback using Google AI..."):
                feedback = generate_feedback(data_to_feed, user_query)
                st.subheader("AI trả lời là:")
                st.markdown(feedback) # Use markdown to render potential formatting from the AI
        else:
            st.warning("Please enter a question or request.")
else:
    st.error("Failed to load data. Please check the file path and ensure the CSV is valid.")