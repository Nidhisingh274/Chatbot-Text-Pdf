import streamlit as st                      # Used to build the web interface
from transformers import pipeline           # Hugging Face function that loads pre-trained NLP models
import PyPDF2                               # used to extract text from PDF files

# Function to extract text from uploaded file
def load_file():
    uploaded_file = st.file_uploader("📂 Upload a .txt or .pdf file", type=['txt', 'pdf'])
    
    if uploaded_file:
        # If it's a text file
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")

        # If it's a PDF file
        elif uploaded_file.type == "application/pdf":
            text = ""                             # "" to avoid None errors if a page has no extractable text.
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""  # Some pages might return None
            return text
    return None

# Main Streamlit app
def main():
    st.title("📄 Ask Your File")
    st.write("Upload your file, ask any question, and get an answer based on its content!")

    # Step 1: Upload and display content
    text = load_file()

    if text:
        with st.expander("🔍 View File Content"):
            st.text_area("File Content", text, height=300)

        # Step 2: Ask a question
        question = st.text_input("Ask any question about the text")

        if question:
            qa_model = pipeline("question-answering", model = "deepset/minilm-uncased-squad2") # Encoder-only Transformer Architecture
            with st.spinner("Searching for the answer..."):
                result = qa_model(question=question, context=text)
                st.success(f"✅ Answer: {result['answer']}")
    else:
        st.info("📤 Please upload a .txt or .pdf file to get started.")

if __name__ == "__main__":
    main()
