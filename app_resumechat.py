import streamlit as st

def draw_sidebar():
    with st.sidebar:
        st.title("Resume Chat")
        st.write("This is a chatbot that can help you chat with your resume.")
        st.write("You can ask questions about your resume, and the chatbot will answer you.")
        st.file_uploader("Upload your resume", type="pdf")


def main():
    st.title("Resume Chat")
    draw_sidebar()
if __name__ == "__main__":
    main()