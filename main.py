import streamlit as st
from login import login
from home import home
from radiology import radiology_page
import sys
print(sys.executable)

PAGES = {
    "Login": login,
    "Home": home,
    "Radiology": radiology_page
}

if "page" not in st.session_state:
    st.session_state["page"] = "Login"

def navigate(page):
    st.session_state["page"] = page

st.sidebar.title("Navigation")
for page in PAGES.keys():
    st.sidebar.button(page, on_click=navigate, args=(page,))

page = st.session_state["page"]
PAGES[page]()
