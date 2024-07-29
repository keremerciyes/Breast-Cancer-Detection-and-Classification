import streamlit as st

def login():
    st.title("Login")
    username = st.text_input("Username", key="login")
    password = st.text_input("Password", type="password", key="password")
    if st.button("Login", key="button1"):
        if username == "admin" and password == "password":  # Simple hardcoded login
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["page"] = "Home"
            #st.rerun()

        else:
            st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    st.success(f"Welcome {st.session_state['username']}!")
