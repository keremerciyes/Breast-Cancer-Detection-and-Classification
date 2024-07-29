import streamlit as st
import pandas as pd
import random

def generate_patient_data():
    names = ["John", "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack", "Kathy", "Liam", "Mona", "Nina", "Oscar", "Pam", "Quinn", "Rick", "Sue"]
    surnames = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]
    
    data = {
        "Name": [name[:2] + '*' * (len(name) - 2) for name in names],
        "Surname": [surname[:2] + '*' * (len(surname) - 2) for surname in surnames],
        "Age": [round(random.randint(10, 100) / 10) * 10 for _ in names]
    }
    return pd.DataFrame(data)

def home():
    st.title("Home Page")
    patient_data = generate_patient_data()
    
    # Create a list of formatted strings with names, surnames, and ages
    patients_with_ages = [f"{row['Name']} {row['Surname']} ({row['Age']} years old)" for _, row in patient_data.iterrows()]
    
    selected_patient = st.selectbox("Select Patient", patients_with_ages)
    if st.button("Add Patient", key="button2"):
        st.write("Add Patient functionality not implemented.")
    if st.button("Delete Patient", key="button3"):
        st.write("Delete Patient functionality not implemented.")
    if st.button("Go to Radiology Page", key="button4"):
        st.session_state["selected_patient"] = selected_patient
        st.session_state["page"] = "Radiology" 
        st.rerun()

if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = None

if st.session_state.get("logged_in", False):
    home()
else:
    st.error("Please login first.")
