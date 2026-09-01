import streamlit as st

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

st.title("Employee Attrition Prediction")
st.write(
    "A lightweight Streamlit entry point for the employee attrition prediction project. "
    "This UI is ready for future model integration."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Input (Placeholder)")
    st.info("Add employee feature inputs here (e.g., age, overtime, monthly income, role).")

with col2:
    st.subheader("Prediction Output (Placeholder)")
    st.info("Predicted attrition risk and class label will be shown here after model integration.")

st.subheader("Explainability (Placeholder)")
st.info("SHAP/global-local explanation components can be added here in the next iteration.")
