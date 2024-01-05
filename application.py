import streamlit as st

from src.pipeline.prediction_pipeline import CustomInput,predictionPipeline

st.header('Loan Advisor')

st.markdown(f'<h2>Please select your gender</h2>', unsafe_allow_html=True)
gender = st.radio(
    "",
    options=["Male", "Female",]
)


st.markdown(f'<h2>Marital Status</h2>', unsafe_allow_html=True)
married = st.radio(
    "",  # Leave the label empty here, as it's already in the heading
    options=["Yes", "No"],
    key="marital_status"
)


st.markdown(f'<h2>Please indicate the number of individuals financially relying on you (0, 1, 2, etc.)</h2>', unsafe_allow_html=True)
dependents = st.selectbox(
    "",
    options=["0", "1", "2", "3+"]
)




st.markdown(f'<h2>What is your highest level of education</h2>', unsafe_allow_html=True)
education = st.radio(
    "",
    options=["Graduate", "Not Graduate"],
    key="education"
)

st.markdown(f'<h2>Are you self-employed</h2>', unsafe_allow_html=True)
self_employed = st.radio(
    "",
    options=["Yes", "No"],
    key="self_employed"
)


st.markdown(f'<h2>Give Your Financial Details</h2>', unsafe_allow_html=True)

number1 = st.number_input(f"Enter the Applicant Income")

try:
    ApplicantIncome = int(number1)
except ValueError:
    st.error("Please enter a valid integer.")



number2 = st.number_input("Enter the Coapplicant Income")

try:
    CoapplicantIncome = int(number2)
except ValueError:
    st.error("Please enter a valid integer.")


number2 = st.number_input("Enter the Loan amount")

try:
    
    LoanAmount = int(number2)
except ValueError:
    st.error("Please enter a valid integer.")


number4 = st.number_input("Select your preferred loan repayment term in months")

try:
    Loan_Amount_Term = int(number4)
except ValueError:
    st.error("Please enter a valid integer")


number5 = st.number_input("What is you credit History")

try:
    Credit_History = int(number5)
except ValueError:
    st.error("Please enter a valid integer")


 

st.markdown(f'<h2>In what type of area is your property located</h2>', unsafe_allow_html=True)
Property_Area = st.radio(
    "",
    options=["Semiurban", "Urban",'Rural'],
    key="Area"
)


def perform_prediction():
    obj = CustomInput(gender,married,dependents,education,self_employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
    featers = obj.get_df()
    predict_obj = predictionPipeline()
    price = predict_obj.predict(featers)
    return price



submit_button = st.button("Submit")

# If the submit button is clicked, call the my_function() function
if submit_button:
    if (perform_prediction() == "Y"):

        st.markdown(f'<h2>Congratulation You are eligible to apply for loan</h2>', unsafe_allow_html=True)

    else:
        st.markdown(f'<h2>sorry You are not eligible to apply for loan</h2>', unsafe_allow_html=True)

    