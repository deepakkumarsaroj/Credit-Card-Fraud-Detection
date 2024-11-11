import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/fraud_detection_model.pkl')

# Define the main function for the Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    # Instruction for user input
    st.header("Input Transaction Details")
    st.write("Enter all features as a comma-separated list (e.g., -1.5, 1.2, -0.8, ..., 0.4, 0.7)")
    
    # Input field for all features as a comma-separated string
    input_df = st.text_input('Input All features:')
    
    # Button to submit input and get prediction
    submit = st.button("Submit")
    
    if submit:
        try:
            # Split the input string into a list and convert to float
            input_df_lst = [float(i) for i in input_df.split(',')]
            
            # Check if the input list has the correct number of features (30 in this example)
            if len(input_df_lst) != 30:
                st.error("Please enter exactly 30 values separated by commas.")
            else:
                # Define column names matching those used during training
                column_names = [
                    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "scaled_amount", "scaled_time"
                ]

                # Convert the input list into a DataFrame with proper column names
                data = pd.DataFrame([input_df_lst], columns=column_names)
                
                # Make a prediction
                prediction = model.predict(data)
                
                # Display the prediction result
                if prediction[0] == 1:
                    st.error("Warning: Potential Fraud Detected!")
                else:
                    st.success("Transaction is not fraudulent.")
        
        except ValueError:
            st.error("Please enter valid numerical values separated by commas.")

if __name__ == "__main__":
    main()
