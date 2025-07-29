
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import io

# --- 1. Load the Trained Model and Encoders ---
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please ensure it's in the correct directory.")
    st.stop()

# --- 2. Define Mappings, Columns, and Defaults ---
original_columns = [
    'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country'
]

# Text-to-number mappings
workclass_map = {'?': 3, 'Federal-gov': 1, 'Local-gov': 2, 'Never-worked': 0, 'Private': 4, 'Self-emp-inc': 5, 'Self-emp-not-inc': 6, 'State-gov': 7, 'Without-pay': 8, 'Others': 3}
marital_status_map = {'Married-civ-spouse': 2, 'Never-married': 4, 'Divorced': 0, 'Separated': 5, 'Widowed': 6, 'Married-spouse-absent': 3, 'Married-AF-spouse': 1}
occupation_map = {'?': 9, 'Adm-clerical': 1, 'Armed-Forces': 0, 'Craft-repair': 3, 'Exec-managerial': 4, 'Farming-fishing': 5, 'Handlers-cleaners': 6, 'Machine-op-inspct': 7, 'Other-service': 8, 'Priv-house-serv': 10, 'Prof-specialty': 10, 'Protective-serv': 11, 'Sales': 12, 'Tech-support': 13, 'Transport-moving': 14, 'Others': 9}
relationship_map = {'Husband': 0, 'Not-in-family': 1, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5, 'Other-relative': 2}
race_map = {'White': 4, 'Black': 2, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 3}
gender_map = {'Male': 1, 'Female': 0}
native_country_list = ['?', 'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico', 'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia']
le_country = LabelEncoder().fit(native_country_list)

# Define columns for cleaning
categorical_cols = ['workclass', 'occupation', 'marital-status', 'relationship', 'race', 'gender']

# --- 3. Build the Streamlit User Interface ---
st.set_page_config(page_title="Income Classification", layout="wide")
st.title("🧑‍💼 Adult Income Classification App")
st.markdown("Predict income for a single individual or upload a **CSV or Excel file** for batch predictions.")

tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction"])

# --- TAB 1: SINGLE PREDICTION (Restored and Fully Functional) ---
with tab1:
    st.header("Enter Details for a Single Individual")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Personal & Work Details")
        age = st.slider("Age", 17, 90, 35, key="age_single")
        workclass = st.selectbox("Work Class", list(workclass_map.keys()), key="workclass_single")
        occupation = st.selectbox("Occupation", list(occupation_map.keys()), key="occupation_single")
        hours_per_week = st.slider("Hours Worked per Week", 1, 99, 40, key="hours_single")
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1, max_value=1500000, value=150000, key="fnlwgt_single", help="A weight assigned by the census bureau.")

    with right_col:
        st.subheader("Education & Relationship")
        educational_num = st.slider("Education Level (Numeric)", 1, 16, 10, key="edu_single", help="1=Preschool, 16=Doctorate")
        marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()), key="marital_single")
        relationship = st.selectbox("Relationship Status", list(relationship_map.keys()), key="rel_single")
        race = st.selectbox("Race", list(race_map.keys()), key="race_single")
        gender = st.selectbox("Gender", list(gender_map.keys()), key="gender_single")

    st.subheader("Financial Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0, key="gain_single")
    with c2:
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=4356, value=0, key="loss_single")
    with c3:
        native_country = st.selectbox("Native Country", native_country_list, key="country_single")

    if st.button("**Predict Single Income**", type="primary"):
        input_data = {
            'age': age,
            'workclass': workclass_map[workclass],
            'fnlwgt': fnlwgt,
            'educational-num': educational_num,
            'marital-status': marital_status_map[marital_status],
            'occupation': occupation_map[occupation],
            'relationship': relationship_map[relationship],
            'race': race_map[race],
            'gender': gender_map[gender],
            'capital-gain': capital_gain,
            'capital-loss': capital_loss,
            'hours-per-week': hours_per_week,
            'native-country': le_country.transform([native_country])[0]
        }
        input_df = pd.DataFrame([input_data], columns=original_columns)
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.markdown("---")
        st.subheader("🔮 Prediction Result")
        income_class = prediction[0]
        if income_class == '<=50K':
            st.success(f"**Predicted Income Class: {income_class}**")
        else:
            st.warning(f"**Predicted Income Class: {income_class}**")
        st.write(f"**Confidence:**")
        st.write(f"- Probability of `<=50K`: **{prediction_proba[0][0]:.2%}**")
        st.write(f"- Probability of `>50K`: **{prediction_proba[0][1]:.2%}**")

# --- TAB 2: BATCH PREDICTION (With NaN row removal) ---
with tab2:
    st.header("Upload a CSV or Excel File for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                batch_df = pd.read_csv(uploaded_file, skipinitialspace=True)
            else:
                batch_df = pd.read_excel(uploaded_file)

            st.write("### 1. Uploaded Data Preview")
            st.dataframe(batch_df.head())

            st.write("### 2. Processing Data...")
            processed_df = batch_df.copy()
            processed_df['original_index'] = processed_df.index

            for col in categorical_cols:
                processed_df[col] = processed_df[col].fillna('?').astype(str).str.strip()
            processed_df['native-country'] = processed_df['native-country'].fillna('?').astype(str).str.strip()

            processed_df['workclass'] = processed_df['workclass'].map(workclass_map)
            processed_df['marital-status'] = processed_df['marital-status'].map(marital_status_map)
            processed_df['occupation'] = processed_df['occupation'].map(occupation_map)
            processed_df['relationship'] = processed_df['relationship'].map(relationship_map)
            processed_df['race'] = processed_df['race'].map(race_map)
            processed_df['gender'] = processed_df['gender'].map(gender_map)
            processed_df['native-country'] = processed_df['native-country'].apply(lambda x: x if x in le_country.classes_ else '?')
            processed_df['native-country'] = le_country.transform(processed_df['native-country'])

            initial_rows = len(processed_df)
            processed_df.dropna(inplace=True)
            final_rows = len(processed_df)

            if final_rows < initial_rows:
                st.warning(f"**Warning:** {initial_rows - final_rows} rows were removed because they contained invalid categories or missing numerical data.")

            if processed_df.empty:
                st.error("No valid data left to predict after cleaning. Please check your file.")
            else:
                st.write("### 3. Making Predictions on Clean Data...")
                predict_data = processed_df[original_columns]
                predictions = model.predict(predict_data)

                results_df = batch_df.loc[processed_df['original_index']].copy()
                results_df['Predicted_Income'] = predictions

                st.write("### ✅ Prediction Results (on valid rows only)")
                st.dataframe(results_df)

                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_output = convert_df_to_csv(results_df)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_output,
                    file_name="predicted_income_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Processing failed. Please ensure your file has the correct column headers (e.g., 'age', 'workclass', etc.).") 
