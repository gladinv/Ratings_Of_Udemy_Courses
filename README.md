# Ratings_Of_Udemy_Courses

Pandas, Scikit-learn, Streamlit

Sure! Here's a step-by-step walkthrough of the code:

1. Import the necessary libraries:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import streamlit as st
```

2. Load the dataset from a CSV file:
```python
Udemy_df_CLEAN = pd.read_csv(r'C:\Users\User\Desktop\GUVI Final Project\RATINGS_OF_UDEMY_COURSES_CLEANED_DATA.csv')
```

3. Split the data into features (X) and target (y):
```python
X = Udemy_df_CLEAN[['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'content_duration', 'level', 'subject']]
y = Udemy_df_CLEAN['Rating']
```

4. Split the data into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. Preprocess the data using pipelines and column transformers:
```python
numeric_features = ['price', 'num_subscribers', 'num_reviews', 'num_lectures', 'content_duration']
numeric_transformer = make_pipeline(StandardScaler(), PCA(n_components=4))

categorical_features = ['level', 'subject']
categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features)
)
```

6. Create a pipeline to preprocess the data and train the Random Forest Regressor model:
```python
pipe = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100, random_state=42))
```

7. Train the model using the training data:
```python
pipe.fit(X_train, y_train)
```

8. Make predictions on the test data:
```python
y_pred = pipe.predict(X_test)
```

9. Evaluate the model performance by calculating the R-squared score:
```python
print('R-squared score:', r2_score(y_test, y_pred))
```

10. Set up a Streamlit application:
```python
st.title('Udemy Rating Prediction')
st.write('Enter the course details to predict its rating:')
```

11. Get user input for course details using sliders and select boxes:
```python
price = st.slider('Price', min_value=0, max_value=200, value=100, step=1)
num_subscribers = st.slider('Number of Subscribers', min_value=0, max_value=300000, value=5000, step=10)
num_reviews = st.slider('Number of Reviews', min_value=0, max_value=30000, value=100, step=1)
num_lectures = st.slider('Number of Lectures', min_value=0, max_value=800, value=50, step=1)
content_duration = st.slider('Content Duration (in minutes)', min_value=0, max_value=100, value=35, step=1)
level = st.selectbox('Course Level', ['All Levels', 'Beginner Level', 'Intermediate Level', 'Expert Level'])
subject = st.selectbox('Course Subject', ['Business Finance', 'Graphic Design', 'Musical Instruments', 'Web Development'])
```

12. Create a DataFrame to store the user input:
```python
input_df = pd.DataFrame({
    'price': [price],
    'num_subscribers': [num_subscribers],
    'num_reviews': [num_reviews],
    'num_lectures': [num_lectures],
    'content_duration': [content_duration],
    'level': [level],
    'subject': [subject]
})
```

13. Make a prediction using the preprocessed input data:
```python
prediction = pipe.predict(input_df)[0]
```

14. Display the predicted rating using Streamlit:
```python
st.write('Predicted Rating:', prediction)
```

That's the step-by-step walkthrough of the provided code. It loads a dataset, preprocesses the data, trains a Random Forest Regressor model, and creates a Streamlit application for predicting the rating of Udemy courses based on user input.
