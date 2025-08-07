import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

#Title
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing Price Prediction')

st.image('https://images.squarespace-cdn.com/content/v1/65a8583b3f2bb32732bff587/63ff3986-3c95-4422-bdaa-6a373b71140d/Custom-Luxury-Home-Dallas.jpg')

st.header('Model of housing prices to predict median house values in California',divider = True)

st.sidebar.title('Select House Features 🏘️ ')

st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20220902/pngtree-rising-house-prices-concept-3d-illustration-sale-construction-render-photo-image_47623692.jpg')

#read_data
temp_df = pd.read_csv('california.csv')
random.seed(13)

all_values = []
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])
   
    var = st.sidebar.slider(f'Select {i} value',int(min_value), int(max_value), random.randint(int(min_value),int(max_value)))
    all_values.append(var)

ss = StandardScaler()

ss.fit(temp_df[col])
final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting price!!!')
place = st.empty()
place.image('https://img.pikbest.com/png-images/20190918/cartoon-snail-loading-loading-gif-animation_2734139.png!bw700',width = 100)

if price>0:
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
        
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollar'
    placeholder.empty()
    place.empty()
    #st.subheader(body)

    st.success(body)

else:
    body = 'Invalid House Features Values'
    st.warning(body)
    
