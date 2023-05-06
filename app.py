import pickle
import streamlit as st
import pandas as pd
import sklearn
teams=['Chennai Super Kings', 'Deccan Chargers', 'Delhi Capitals',
       'Delhi Daredevils', 'Gujarat Titans', 'Kings XI Punjab',
       'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
       'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore',
       'Sunrisers Hyderabad']

Venue=['Arun Jaitley Stadium, Delhi', 'Barabati Stadium',
       'Brabourne Stadium', 'Buffalo Park',
       'De Beers Diamond Oval', 'Dr DY Patil Sports Academy, Mumbai',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'Dubai International Cricket Stadium', 'Eden Gardens',
       'Eden Gardens, Kolkata', 'Feroz Shah Kotla',
       'Himachal Pradesh Cricket Association Stadium',
       'Holkar Cricket Stadium', 'JSCA International Stadium Complex',
       'Kingsmead', 'M Chinnaswamy Stadium',
       'MA Chidambaram Stadium, Chepauk, Chennai',
       'Narendra Modi Stadium, Ahmedabad', 'New Wanderers Stadium',
       'Newlands', 'OUTsurance Oval',
       'Punjab Cricket Association IS Bindra Stadium, Mohali',
       'Rajiv Gandhi International Stadium, Uppal, Uppal',
       'Sardar Patel Stadium, Motera', 'Sawai Mansingh Stadium',
       'Shaheed Veer Narayan Singh International Stadium',
       'Sharjah Cricket Stadium', 'Sheikh Zayed Stadium',
       "St George's Park", 'Subrata Roy Sahara Stadium',
       'Subrata Roy Sahara Stadium, Pune', 'SuperSport Park',
       'Vidarbha Cricket Association Stadium, Jamtha',
       'Wankhede Stadium, Mumbai, Mumbai',
       'Zayed Cricket Stadium, Abu Dhabi']

with open('modelx.pkl', 'rb') as file:
        pipe, le, LE = pickle.load(file)
st.title('Cricket Data Analyser and Predictor ')
col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox("Select The batting Team",sorted(teams))
with col1:
    bowling_team=st.selectbox("Select The bowling Team",sorted(teams))

venue=st.selectbox('Select Venue',Venue)


batting_team_p = le.transform([batting_team])[0]
bowling_team_p = le.transform([bowling_team])[0]
venue_p = LE.transform([venue])[0]

target=st.number_input('Target')


col3,col4,col5=st.columns(3)

with col3:
       score=st.number_input('Score')
with col4:
       overs=st.number_input('Overs completed')
with col5:
       wickets=st.number_input('Wickets Out')

if st.button('Predict'):
       runs_left=target-score
       balls_left=120-overs*6
       wickets_left=10-wickets
       input_df=pd.DataFrame({'BattingTeam':[batting_team_p],'BowlingTeam':[bowling_team_p],'Venue':[venue_p],'Runs_Left':[runs_left],'Balls_Left':[balls_left],'Wickets_Left':[wickets_left]})
       result=pipe.predict_proba(input_df)
       loss=result[0][0]
       win=result[0][1]
       st.header(batting_team+"-"+str(round(win*100))+"%")
       st.header(bowling_team + "-" + str(round(loss*100)) + "%")