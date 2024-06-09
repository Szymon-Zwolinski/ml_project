import streamlit as st

st.title('Metody scoringowe')

tab1, tab2, tab3 = st.tabs(['Weight of evidence','TAB2', 'TAB3'])

with tab1:
    from pages.nauka.metodyScoringowe import woe