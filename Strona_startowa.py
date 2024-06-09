import streamlit as st

st.set_page_config(page_title="Main page", page_icon="🏠")

st.title("Metody Scoringowe i Techniki klasyfikacji danych")

st.divider()

st.write("""
    Witamy na naszej stronie, która jest poświęcona metodom scoringowym i technikom klasyfikacji danych.
    
    W menu po lewej stronie możesz wybrać różne strony, aby dowiedzieć się więcej o poszczególnych metodach oraz zobaczyć ich praktyczne zastosowania.
    """)

tab1, tab2, tab3 = st.tabs(['Weight of evidence','TAB2', 'TAB3'])

with tab1:
    from nauka.metodyScoringowe import woe