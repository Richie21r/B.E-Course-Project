# helper.py
import streamlit as st

def show_predictions(results):
    for r in results:
        output_img = r.plot()  # draws predictions
        st.image(output_img, caption="Prediction", use_column_width=True)