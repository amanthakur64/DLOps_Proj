import streamlit as st
import requests
from streamlit_lottie import st_lottie

def load_animation(url):
    r=requests.get(url)
    return r.json()




def app():
    st.title("Home")
    st.subheader("This is our DL Ops Project")
    st.title("Efficient Super Resolution Without Upsampling Using Deep Convolutional Networks")

    lottie=load_animation("https://assets6.lottiefiles.com/packages/lf20_hbcvqlsb.json")

    st_lottie(lottie,height=300,width=400,quality='high',key="tree",)

    st.write("Single image super-resolution (SISR) is a fundamental task in computer vision that involves generating a high-resolution (HR) image with high-frequency details from a degraded low-resolution (LR) image. However, it is an inherently challenging problem as there are multiple possible solutions for any given LR pixel, making it an underdetermined inverse problem. To overcome this challenge, prior information is used to constrain the solution space. Recent state-of-the-art methods have employed the example-based approach to learn this prior information. These methods either utilize the internal similarities within an image or learn mapping functions from external low- and high-resolution pairs of images. These external example-based methods can be used for generic image super-resolution or can be tailored to specific domains such as face hallucination, based on the provided training samples")
   