from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,log_loss
import streamlit.components.v1 as stc


# File Processing Pkgs
import pandas as pd
import docx2txt
from PIL import Image 


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img 


def main():
    st.title("EZ-FOTO SYSTEM")
    st.sidebar.title("EZ-FOTO SYSTEM")


    Data = ["","Dataset"]
    choice = st.sidebar.selectbox("Uploding Data",Data)
    if choice == "Dataset":
        #st.subheader("Data")
        image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
        if image_file is not None:
            img = load_image(image_file)
            #st.image(img)
            st.image(img,caption='Actual Image',use_column_width=True)


if __name__ == '__main__':
    main()
st.sidebar.subheader("Choose Model")
classifier = st.sidebar.selectbox("Image Generation", ("","SmoothBigGAN"))


if st.sidebar.button('Generate'):
    image=Image.open('./Images/Generated Image.jpg')
    st.image(image,caption='Image generated with high quality and diversity after 15k runs',use_column_width=True)
    #st.write ('Image geneated with high quality and diversity')



