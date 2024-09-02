import streamlit as st
import plotly.express as px
import pandas as pd

from PIL import Image
from classifier import ImageClassifier

##################
# Set page config
##################
st.set_page_config(page_title="Image Classifier Demo", page_icon="👁️", layout="wide")

###################
# Helper functions 
###################

@st.cache_resource
def load_classifier(model_card):
    image_classifier = ImageClassifier(model_card)
    return image_classifier

######################
#  Global Variables
######################

models = {
    "ResNet18": "microsoft/resnet-18",
    "ResNet50": "microsoft/resnet-50",
    "Vision Transformer (Patch 16x16)": "google/vit-base-patch16-384",
    "Vision Transformer (Patch 32x32)": "google/vit-base-patch32-384",
}

##########
# App GUI
##########

st.title("Demo: Image Classification")

selected_model = st.selectbox(
    "Select a model for image classification:",
    models.keys(),
    placeholder="Select a model..."
)

if selected_model is not None:
    image_classifier = load_classifier(models[selected_model])
    
    st.header(f"Classifying with {selected_model} ({image_classifier.model.config._name_or_path})")

    upload = st.file_uploader(label="Upload image:", type=["png", "jpg", "jpeg"])

    if upload:
        img = Image.open(upload)

        scores = image_classifier.predict(img)
        scores_df = pd.DataFrame.from_dict(scores)

        top_5 = scores_df.sort_values(by="score", ascending=False).head(5)
        top_prob = top_5.iloc[0]['score']

        fig = px.bar(top_5, x="score", y="label", orientation='h')

        st.subheader(f"Predicted label:")
        st.markdown(f"**{top_5.iloc[0]['label']}** with {top_prob*100:.2f}% certainty.")

        st.subheader("Predicted top 5 probabilities:")
        st.plotly_chart(fig, theme="streamlit")

        st.subheader("Image:")
        st.image(img, caption=f"Uploaded image: {upload.name}")