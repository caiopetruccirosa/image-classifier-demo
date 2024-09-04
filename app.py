import streamlit as st
import plotly.express as px
import pandas as pd

from PIL import Image
from classifier import ImageClassifier

##################
# Set page config
##################
st.set_page_config(page_title="Classificador de Imagens", page_icon="👁️", layout="wide")

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

st.title("Demo: Classificação de Imagens")

# placeholder for the header
header_placeholder = st.empty()

selected_model = st.selectbox(
    "Escolha um modelo para classificar as imagens:",
    models.keys(),
    placeholder="Escolha um modelo..."
)

if selected_model is not None:    
    # assign the header title
    header_placeholder.header(f"Classificando com {selected_model} ({models[selected_model]})")

    image_classifier = load_classifier(models[selected_model])

    upload = st.file_uploader(label="Carregar imagem:", type=["png", "jpg", "jpeg"])
    css = """
    <style>
        div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button {{
            display: none;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>span {{
            visibility:hidden;
            font-size: 0px;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {{
            content:"{INSTRUCTIONS_TEXT}";
            visibility:visible;
            display:block;
            font-size: 16px;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>small {{
            visibility:hidden;
            font-size: 0px;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>small::before {{
            content:"{FILE_LIMITS}";
            visibility: visible;
            display:block;
            font-size: 14px;
        }}
    </style>
    """.format(
        INSTRUCTIONS_TEXT="Arraste e solte a imagem aqui ou clique aqui para carregar.",
        FILE_LIMITS="Limite de 200MB por arquivo • PNG, JPG, JPEG",
    )
    st.markdown(css, unsafe_allow_html=True)

    if upload:
        img = Image.open(upload).convert("RGB")

        scores = image_classifier.predict(img)
        scores_df = pd.DataFrame.from_dict(scores)
        
        prediction = scores_df['score'].idxmax()

        top_label = scores_df.loc[prediction, 'label']
        top_prob = scores_df.loc[prediction, 'score']

        top_5 = scores_df.sort_values(by="score").tail(5)
        top_5['score'] = top_5['score'] * 100

        fig = px.bar(top_5, x="score", y="label", orientation='h', title="Top 5 classes possíveis")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Classificação:")
            st.markdown(f"Classe \"**{top_label}**\" com {top_prob*100:.2f}% de probabilidade.")
            st.plotly_chart(fig, theme="streamlit")

        with col2:
            st.subheader("Imagem:")
            st.image(img, caption=f"Uploaded image: {upload.name}")