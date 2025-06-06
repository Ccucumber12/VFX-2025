import streamlit as st
from PIL import Image
from reinhard import *

slider = st.slider("Step", 1, 50, 25)

row1 = st.columns(3)
row2 = st.columns(3)

def reset_result_path():
    st.session_state.result_path = None

if 'result_path' not in st.session_state:
    reset_result_path()

source = row1[0].file_uploader("Source", type=["png", "jpg"], on_change=reset_result_path)
if source:
    row2[0].image(source)

reference = row1[1].file_uploader("Reference", type=["png", "jpg"], on_change=reset_result_path)
if reference:
    row2[1].image(reference)

row1[2].html("<p style='font-size: 14px'>Result</p>")

if st.button("Transfer", type="primary", disabled=(source == None or reference == None)):
    source_image = cv2.cvtColor(np.array(Image.open(source)), cv2.COLOR_RGB2BGR)
    reference_image = cv2.cvtColor(np.array(Image.open(reference)), cv2.COLOR_RGB2BGR)
    st.session_state.result_path = color_transfer_sequence(source_image, reference_image)

if st.session_state.result_path:
    source_name = os.path.splitext(source.name)[0]
    image_path = f"{st.session_state.result_path}/sequence/{slider-1}.jpg"
    with open(image_path, "rb") as image:
        row1[2].download_button(
            "Download picture",
            data=image,
            file_name=f"{source_name}_{slider}.jpg",
            mime="image/jpg",
            icon=":material/download:"
        )
    video_path = f"{st.session_state.result_path}/result.mp4"
    with open(video_path, "rb") as video:
        row1[2].download_button(
            "Download video",
            data=video,
            file_name=f"{source_name}.mp4",
            mime="video/mp4",
            icon=":material/download:"
        )
    row2[2].image(image_path)