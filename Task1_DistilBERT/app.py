import streamlit as st
from PIL import Image

from predict import classify_text
from imagecaption import generate_caption
from database import create_database, save_result, get_results


create_database()

st.title("Toxic Content Classification (DistilBERT + LoRA)")

st.write(
    "Classify text or images into a toxic content category."
)

text_tab, image_tab, database_tab = st.tabs(
    [
        "📝 Text Classification",
        "🖼️ Image Classification",
        "📊 Database"
    ]
)


# Text tab
with text_tab:

    st.subheader("Classify Text")

    text = st.text_area(
        "Enter your text:",
        placeholder="Write your text here..."
    )

    if st.button("Classify Text"):

        if text.strip():

            result = classify_text(text)

            st.success(f"Classification: {result}")

            save_result(
                "text",
                text,
                result
            )

        else:

            st.warning("Please enter some text.")


# Image tab
with image_tab:

    st.subheader("Classify Image")

    uploaded_image = st.file_uploader(
        "Upload an image:",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Classify Image"):

            caption = generate_caption(image)

            result = classify_text(caption)

            st.write(f"Caption: {caption}")

            st.success(f"Classification: {result}")

            save_result(
                "image",
                caption,
                result
            )


# Database tab
with database_tab:

    st.subheader("Classification History")

    results = get_results()

    if results:

        data = []

        for row in results:

            data.append({
                "timestamp": row[0],
                "type": row[1],
                "input": row[2],
                "classification": row[3]
            })

        st.dataframe(
            data,
            use_container_width=True
        )

    else:

        st.info("No classification results yet.")