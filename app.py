import gradio as gr
import numpy as np
import joblib

# Load pipeline/model
loaded = joblib.load('customerpipeline.joblib')
pipeline = loaded if not isinstance(loaded, (tuple, list)) else loaded[0]  # Adjust index if needed!

def predict_cluster(id_num, sex, marital_status, age, education, income, occupation, settlement_size):
    # 8 - feature input (order must match training)
    input_data = np.array([[id_num, sex, marital_status, age, education, income, occupation, settlement_size]])
    cluster = pipeline.predict(input_data)
    return f"Predicted Cluster/Segment: {int(cluster[0])}"

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    gr.Markdown(
        """
        # 🧑‍💻 Customer Segmentation & Cluster Prediction App  
        Enter all features **as numbers** (including a dummy ID!) in the SAME ORDER as pipeline training:
        ID, Sex, Marital Status, Age, Education, Income, Occupation, Settlement Size.
        """
    )
    with gr.Row():
        with gr.Column():
            id_num = gr.Number(label="ID (can be any integer)", value=0)
            sex = gr.Radio([0, 1], label="Sex (0=Male, 1=Female)", value=0)
            marital_status = gr.Radio([0, 1], label="Marital Status (0=Unmarried, 1=Married)", value=0)
            age = gr.Slider(18, 80, label="Age", value=25)
            education = gr.Radio([0, 1, 2, 3], label="Education (0-3)", value=1)
            income = gr.Number(label="Income (Annual)", value=50000)
            occupation = gr.Radio([0, 1, 2], label="Occupation (0-2)", value=1)
            settlement_size = gr.Radio([0, 1, 2], label="Settlement Size (0-2)", value=1)
            submit_btn = gr.Button("Predict Segment")

        with gr.Column():
            output = gr.Textbox(label="Prediction Result", lines=2)
            gr.Info("Model automatically performs preprocessing and segmentation.")

    submit_btn.click(
        fn=predict_cluster,
        inputs=[
            id_num, sex, marital_status, age, education, income, occupation, settlement_size
        ],
        outputs=output,
    )

if __name__ == "__main__":
    app.launch()
