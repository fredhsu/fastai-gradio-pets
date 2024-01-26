from fastai.vision.all import load_learner, PILImage
import gradio as gr

learn = load_learner("export.pkl")

categories = ("Dog", "Cat")
labels = learn.dls.vocab
print(len(labels))


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


gr.Interface(fn=predict, inputs=gr.Image(), outputs=gr.Label()).launch(share=True)
