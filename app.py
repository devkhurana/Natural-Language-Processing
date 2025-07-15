from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

text_generator = pipeline("text-generation")
translator = pipeline("translation_en_to_fr")
zero_shot_classifier = pipeline("zero-shot-classification")
qa_pipeline = pipeline("question-answering")



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text_generation', methods=['GET', 'POST'])
def text_generation():
    result = None
    if request.method == 'POST':
        user_input = request.form['text_input']
        generated_text = text_generator(user_input, max_length=50)
        result = generated_text[0]['generated_text']  # Extracting the generated text
    return render_template('text_generation.html', result=result)

@app.route('/translation', methods=['GET', 'POST'])  # Corrected "methods"
def translation():
    result = None
    if request.method == 'POST':
        user_input = request.form['text_input']
        translated_text = translator(user_input)
        result = translated_text[0]['translation_text']  # Extracting the translation text
    return render_template('translation.html', result=result)

@app.route('/zero_shot_classification', methods=['GET', 'POST'])
def zero_shot_classification():
    result = None
    if request.method == 'POST':
        text_input = request.form['text_input']
        labels = request.form['labels'].split(',')  # Split labels by comma
        result = zero_shot_classifier(text_input, candidate_labels=labels)
    return render_template('zero_shot_classification.html', result=result)

@app.route('/question_answering', methods=['GET', 'POST'])
def question_answering():
    result = None
    if request.method == 'POST':
        context = request.form['context']
        question = request.form['question']
        result = qa_pipeline(question=question, context=context)
    return render_template('question_answering.html', result=result)

@app.route('/sentiment_analysis', methods=['GET'])
def sentiment_analysis():
    return render_template('under_construction.html')


if __name__ == '__main__':
    app.run(debug=True)
