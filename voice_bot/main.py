import gradio as gr

from utils.audio import process_audio_query, text_to_speech
from utils.process_PDF import get_chain


def respond_to_query(audio_file=None, pdf_file=None, text_query=None):
    """
    Xử lý file audio, PDF và văn bản người dùng nhập và tạo phản hồi dưới dạng văn bản và âm thanh.
    """
    audio_text = ""
    if audio_file:
        audio_text1 = process_audio_query(audio_file)
        audio_text = f"You said: {audio_text1}"
    else:
        audio_text1 = ""

    # Case 3: Xử lý văn bản người dùng nhập
    if text_query:
        user_input = f"You typed: {text_query}"
        user_input1 = text_query
    else:
        user_input = ""
        user_input1 = ""

    chain_response = ""
    folder_path = "./PDF_data/"
    input_query = "User wants to know about " + audio_text1 + " " + user_input1
    if pdf_file:
        chain_response = get_chain(input_query, folder_path, pdf_file)
    else:
        pdf_file = "notification.txt"
        chain_response = get_chain(input_query, folder_path, pdf_file)

    final_response = audio_text + "\n" + user_input + "\n" + chain_response + "\n"

    # Tạo âm thanh từ kết quả
    audio_path = text_to_speech(final_response)

    return final_response, audio_path


def gradio_interface(audio_file=None, text_query=None, pdf_file=None):
    # text_pdf, audio_pdf = response_pdf_input(pdf_file,  output_dir="output_audio")
    text, audio_path = respond_to_query(audio_file, pdf_file, text_query)
    # return text_pdf, audio_pdf, text, audio_path
    return text, audio_path


iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Audio(label="Upload your audio query:", type="filepath"),
        gr.Textbox(label="Enter your text query:", lines=2),
        gr.File(label="Upload your PDF file:"),
    ],
    outputs=[
        # gr.Textbox(label="Bot's Response from PDF processing"),  # Output 1
        #            gr.Audio(label="Generated Audio from PDF processing"),  # Output 2
        gr.Textbox(label="Bot's Response to Query"),  # Output 3
        gr.Audio(label="Generated Audio Response to Query"),  # Output 4
    ],
    title="Bot with Audio, PDF, and Text Input",
)

iface.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)
