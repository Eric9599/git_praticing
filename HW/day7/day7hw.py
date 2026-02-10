import pdfplumber


file_list = ["1.pdf", "2.pdf", "3.pdf", "4,png", "5.docx"]
file_path = "HW"

with pdfplumber.open(file_path + file_list) as pdf:
    if pdf is ".png":
        for page in pdf.pages:
            text = page.extract_text()
            print(text)
