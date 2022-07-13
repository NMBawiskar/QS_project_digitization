from qs_ocr import ExtractToJson
import os
import re


def find_files_recursively(dir_path):
    files_to_process = list()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
                files_to_process.append(os.path.join(root, file))
    return files_to_process


input_dir = r"C:\Users\Asus\Documents\Digitization\Reports\extracted_text_files"
file_to_process = find_files_recursively(input_dir)

for file in file_to_process:
    print(file)
    file_dict = eval(open(file, "r").read())
    ejson = ExtractToJson(file_directory='Reports/extracted_text_files')
    output_json_path = ejson.make_predictions(text=file_dict, file_name=re.sub(r"^\W+|\W+$", "", file.replace(input_dir, "")), return_json_path=True)
