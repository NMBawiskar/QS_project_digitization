from flask import Flask, request, Response, render_template, send_from_directory, jsonify
from flask_mysqldb import MySQL
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint

import database_credentials
import os
import json
import time
from qs_ocr import ConvertToTxt, ExtractToJson
import tkinter as tk
import tkinter.messagebox
from subprocess import check_output
import requests
from configparser import ConfigParser
from bs4 import BeautifulSoup
import sys
import re
# import difflib as dl


app = Flask(__name__)
app.config['MYSQL_HOST'] = database_credentials.host
app.config['MYSQL_USER'] = database_credentials.user
app.config['MYSQL_PASSWORD'] = database_credentials.password
app.config['MYSQL_DB'] = database_credentials.database
app.config['MYSQL_PORT'] = database_credentials.port
app.config['MYSQL_CONNECT_TIMEOUT'] = 28800

app.secret_key = 'SecurityIsAnIllusion'
mysql_connector = MySQL(app)

allTags = ['CAT', 'TST', 'RES', 'REF', 'UNT', 'MTD', 'PER', 'AGE', 'SEX', 'DTE', 'DCN', 'DPR', 'PGE', 'O']
ALLOWED_FILE_EXTENSIONS = ['pdf']
configuration = ConfigParser()
configuration.read('config.ini')

TAG_METADATA = [{'PER': 'Patient\'s Name',
                 'AGE': 'Age',
                 'SEX': 'Sex',
                 'DTE': 'Date',
                 'DPR': 'Doctor\'s Name',
                 'DCN': 'Diagnostic Center'},
                {'TST': 'Test',
                 'RES': 'Result',
                 'REF': 'Reference Range',
                 'UNT': 'Unit',
                 'MTD': 'Method',
                 'CAT': 'Category'}]
TAG_METADATA_COLOURED = {'TST': {'colour': 'blue'}, 'RES': {'colour': 'green'}, 'UNT': {'colour': 'yellow'}, 'REF': {'colour': 'purple'}, 'MTD': {'colour': 'grey'}}


SWAGGER_URL = '/api-docs'
API_URL = '/api/docs'


def insert_into_db(report_name, report_path, page_count, time_to_process, status="Digitized"):
    print(f"Inserting {report_name} into DB...")
    curr_time = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor = mysql_connector.connection.cursor()
    print("INSERT INTO processed_reports VALUES ({0}, '{1}', '{2}', {3}, {4}, {5}, '{6}', '{7}', '{8}')".format(0, report_name.replace('\\', '/'), report_path.replace('\\', '/'), page_count, str(time_to_process), 0, status, curr_time, curr_time))
    cursor.execute("INSERT INTO processed_reports VALUES ({0}, '{1}', '{2}', {3}, {4}, {5}, '{6}', '{7}', '{8}')".format(0, report_name.replace('\\', '/'), report_path.replace('\\', '/'), page_count, str(time_to_process), 0, status, curr_time, curr_time))
    cursor.close()


def select_from_db(field=None, value=None):
    cursor = mysql_connector.connection.cursor()
    if field is None or value is None:
        sql = "SELECT report_name, report_path, page_count, time_to_process, qc_time, status, date_created, date_modified FROM processed_reports ORDER BY date_created DESC"
    else:
        sql = f"SELECT report_name, report_path, page_count, time_to_process, qc_time, status, date_created, date_modified FROM processed_reports WHERE {field}='{value}' ORDER BY date_created DESC"
    print(sql)
    cursor.execute(sql)
    rows = cursor.fetchall()
    if type(rows) != list:
        rows = list(rows)
    cursor.close()
    return rows


def update_in_db(field, value, where_field, where_value):
    cursor = mysql_connector.connection.cursor()
    cursor.execute("UPDATE processed_reports SET {0}='{1}' WHERE {2}='{3}'".format(field, value, where_field, where_value))
    cursor.close()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[-1].lower() in ALLOWED_FILE_EXTENSIONS


def find_files_recursively(dir_path):
    files_to_process = list()
    invalid_files = list()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if allowed_file(file):
                files_to_process.append(os.path.join(root, file))
            else:
                invalid_files.append(file)
    return files_to_process, invalid_files


@app.route('/process_file', methods={"POST"})
def process_file():
    """
        Digitize a report.
        ---
        tags:
          - Digitization
        consumes: multipart/form-data
        parameters:
          - in: formData
            name: file
            description: PDF file object
            type: file
            required: true
        responses:
          200:
            description: Digitized JSON for the file uploaded.
    """
    if 'file' in request.files:
        pdf_file = request.files['file']
        # if len(select_from_db('report_name', ".".join(pdf_file.filename.rsplit('.', 1)[:-1]).replace("\\", "/"))) > 0:
        #     return Response(json.dumps(dict(message="File has previously been processed.")), status=200)
        if not allowed_file(pdf_file.filename):
            return Response(json.dumps(dict(message="Invalid file extension.")), status=200)
        else:
            ctxt = ConvertToTxt(directory_path=r'Reports', file=pdf_file, thread_limit=1)
            ejson = ExtractToJson(file_directory='Reports/extracted_text_files')
            files = ctxt.get_files_by_name_or_count(file_name=pdf_file.filename)
            status = "Failed to process"
            output_json_path, page_count = "-", -1
            start = time.time()
            try:
                filename, txt, page_count, mer_output = ctxt.process_files(files[0], save_to_txt=True, move_file=True, return_page_count=True)
                output_json_path = ejson.make_predictions(text=txt, file_name=filename + '.txt', return_json_path=True)
                output_json = ejson.make_predictions(text=txt, file_name=filename + '.txt', return_json=True)
                status = "Digitized"
                return Response(json.dumps(output_json), status=200)
            except PermissionError as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                return Response(json.dumps(dict(message=e)), status=200)
            finally:
                end = time.time()
                insert_into_db(pdf_file.filename.rsplit('.', 1)[0], output_json_path, page_count, round(end - start, 2), status)
                print(f"Time taken for file {pdf_file.filename} was {round(end - start, 2)} secs.")
    elif 'directory_path' in request.form.keys():
        pdf_files, invalid_files = find_files_recursively(request.form['directory_path'])
        for pdf_file in pdf_files:
            print(os.path.relpath(".".join(pdf_file.rsplit('.', 1)[:-1]), request.form['directory_path']).replace("\\", "/"))
            if len(select_from_db('report_name', os.path.relpath(".".join(pdf_file.rsplit('.', 1)[:-1]), request.form['directory_path']).replace("\\", "/"))) > 0:
                invalid_files.append(os.path.relpath(".".join(pdf_file.rsplit('.', 1)[:-1]), request.form['directory_path']).replace("\\", "/"))
                continue
            status = "Failed to process"
            output_json_path, page_count = "-", -1
            start = time.time()
            try:
                try:
                    ctxt = ConvertToTxt(directory_path=r'Reports', file=pdf_file, thread_limit=2, base_path=request.form['directory_path'])
                    ejson = ExtractToJson(file_directory='Reports/extracted_text_files')
                    files = ctxt.get_files_by_name_or_count(file_name=pdf_file.rsplit('/', 1)[-1])
                    filename, txt, page_count, mer_output = ctxt.process_files(files, save_to_txt=True, move_file=True, return_page_count=True)
                    output_json_path = ejson.make_predictions(text=txt, file_name=filename + '.txt', return_json_path=True)
                    status = "Digitized"
                except Exception:
                    invalid_files.append(pdf_file)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                finally:
                    end = time.time()
                    insert_into_db(os.path.relpath(pdf_file, request.form['directory_path']).rsplit('.', 1)[0], output_json_path, page_count, round(end - start, 2), status)
                    print(f"Time taken for file {os.path.relpath(pdf_file, request.form['directory_path'])} was {round(end - start, 2)} secs.")
            except Exception as e:
                print(e)
        return Response(json.dumps(
            dict(message="Processed {0}/{1} files".format((len(pdf_files) - len(invalid_files)), len(pdf_files)))))
    else:
        return Response(json.dumps(dict(message="Please provide either a file or a directory path")), status=200)


@app.route('/<path:file_type>/<path:file_name>')
def send_file(file_type, file_name):
    directory = ''
    if file_type == 'pdf':
        directory = 'Reports/processed_reports'
    elif file_type == 'txt':
        directory = 'Reports/extracted_text_files'
    elif file_type == 'html':
        directory = 'Reports/extracted_html_files'
    elif file_type == 'report':
        directory = f'extracted_png/{file_name.rsplit(".", 1)[0].rsplit("-", 1)[0]}'
        file_name = file_name.rsplit('/', 1)[-1]
    elif file_type == 'report_ocr':
        directory = f'extracted_png/{file_name.rsplit(".", 1)[0].rsplit("-", 1)[0]}_ocr'
        file_name = file_name.rsplit('/', 1)[-1]
    return send_from_directory(directory, filename=file_name)


@app.route('/', methods={"POST", "GET"})
def index():
    if request.method == 'GET':
        return render_template("login.html")
    else:
        return Response(status=200)


@app.route('/dashboard-<path:file_name>')
def dashboard(file_name):
    if request.method == 'GET':
        files = select_from_db('report_name', file_name)
        if files is not None and len(files) != 0:
            if files[0][2] == 0:
                page_count = 1
            else:
                page_count = files[0][2]
        else:
            page_count = 1
        return render_template("form.html", tags_list=TAG_METADATA, page_count=page_count, file_name=file_name)


@app.route('/update-status', methods={"POST"})
def update_status():
    try:
        filename = request.get_json()['filename']
        status = request.get_json()['status']
        update_in_db('status', status, 'report_name', filename)
        return Response(json.dumps({"message": "Data updated successfully"}), status=200)
    except Exception as e:
        return Response(dict(message=e), status=200)


@app.route('/qc', methods={"POST", "GET"})
def qc():
    if request.method == 'GET':
        # pdfs = os.listdir('Reports/processed_reports')
        # pdfs = [pdf.rsplit('.', 1)[0] for pdf in pdfs]
        # txts = os.listdir('Reports/extracted_text_files')
        # txts = [txt.rsplit('.', 1)[0] for txt in txts]
        # jsons = os.listdir('Reports/extracted_json_files')
        # jsons = [json.rsplit('.', 1)[0] for json in jsons]
        #
        # reports = list(set(pdfs).intersection(set(txts)).intersection(set(jsons)))

        return render_template("report_listing.html", rows=select_from_db())
    else:
        start = time.time()
        pdf_file = request.files['file']
        ctxt = ConvertToTxt(directory_path=r'Reports', file=pdf_file, thread_limit=1)
        ejson = ExtractToJson(file_directory='Reports/extracted_text_files')
        files = ctxt.get_files_by_name_or_count(file_name=pdf_file.filename)
        filename, txt, mer_output = ctxt.process_files(files, save_to_txt=True, move_file=True)
        output_json_path = ejson.make_predictions(text=txt, file_name=filename + '.txt', return_json_path=True)
        end = time.time()
        insert_into_db(filename.rsplit('.', 1)[0], output_json_path, 0, round(end - start, 2))

        return render_template("report_listing.html", rows=select_from_db())


@app.route('/qc-<path:file_name>')
def qc_report(file_name):
    output_json = json.loads(open('Reports/extracted_json_files/' + file_name + '.json').read())
    return render_template('qc_form.html', filename=file_name, data=output_json, tags=TAG_METADATA_COLOURED,
                           pdf_name=file_name + '.pdf', user_name='Admin')


@app.route('/submit-qc-data', methods={"POST"})
def submit_qc_data():
    filename = request.get_json()['filename']
    qc_data = request.get_json()['qcData']
    timeElapsed = request.get_json()['timeElapsed']

    with open(os.path.join("Reports/extracted_html_files", filename) + ".json", 'w') as f:
        json.dump(qc_data, f, indent=4)

    update_in_db('date_modified', time.strftime('%Y-%m-%d %H:%M:%S'), 'report_name', filename)
    update_in_db('qc_time', timeElapsed, 'report_name', filename)
    update_in_db('status', "QC Completed", 'report_name', filename)

    return Response(json.dumps({"message": "Data updated successfully"}), status=200)


@app.route('/create-dataset')
def create_dataset():
    total_count = 0
    error_count = 0
    total_word_count = 0
    spelling_error_word_count = 0
    tagging_error_word_count = 0
    dataset_lines = list()
    qc_files = [os.path.join("Reports/extracted_html_files", file[0]) + ".json" for file in select_from_db('status', 'QC Completed')]
    og_files = [os.path.join("Reports/extracted_html_files_bak", file[0]) + ".json" for file in select_from_db('status', 'QC Completed')]
    for qc_file, og_file in zip(qc_files, og_files):
        qc_data = json.loads(open(qc_file, 'r').read())
        og_data = json.loads(open(og_file, 'r').read())
        assert (og_data.keys() == qc_data.keys())
        for page_idx, qc_page in qc_data.items():
            og_page = og_data[page_idx]
            for qc_line, og_line in zip(qc_page.splitlines(), og_page.splitlines()):
                qc_soup = BeautifulSoup(re.sub(r"\s+", " ", qc_line), "html.parser")
                og_soup = BeautifulSoup(re.sub(r"\s+", " ", og_line), "html.parser")
                total_count += 1
                total_word_count += len(og_soup.getText().strip().split())
                if qc_soup != og_soup:
                    # print(og_soup.getText().strip().split(), qc_soup.getText().strip().split())
                    # diff_items = dl.ndiff(og_soup.getText().strip().split(), qc_soup.getText().strip().split())
                    # for item in diff_items:
                    #     print(item)
                    for pair in zip(og_soup.getText().strip().split(), qc_soup.getText().strip().split()):
                        if pair[0] != pair[1]:
                            spelling_error_word_count += 1
                    error_count += 1
                    final_sentence = ""
                    tagged_words = ""
                    tags = ""
                    if len(qc_soup.find_all('mark')) > 0:
                        for m_tag in qc_soup.find_all('mark'):
                            if m_tag not in og_soup:
                                tagging_error_word_count += 1
                            tagged_words += m_tag.getText() + "   "
                            tags += m_tag["data-entity"].upper() + " "
                            final_sentence = qc_soup.getText().strip() + "    " + tagged_words.strip() + "    " + tags.strip()
                    else:
                        final_sentence = qc_soup.getText().strip() + "    " + qc_soup.getText().strip() + "    O"
                    if final_sentence != "":
                        dataset_lines.append(final_sentence)

    folder_path = "datasets"
    file_name = os.path.join(folder_path, str(int(time.time()))) + ".txt"
    # os.makedirs(folder_path, exist_ok=True)
    # with open(file_name, 'w') as f:
    #     for line in dataset_lines:
    #         for i in range(7):
    #             f.write(line + "\n")

    return Response(json.dumps({"file_path": file_name,
                                "lines": {
                                    "total_lines": total_count,
                                    "changed_lines": error_count},
                                "words": {
                                    "total_words": total_word_count,
                                    "spelling_issues": spelling_error_word_count,
                                    "tagging_issues": tagging_error_word_count}}))


def is_validated(license_key):
    if 'nt' in os.name:
        device_id = check_output("wmic csproduct get uuid".split(" ")).decode('utf-8').split()[1]
    else:
        device_id = check_output("hal-get-property --udi /org/freedesktop/Hal/devices/computer --key system.hardware.uuid".split(" ")).decode('utf-8').split()[1]
    try:
        response = requests.post('http://' + ':'.join([configuration.get('qs_server', 'url'), configuration.get('qs_server', 'port')]) + "/validate_product", data=json.dumps(dict(license_key=license_key, device_id=device_id)))
        if response.json()['new_registration']:
            configuration.add_section('license')
            configuration.set('license', 'key', license_key)
            with open('config.ini', 'w') as f:
                configuration.write(f)
        return response.json()['authenticated']
    except:
        return False


@app.route("/api/docs")
def spec():
    swag = swagger(app)
    swag['info']['version'] = "1.0"
    swag['info']['title'] = "Digitization APIs"
    return jsonify(swag)


if __name__ == "__main__":
    # Call factory function to create our blueprint
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
        API_URL,
        config={  # Swagger UI config overrides
            'app_name': "Digitization"
        },
    )

    app.register_blueprint(swaggerui_blueprint)

    app.run(host='0.0.0.0', port=5005, debug=False)
    # key = None
    # if configuration.has_section('license') and configuration.has_option('license', 'key'):
    #     key = configuration.get('license', 'key')
    # if is_validated(key):
    #     app.run(host='0.0.0.0', port=5000)
    # else:
    #     def character_limit(entry_text, prev_entry=None, next_entry=None):
    #         if len(entry_text.get()) > 4:
    #             entry_text.set(entry_text.get()[:5])
    #             if next_entry is not None:
    #                 next_entry.focus()
    #         elif len(entry_text.get()) == 0:
    #             if prev_entry is not None:
    #                 prev_entry.focus()
    #         key = c1.get() + c2.get() + c3.get() + c4.get() + c5.get()
    #         if len(key) >= 25:
    #             verify_button['state'] = 'normal'
    #         else:
    #             verify_button['state'] = 'disabled'
    #
    #     def check_key():
    #         key = c1.get() + c2.get() + c3.get() + c4.get() + c5.get()
    #         if is_validated(key):
    #             window.destroy()
    #             app.run(host='0.0.0.0', port=5000)
    #         else:
    #             tk.messagebox.showerror(title="Unsuccessful", message="Could not verify the key at the moment")
    #
    #
    #     window = tk.Tk()
    #     window.title("Digitization")
    #     frame_header = tk.Frame()
    #     tk.Label(master=frame_header, text="Enter your product key").pack(side=tk.LEFT)
    #
    #     frame_buttons = tk.Frame()
    #     verify_button = tk.Button(master=frame_buttons, text="Verify", command=check_key, state=tk.DISABLED)
    #     verify_button.pack()
    #
    #     frame_key = tk.Frame()
    #     c1_text = tk.StringVar()
    #     c1 = tk.Entry(master=frame_key, textvariable=c1_text)
    #     c1.grid(column=0, row=0)
    #     lbl = tk.Label(master=frame_key, text="\t-\t")
    #     lbl.grid(column=1, row=0)
    #     c2_text = tk.StringVar()
    #     c2 = tk.Entry(master=frame_key, textvariable=c2_text)
    #     c2.grid(column=2, row=0)
    #     lbl = tk.Label(master=frame_key, text="\t-\t")
    #     lbl.grid(column=3, row=0)
    #     c3_text = tk.StringVar()
    #     c3 = tk.Entry(master=frame_key, textvariable=c3_text)
    #     c3.grid(column=4, row=0)
    #     lbl = tk.Label(master=frame_key, text="\t-\t")
    #     lbl.grid(column=5, row=0)
    #     c4_text = tk.StringVar()
    #     c4 = tk.Entry(master=frame_key, textvariable=c4_text)
    #     c4.grid(column=6, row=0)
    #     lbl = tk.Label(master=frame_key, text="\t-\t")
    #     lbl.grid(column=7, row=0)
    #     c5_text = tk.StringVar()
    #     c5 = tk.Entry(master=frame_key, textvariable=c5_text)
    #     c5.grid(column=8, row=0)
    #     c1_text.trace("w", lambda *args: character_limit(c1_text, next_entry=c2))
    #     c2_text.trace("w", lambda *args: character_limit(c2_text, prev_entry=c1, next_entry=c3))
    #     c3_text.trace("w", lambda *args: character_limit(c3_text, prev_entry=c2, next_entry=c4))
    #     c4_text.trace("w", lambda *args: character_limit(c4_text, prev_entry=c3, next_entry=c5))
    #     c5_text.trace("w", lambda *args: character_limit(c5_text, prev_entry=c4))
    #
    #     frame_header.pack(padx=20, pady=20)
    #     frame_key.pack(padx=50, pady=20)
    #     frame_buttons.pack(padx=20, pady=20)
    #
    #     window.mainloop()
