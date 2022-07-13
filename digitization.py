import json
import locale
import operator
import os
from sys import path
import regex as re
import shutil
import time
from collections import OrderedDict
from configparser import ConfigParser
from glob import glob
from queue import Queue
from threading import Thread

import cv2
import ghostscript
import math
import nltk
import numpy as np
# import pdftotext
import pytesseract
from pytesseract import Output
import spacy
from dateutil.parser import parse
from scipy import ndimage
import csv
import collections
from scipy.stats import skew

import random
# from spacy.gold import GoldParse
# from spacy.util import minibatch, compounding, decaying
from tqdm import tqdm
from spacy.scorer import Scorer

from components.LineItem import LineItem, BoxItem
from components.TestItem import TestItem, TestItemType
from utils.pandas_match_finder import MatchFinder
from io import StringIO
from PIL import Image, ImageChops
import pandas as pd
import boto3
import mimetypes
import textract
from pdf2image import convert_from_path


pd.options.mode.chained_assignment = None
Image.MAX_IMAGE_PIXELS = None

ALLOWED_EXTENSIONS = ['pdf', 'jpg', 'jpeg', 'png']
configuration = ConfigParser()
configuration.read('config.ini')

s3 = boto3.client(
    's3',
    region_name = str(configuration.get('s3', 'region_name')),
    aws_access_key_id = str(configuration.get('s3', 'aws_access_key_id')),
    aws_secret_access_key = str(configuration.get('s3', 'aws_secret_access_key'))
)

class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """

    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            # try:
            func(*args, **kargs)
            # except Exception as e:
            # An exception happened in this thread
            # print("Worker:", e)
            # finally:
            # Mark this task as done, whether an exception happened or not
            self.tasks.task_done()


class ThreadPool:
    """ Pool of threads consuming tasks from a queue """

    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()


class ConvertToTxt:
    directory_path = None
    medical_words = list()
    words = list()
    file_text = dict()
    time_taken = dict()

    def __init__(self, directory_path=None, thread_limit=1):
        """
        Constructor for the ConvertToTxt class.

        :param directory_path: Optional. The path of the directory where the files are to be saved. Read from config.ini by default.
        """
        self.medical_words = list()
        self.words = list()
        self.file_text = dict()
        self.time_taken = dict()
        self.folder_structure = ""
        self.folder_paths = dict()

        config_dict = dict(configuration.items('digitization'))
        self.extracted_image_path = config_dict.get("extracted_image_path")
        os.makedirs(self.extracted_image_path, exist_ok=True)

        # self.extracted_pages_path = config_dict.get("extracted_pages_path")
        # os.makedirs(self.extracted_pages_path, exist_ok=True)

        configured_path = config_dict.get('default_reporting_directory')
        if directory_path is None and configured_path is None:
            raise FileNotFoundError("Directory path cannot be null.")
        elif directory_path is None:
            self.directory_path = configured_path
        else:
            self.directory_path = directory_path

        self.__setup_folder_structure()

        self.medical_words.extend(pd.read_csv('medical_words.csv')['Test Name'].to_list())
        self.words = set(nltk.corpus.words.words())

        self.thread_limit = thread_limit
        self.__filename = None

    @staticmethod
    def __check_if_extension_allowed(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_files_by_name_or_count_aws_s3(self, file_name=None, count=1):
        self.__filename = None
        files = []
        if file_name is None:
            response = s3.list_objects_v2(
                            Bucket = configuration.get('s3', 'bucket_name'),
                            Prefix = self.folder_paths['input_path'],
                            MaxKeys = count)
            for objectKey in response['Contents']:
                filename = objectKey['Key']
                if self.__check_if_extension_allowed(filename):
                    files.append(filename)
                    ## Download the file locally
                    s3.download_file(configuration.get('s3', 'bucket_name'), filename, filename)
        else:
            try:
                file_path = str(self.folder_paths['input_path'] + '/' + file_name)
                response = s3.get_object(
                                Bucket = configuration.get('s3', 'bucket_name'),
                                Key = file_path)
                files.append(file_path)
                s3.download_file(configuration.get('s3', 'bucket_name'), file_path, file_path)
            except Exception as e:
                raise FileNotFoundError(f"File '{file_name}' does not exist in '{self.folder_paths['input_path']}'!")
        return files

    def upload_full_dir_to_s3(self, path=None):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(subdir, file)
                file_mime = mimetypes.guess_type(file)[0] or 'binary/octet-stream'
                with open(full_path, 'rb') as data:
                    s3.put_object(Key=full_path, Body=data, ContentType=file_mime, Bucket=configuration.get('s3', 'bucket_name'))
                    os.remove(full_path)

    def upload_results_to_s3(self, filename=None):
        if filename == None:
            return
        
        ## Check Processed Reports
        file_path = os.path.join(self.folder_paths['output_processed_path'], filename + '.pdf')
        move_file_from = os.path.join(self.folder_paths['input_path'], filename + '.pdf')
        if os.path.exists(file_path):
            s3.put_object(Body=open(file_path, 'rb'),Bucket=configuration.get('s3', 'bucket_name'), Key=file_path)
            s3.delete_object(Bucket=configuration.get('s3', 'bucket_name'), Key=move_file_from)
            os.remove(file_path)

        ## Check Unprocessed Reports
        file_path = os.path.join(self.folder_paths['output_unprocessed_path'], filename + '.pdf')
        move_file_from = os.path.join(self.folder_paths['input_path'], filename + '.pdf')
        if os.path.exists(file_path):
            s3.put_object(Body=open(file_path, 'rb'),Bucket=configuration.get('s3', 'bucket_name'), Key=file_path)
            s3.delete_object(Bucket=configuration.get('s3', 'bucket_name'), Key=move_file_from)
            os.remove(file_path)

        ## Check extracted HTML files
        file_path = os.path.join(self.folder_paths['output_html_path'], filename + '.json')
        if os.path.exists(file_path):
            s3.put_object(Body=open(file_path, 'rb'),Bucket=configuration.get('s3', 'bucket_name'), Key=file_path)
            os.remove(file_path)
        
        ## Check extracted TXT files
        file_path = os.path.join(self.folder_paths['output_txt_path'], filename + '.txt')
        if os.path.exists(file_path):
            s3.put_object(Body=open(file_path, 'rb'),Bucket=configuration.get('s3', 'bucket_name'), Key=file_path)
            os.remove(file_path)
        
        ## Check extracted PNG files
        self.upload_full_dir_to_s3(path=self.folder_paths['extracted_pages_path'])


        ## Check extracted JSON Files
        file_path = os.path.join(self.folder_paths['extracted_json_path'], filename + '.json')
        if os.path.exists(file_path):
            s3.put_object(Body=open(file_path, 'rb'),Bucket=configuration.get('s3', 'bucket_name'), Key=file_path)
            os.remove(file_path)



    def get_files_by_name_or_count(self, file_name=None, count=1):
        """
        Gets all files from the folder path specified while creating the object.

        :param file_name: Optional. Provide a filename to process. If the file does not exist, an error is raised.
        :param count: Optional. The number of files to process. Defaults to 1.
        :return: Returns a list of files found in the folder path.
        """
        if file_name is None:
            files = [y.replace('\\', '/') for x in os.walk(self.folder_paths['input_path'])
                     for y in glob(os.path.join(x[0], '*.*')) if self.__check_if_extension_allowed(y)
                     if os.path.isfile(y.replace('\\', '/'))][:count]
            if len(files) == 0:
                raise FileNotFoundError(f"Input directory '{self.folder_paths['input_path']}' is empty!")
        else:
            file = os.path.join(self.folder_paths['input_path'], file_name)
            if os.path.isfile(file):
                files = [file.replace('\\', '/')]
            else:
                raise FileNotFoundError(f"File '{file_name}' does not exist in '{self.folder_paths['input_path']}'!")
        return files

    def __setup_folder_structure(self):
        self.folder_paths['input_path'] = os.path.join(self.directory_path,
                                                       configuration.get('digitization', 'input_path'))
        self.folder_paths['output_processed_path'] = os.path.join(self.directory_path,
                                                                  configuration.get('digitization', 'output_processed_path'))
        self.folder_paths['output_unprocessed_path'] = os.path.join(self.directory_path,
                                                                    configuration.get('digitization', 'output_unprocessed_path'))
        self.folder_paths['output_txt_path'] = os.path.join(self.directory_path,
                                                            configuration.get('digitization', 'output_txt_path'))
        self.folder_paths['output_html_path'] = os.path.join(self.directory_path,
                                                                    configuration.get('digitization', 'output_html_path'))
        self.folder_paths['extracted_pages_path'] = os.path.join(self.directory_path,
                                                       configuration.get('digitization', 'extracted_pages_path'))
        self.folder_paths['extracted_json_path'] = os.path.join(self.directory_path,
                                                       configuration.get('digitization', 'extracted_json_path'))
        for path in self.folder_paths.values():
            os.makedirs(path, exist_ok=True)

    def __rescale_image(self, img_path=None, img=None, required_width=1600, required_height=2250):
        if img_path is None and img is None:
            raise FileNotFoundError('Both image path and image cannot be None. Please provide one.')
        elif img is None and img_path is not None:
            if not os.path.isfile(img_path):
                raise FileNotFoundError('Invalid file.')
            else:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = self.__crop(img)

        og_width, og_height = img.shape[::-1]
        scale_width, scale_height = required_width / og_width, required_height / og_height
        __dims = int(og_width * scale_width), int(og_height * scale_height)
        return cv2.resize(img, __dims, interpolation=cv2.INTER_AREA)

    @staticmethod
    def __crop(im):
        im = Image.fromarray(im)
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()

        if bbox:
            return np.array(im.crop(bbox))
        else:
            return np.array(im)
    
    ###################################### Code to add category in the text ################################
    def add_category(self, final_text_dict):
        df = pd.read_csv('Category-tests.csv')
        col = ''
        found = False
        
        for page_no, page_data in final_text_dict.items():
            catgry = {}
            page_data = f'''{page_data}'''
            lines = page_data.strip().split('\n')
            for line in lines:
                found = False
                for col in df.columns:
                    rows = [val for val in df[col] if val == val]
                    for each_wrd in rows:
                        if each_wrd.lower() in line.lower():
                            if col not in catgry.keys():
                                if self.check_tests(lines, rows):
                                    if page_data.lower().find('TEST'.lower()) and \
                                    page_data.lower().find('TEST'.lower()) < page_data.lower().find(each_wrd.lower()):
                                        catgry[col] = page_data.lower().find('TEST'.lower())
                                    elif page_data.lower().find(each_wrd.lower()):
                                        catgry[col] = page_data.lower().find(each_wrd.lower())
                                    found = True
                    if found:
                        break

            for cat, indx in catgry.items():
                if cat.lower() not in page_data.lower():
                    
                    final_text_dict[page_no] = page_data[:indx] + '''\n''' + cat.upper() + '''\n''' + page_data[indx:]

        return final_text_dict

    def check_tests(self, lines, rows):
        count = 0
        for row in rows:
            for line in lines:
                if row in line:
                    count += 1

        if count > 3:
            return True
        else:
            return False
    ###################################### Code to add category in the text ################################

    def process_files(self, files, dpi=900, save_to_txt=False, return_page_count=False, move_file=False):
        """Process a single file or a list of files passed.

            :param files: A single file path or a list of file paths.
            :param dpi: The dpi to be used for the PDF2IMG and OCR processes. Defaults to 450.
            :param save_to_txt: Whether to save the text in a txt file or not. Defaults to False.
            :param return_page_count: Whether to return the number of pages processed. Defaults to False.
            :param move_file: Whether to move file to it's respective folder or only copy it.

            :return: A string containing the text result of the OCR process.
        """
        page_count = 0
        output_list = list()
        if type(files) is not list:
            files = [files]
        final_text_dict = dict()
        for file in files:
            file_name_with_ext = file.rsplit('/', 1)[1]
            file_name = file_name_with_ext.rsplit(".", 1)[0]
            try:
                if self.__filename is None:
                    self.__filename = os.path.join(self.folder_structure, str(file.rsplit('/', 1)[-1].rsplit('.', 1)[0])).replace("\\", '/')
                else:
                    self.__filename = (self.folder_structure + self.__filename + '_' + str(file.rsplit('/', 1)[-1].rsplit('.', 1)[0])).replace("\\", '/')
                # phase_one_text_dict = self.__extract_from_pdf_a(file)
                phase_two_text_dict = self.__extract_from_textract(file, dpi)
                print(phase_two_text_dict)

                # phase_one_text_dict = {int(k) + 1: v for k, v in phase_one_text_dict.items()}
                # phase_two_text_dict = {int(k) + 1: v for k, v in phase_two_text_dict.items()}

                # counter_dict = phase_one_text_dict if len(phase_one_text_dict.keys()) >= len(phase_two_text_dict.keys()) \
                #     else phase_two_text_dict

                counter_dict = phase_two_text_dict
                counter_dict = {int(k): v for k, v in counter_dict.items()}

                page_count_dict = dict()

            ###################################### OCR selection logic ###################################
                # for k, v in sorted(counter_dict.items(), key=lambda item: item[0]):
                #     print(f"Saving page {k}")
                    # if k in phase_one_text_dict and k in phase_two_text_dict:
                    #     phase_one_eng = " ".join(
                    #         w for w in nltk.wordpunct_tokenize(phase_one_text_dict[k]) if w.lower() in self.words)
                    #     phase_two_eng = " ".join(
                    #         w for w in nltk.wordpunct_tokenize(phase_two_text_dict[k]) if w.lower() in self.words)

                    #     if len(phase_one_text_dict[k].split()) == 0:
                    #         one_eng_percent = 0
                    #     else:
                    #         one_eng_percent = (len(phase_one_eng.split()) / len(phase_one_text_dict[k].split())) * 100

                    #     if len(phase_two_text_dict[k].split()) == 0:
                    #         two_eng_percent = 0
                    #     else:
                    #         two_eng_percent = (len(phase_two_eng.split()) / len(phase_two_text_dict[k].split())) * 100

                    #     if one_eng_percent >= two_eng_percent:
                    #         final_text_dict[k] = phase_one_text_dict[k].encode('ascii', errors='ignore').decode('ascii')
                    #         page_count_dict[k] = len(phase_one_text_dict[k].strip())
                    #     else:
                    #         med_count_one = 0
                    #         med_count_two = 0
                    #         for word in phase_one_text_dict[k].split():
                    #             if word.lower() in self.medical_words:
                    #                 med_count_one += 1

                    #         for word in phase_two_text_dict[k].split():
                    #             if word.lower() in self.medical_words:
                    #                 med_count_two += 1

                    #         if len(phase_one_text_dict[k].split()) == 0:
                    #             one_med_percent = 0
                    #         else:
                    #             one_med_percent = (med_count_one / len(phase_one_text_dict[k].split())) * 100

                    #         if len(phase_two_text_dict[k].split()) == 0:
                    #             two_med_percent = 0
                    #         else:
                    #             two_med_percent = (med_count_two / len(phase_two_text_dict[k].split())) * 100

                    #         if one_med_percent >= two_med_percent and len(phase_one_text_dict[k]) != 0:
                    #             final_text_dict[k] = phase_one_text_dict[k].encode('ascii', errors='ignore').decode('ascii')
                    #             page_count_dict[k] = len(phase_one_text_dict[k].strip())
                    #         else:
                    #             final_text_dict[k] = phase_two_text_dict[k].encode('ascii', errors='ignore').decode('ascii')
                    #             page_count_dict[k] = len(phase_two_text_dict[k].strip())
                    # elif k in phase_one_text_dict:
                    #     final_text_dict[k] = phase_one_text_dict[k].encode('ascii', errors='ignore').decode('ascii')
                    #     page_count_dict[k] = len(phase_one_text_dict[k].strip())
                    # elif k in phase_two_text_dict:
                    #     final_text_dict[k] = phase_two_text_dict[k].encode('ascii', errors='ignore').decode('ascii')
                    #     page_count_dict[k] = len(phase_two_text_dict[k].strip())

            #########################################################################

                for k, v in sorted(counter_dict.items(), key=lambda item: item[0]):
                    final_text_dict[k] = phase_two_text_dict[k].encode('ascii', errors='ignore').decode('ascii')
                    page_count_dict[k] = len(phase_two_text_dict[k].strip())

                final_text_dict = self.add_category(final_text_dict)
                
                output_list.extend([line for page in final_text_dict.values() for line in page.splitlines()])
                page_count += len(page_count_dict.keys())
                if len(final_text_dict.keys()) == 0:
                    dest_path = os.path.join(self.folder_paths['output_unprocessed_path'], self.folder_structure, file_name_with_ext)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    if move_file:
                        shutil.move(file, dest_path)
                    else:
                        shutil.copy(file, dest_path)
                else:
                    if save_to_txt:
                        output_file_path = os.path.join(self.folder_paths['output_txt_path'], self.folder_structure, file_name + ".txt")
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        with open(output_file_path, "w") as outputfile:
                            json.dump(final_text_dict, outputfile, indent=4)
                            print('Text file saved.---------------------')

                    dest_path = os.path.join(self.folder_paths['output_processed_path'], self.folder_structure, file_name_with_ext)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    if move_file:
                        shutil.move(file, os.path.join(self.folder_paths['output_processed_path'], self.folder_structure, file_name_with_ext))
                    else:
                        shutil.copy(file, os.path.join(self.folder_paths['output_processed_path'], self.folder_structure, file_name_with_ext))
            except Exception as e:
                print("Exception-------------------------",e)
                dest_path = os.path.join(self.folder_paths['output_unprocessed_path'], self.folder_structure,
                                         file_name_with_ext)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if move_file:
                    shutil.move(file, dest_path)
                else:
                    shutil.copy(file, dest_path)

        if return_page_count:
            return self.__filename, final_text_dict, page_count
        else:
            return self.__filename, final_text_dict

    @staticmethod
    def __convert_pdf_to_png(pdf_input_path, output_folder, dpi):
        filename = pdf_input_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        file_path = os.path.join(output_folder, filename)
        os.makedirs(file_path, exist_ok=True)
        args = ["pdf2png",
                "-dNOPAUSE",
                # "-dQUIET",
                "-sDEVICE=pnggray",
                "-r" + str(dpi),
                "-sOutputFile=" + os.path.join(file_path, filename) + "-%d.png",
                pdf_input_path]

        encoding = locale.getpreferredencoding()
        args = [a.encode(encoding) for a in args]

        with ghostscript.Ghostscript(*args):
            ghostscript.cleanup()

        return glob(os.path.join(file_path, './*.png'))

    @staticmethod
    def __convert_pdf_to_png_using_pdf2img(pdf_input_path, output_folder, dpi):
        filename = pdf_input_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        file_path = os.path.join(output_folder, filename)
        
        images = convert_from_path(pdf_input_path, dpi= dpi, size=(765,990))        
        
        os.makedirs(file_path, exist_ok=True)
        
        for i in range(len(images)):
            outFilePath = os.path.join(file_path, filename+"-"+str(i+1)+".png")
            # resized_img = images[i].resize((765,990))
            images[i].save(outFilePath, 'PNG')
            
        
        return glob(os.path.join(file_path, './*.png'))





    @staticmethod
    def __extract_from_pdf_a(file_path):
        with open(file_path, "rb") as f:
            text = list(pdftotext.PDF(f))

        phase_one_pages = dict()
        for page in text:
            phase_one_pages[str(text.index(page))] = page
        return phase_one_pages

    def __savePNGs(self, file_path, dpi):
        global ALLOWED_EXTENSIONS
        image_ext = ALLOWED_EXTENSIONS.copy()
        image_ext.remove('pdf')
        if file_path.rsplit('.', 1)[-1].lower() == 'pdf':
            temp_path = os.path.join(os.getcwd(), "Reports", "extracted_png")
            pages = self.__convert_pdf_to_png_using_pdf2img(file_path, dpi=900, output_folder=os.path.join(temp_path, self.folder_structure))
            pages = sorted(pages, key=lambda x: float(x.rsplit('-', 1)[-1].rsplit('.', 1)[0]))
        elif file_path.rsplit('.', 1)[-1] in image_ext:
            pages = [file_path]
        else:
            raise TypeError(f"Invalid file type {file_path.rsplit('.', 1)[-1]}. Please ensure that all files types are one of the following: {ALLOWED_EXTENSIONS}")

    def __extract_from_pdf_b(self, file_path, dpi):
        pages = self.__savePNGs(file_path, dpi)
        thread_pool = ThreadPool(self.thread_limit)
        print("Pages -----> ", pages)
        args = [(page, dpi, int(page.rsplit('-', 1)[-1].rsplit('.', 1)[0])) for page in pages]
        thread_pool.map(self.__perform_ocr, args_list=args)
        thread_pool.wait_completion()
        return self.file_text, self.time_taken

    def __extract_from_textract(self, file_path, dpi):
        print('File Path ------> ', file_path)
        self.__savePNGs(file_path, dpi)
        print("Pngs saved....")
        ## Nilesh's edit  removed threading for below functio
        t = Thread(target=textract.extract_main, args=[file_path])
        t.start()
        while t.isAlive():
            pass
        #textract.extract_main(file_path)
        print("RESPONSE DICT ----> ", textract.pages_dict)
        return textract.pages_dict

    @staticmethod
    def df_to_tsv(df):
        display_str = ""
        for i in range(len(df.index)):
            row = list(df.iloc[i])
            display_str = (display_str + "    ".join(map(str, row))).strip() + "\n"

        display_str = re.sub(" nan ", " ", (" " + display_str + " "), flags=re.I)
        return display_str

    def __perform_ocr(self, args):
        line_items = list()
        start = time.time()
        image_path, dpi, page_idx = args
        print(f"Extracting from page {page_idx}")

        large = cv2.imread(image_path)

        # large = cv2.cvtColor(large, cv2.COLOR_GRAY2BGR)
        image = large.copy()

        img_rotated = self.__rotate_img(image, page_idx)
        img_path = os.path.join(self.folder_paths['extracted_pages_path'], self.__filename, self.__filename.rsplit('/', 1)[-1] + '-' + str(page_idx) + ".png")
        # print(img_path)
        cv2.imwrite(img_path, img_rotated)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        # cv2.resizeWindow('image', 750, 900)
        # cv2.imshow("image", img_rotated)  # Show image
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        img = img_rotated
        # print(img.shape)
        if len(img.shape) > 2:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        # enhanced_image = self.__enhance_image(img_gray)
        small = cv2.pyrDown(img_gray)
        # small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        image_final = small.copy()
        result = self.__remove_hor_vet_lines(image_final)
        boxes = self.__get_boxes(small, result)
        val1, val2 = self.__check_orientation(boxes, image_final)

        if abs(val1 + val2) == 1:
            image_final = self.__rescale_image(img=image_final, required_width=1600 * 4, required_height=2250 * 4)
            rot_data = pytesseract.image_to_osd(image_final, output_type=Output.DICT)
            angle = float(rot_data['orientation'])
            print(f"Orientation of page {page_idx} is {round(angle, 2)}\N{DEGREE SIGN}.")

            val3 = -1 if angle > 0 else 1

            if (val1 + val2 + val3) < 0:
                image_final = cv2.cvtColor(image_final, cv2.COLOR_GRAY2BGR)
                image_final = self.__rescale_image(img=image_final)
                image_final = np.asarray(Image.fromarray(image_final).rotate(angle))
                # image_final = ndimage.rotate(image_final, angle)

                large = cv2.cvtColor(image_final, cv2.COLOR_GRAY2BGR)
                image = large.copy()

                img_rotated = self.__rotate_img(image, page_idx)
                img_path = os.path.join(self.folder_paths['extracted_pages_path'], self.__filename,
                                        self.__filename.rsplit('/', 1)[-1] + '-' + str(page_idx) + ".png")
                cv2.imwrite(img_path, img_rotated)
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
                # cv2.resizeWindow('image', 750, 900)
                # cv2.imshow("image", img_rotated)  # Show image
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                img_gray = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
                # enhanced_image = self.__enhance_image(img_gray)
                small = cv2.pyrDown(img_gray)
                # small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

                image_final = small.copy()
                result = self.__remove_hor_vet_lines(image_final)
                boxes = self.__get_boxes(small, result)

        line_items = self.__create_lines(line_items, result, boxes, str(page_idx))

        text_df = pd.DataFrame()
        for idx, line in enumerate(line_items[::-1]):
            text_df = text_df.append({"col_" + str(box_idx): box.text for box_idx, box in enumerate(line.get_boxes())}, ignore_index=True)
        text_df = text_df.fillna('')
        text = self.df_to_tsv(text_df)
        self.file_text[page_idx] = text
        # print("-" * 15, page_idx, "-" * 15)
        # print(text)
        end = time.time()
        self.time_taken[page_idx] = end - start
        print(f"Extracted text from page {str(page_idx)} in {round(end - start, 2)} seconds.")

    @staticmethod
    def __check_orientation(boxes, display_img):
        boxes = sorted(boxes, key=lambda item: item[0])
        boxes = sorted(boxes, key=lambda item: item[1])

        current_y = -1
        grouped_boxes = []
        active_group = []

        boxes = [box for box in boxes if 300 <= box[1] <= 1000]

        for box in boxes:
            if current_y == -1:
                current_y = box[1]

            elif abs(box[1] - current_y) >= 8:
                if len(active_group) > 0:
                    grouped_boxes.append(active_group)
                    active_group = []

                current_y = box[1]
                active_group.append(box)

            else:
                active_group.append(box)

        if len(active_group) > 0:
            grouped_boxes.append(active_group)

        data = []

        for group in grouped_boxes:
            ind_item = {}
            for box in group:
                ind_item[box[0]] = box[4]
            data.append(ind_item)

        skews = []
        sides = []
        skews_of_skews = 0
        skews_of_sides = 0

        for item in data:
            arr = []
            od = collections.OrderedDict(sorted(item.items()))
            start_x = list(od.keys())[0]
            end_x = list(od.keys())[-1] + list(od.values())[-1]
            total_width = display_img.shape[1]
            considered_section = total_width * 0.45
            weight_distribution = ((start_x + end_x) / 2)
            sides.append(1 if weight_distribution <= considered_section else -1 if weight_distribution >= (
                        total_width - considered_section) else 0)
            for k, v in od.items():
                arr.extend([v] * k)

            skewness = skew(arr)
            if not np.isnan(skewness):
                skews.append(skewness)

            skews_of_skews = np.mean(skews)
            skews_of_sides = np.mean(sides)

            if np.isnan(skews_of_skews) or skews_of_skews > 0:
                skews_of_skews = 1
            else:
                skews_of_skews = -1

            if np.isnan(skews_of_sides) or skews_of_sides > 0:
                skews_of_sides = 1
            else:
                skews_of_sides = -1

        return skews_of_skews, skews_of_sides

    @staticmethod
    def __enhance_image(image):
        # TODO - Fix this function
        img_blur = cv2.GaussianBlur(image, (5, 5), 1)

        img_adaptive_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 5, 2)
        img_adaptive_threshold = cv2.bitwise_not(img_adaptive_threshold)
        img_adaptive_threshold = cv2.medianBlur(img_adaptive_threshold, 3)

        return img_adaptive_threshold

    @staticmethod
    def __rotate_img(image, page_no):
        edges = cv2.Canny(image, 400, 500, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        main_lines = []

        display_img = image.copy()
        display_img = cv2.line(display_img, (0, 300), (image.shape[0], 300), (0, 0, 255), 2)
        display_img = cv2.line(display_img, (0, 1800), (image.shape[0], 1800), (0, 0, 255), 2)

        angles = []
        if lines is not None:
            for i in range(0, len(lines)):
                for rho, theta in lines[i]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * a)
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if -70 > angle > -90:
                        angle = 90 + angle
                    elif 70 < angle < 90:
                        angle = angle - 90
                    if 1800 > y1 > 300 and 1800 > y2 > 300 and 0 < abs(angle) < 35:
                        angles.append(angle)
                        display_img = cv2.line(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        main_lines.append((x1, y1, x2, y2, angle))

            median_angle = 0
            if len(angles) > 0:
                min_angle, max_angle = np.mean(angles) - np.std(angles), np.mean(angles) + np.std(angles)
                considerable_lines = [line for line in main_lines if max_angle >= line[4] >= min_angle]
                median_angle = np.mean([line[4] for line in considerable_lines]) if len(considerable_lines) > 0 else 0
            # print(median_angle)
            if median_angle < -45:
                median_angle = 90 + median_angle
            print(f"Angle of rotation for page {page_no} is {round(median_angle, 2)}\N{DEGREE SIGN}.")
            # img_rotated = ndimage.rotate(image, median_angle)
            img_rotated = np.asarray(Image.fromarray(image).rotate(median_angle))
        else:
            img_rotated = image.copy()
        return img_rotated

    @staticmethod
    def __remove_hor_vet_lines(image_final):
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        remove_vertical = 255 - cv2.morphologyEx(image_final, cv2.MORPH_CLOSE, kernel_vertical)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        remove_horizontal = 255 - cv2.morphologyEx(image_final, cv2.MORPH_CLOSE, horizontal_kernel)

        remove_both = cv2.add(remove_vertical, remove_horizontal)
        result = cv2.add(remove_both, image_final)
        return result

    @staticmethod
    def __get_boxes(og_image, result):
        # display_img = og_image.copy()
        temp_img = result.copy()

        ret, grad = cv2.threshold(temp_img, 200, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if ret < 240:
            ret += 10
        ret, bw = cv2.threshold(temp_img, ret, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(bw.shape, dtype=np.uint8)
        boxes = []
        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])
            mask[y:y + h, x:x + w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
            h_range = range(6, 100)
            if r > 0.3 and w > 7 and h in h_range:
                boxes.append([x, y, x + w, y + h, w, h])

        return boxes

    def df_to_text(self, img_block):
        resized_img = cv2.resize(img_block, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        thresholded_img = cv2.threshold(resized_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        tsv = pytesseract.image_to_data(thresholded_img, config="-l eng --psm 8")
        df = pd.read_csv(StringIO(tsv), sep='\t', quoting=csv.QUOTE_NONE)
        df = df[df.conf != -1]

        def int_fixer(value):
            try:
                if type(value) is not str:
                    value = str(value)
                if re.match(r'^[0-9.]+$', value):
                    if value[0] == '.':
                        return str(int(float(value[1:])) if float(value) == float(int(float(value))) else value)
                    elif value[-1] == '.':
                        return str(int(float(value[:-1])) if float(value) == float(int(float(value))) else value)
                    else:
                        return str(int(float(value)) if float(value) == float(int(float(value))) else value)
                else:
                    return re.sub(r'^\W?([0-9.]+)\W?$', r'\1', value)
            except ValueError:
                return value

        df.dropna(inplace=True)
        df['text'] = df['text'].apply(int_fixer)
        df['dummy'] = '0'
        df['text'] = df['text'].fillna('')
        df['text'] = df['text'].astype(str)
        df1 = df.groupby(['dummy'])['text'].apply(' '.join).reset_index()
        text = ""
        if 'text' in df1 and len(df1['text']) == 1:
            text = df1['text'][0].strip()
            text = re.sub(" nan ", " ", (" " + text + " "), flags=re.I)
            os.makedirs(os.path.join(self.extracted_image_path, self.__filename), exist_ok=True)
            cv2.imwrite(os.path.join(self.extracted_image_path, self.__filename, text) + ".png", thresholded_img)
        return text

    def __create_lines(self, line_items, page_image, boxes, page_no):
        padding = 3
        display_img = cv2.cvtColor(page_image.copy(), cv2.COLOR_GRAY2RGB)
        for box in boxes:
            if len(line_items) > 0:
                if abs(line_items[-1].line_coordinates[1] - box[1]) < 8:
                    box_item = BoxItem(box)
                    box_item.text = self.df_to_text(page_image[max(box[1] - padding, 0):min(box[3] + padding, page_image.shape[0]), max(box[0] - padding, 0):min(box[2] + padding, page_image.shape[1])])
                    line_items[-1].add_box(box_item)
                else:
                    item = LineItem(len(line_items) + 1)
                    box_item = BoxItem(box)
                    box_item.text = self.df_to_text(page_image[max(box[1] - padding, 0):min(box[3] + padding, page_image.shape[0]), max(box[0] - padding, 0):min(box[2] + padding, page_image.shape[1])])
                    item.add_box(box_item)
                    line_items.append(item)
            else:
                item = LineItem(1)
                box_item = BoxItem(box)
                box_item.text = self.df_to_text(page_image[max(box[1] - padding, 0):min(box[3] + padding, page_image.shape[0]), max(box[0] - padding, 0):min(box[2] + padding, page_image.shape[1])])
                item.add_box(box_item)
                line_items.append(item)

        for line in line_items:
            for box in line.boxes:
                cv2.rectangle(display_img, (max(box.coordinates[0] - padding, 0), max(box.coordinates[1] - padding, 0)),
                              (min(box.coordinates[2] + padding, page_image.shape[1]), min(box.coordinates[3] + padding, page_image.shape[0])), (255, 0, 0), 2)

        os.makedirs(os.path.join(self.folder_paths['extracted_pages_path'], self.__filename + '_ocr'), exist_ok=True)
        cv2.imwrite(
            os.path.join(self.folder_paths['extracted_pages_path'], self.__filename + '_ocr', self.__filename.rsplit('/', 1)[-1] + '-' + page_no + ".png"),
            display_img)
        return line_items


class ExtractToJson:
    __LIMIT = 5
    __predict_fn = None
    __file_directory = None
    df = pd.read_csv("test.csv")
    if configuration.has_section('digitization') and configuration.has_option('digitization', 'model_directory'):
        __model_dir = configuration.get('digitization', 'model_directory')
    else:
        __model_dir = None

    def __init__spacy2(self, model_directory=None, file_directory=None):
        spacy.prefer_gpu()
        if model_directory is not None:
            self.__model_dir = model_directory
        elif self.__model_dir is None:
            raise FileNotFoundError('Model __directory cannot be null!')
        if file_directory is None:
            raise FileNotFoundError('File __directory cannot be null!')
        self.__file_directory = file_directory

        # __directory = r'C:\Users\Asus\Documents\ELMo NER\spacy_model'
        # __directory = r'spacy_model'
        __all_subdirs = os.listdir(self.__model_dir)
        __latest_subdir = os.path.join(self.__model_dir, max(__all_subdirs))
        self.__model = self.__load_model(__latest_subdir)
        self.match_finder = MatchFinder(enable_logging=False)

    def __init__(self, model_directory=None, file_directory=None):
        spacy.prefer_gpu()
        if model_directory is not None:
            self.__model_dir = model_directory
        elif self.__model_dir is None:
            raise FileNotFoundError('Model __directory cannot be null!')
        if file_directory is None:
            raise FileNotFoundError('File __directory cannot be null!')
        self.__file_directory = file_directory

        self.match_finder = MatchFinder(enable_logging=False)

        # self.model = self.__load_model(os.path.join(configuration.get('digitization', 'model_directory'),'Spacy3model-1'))
        # self.model2 = self.__load_model(os.path.join(configuration.get('digitization', 'model_directory'),'Spacy3model-2'))

        self.model = self.__load_model(os.path.join(configuration.get('digitization', 'model_directory'),'Spacy3model'))
        print("Models loaded successfully")

    @staticmethod
    def __load_model_spacy2(model_path):
        """ Loads a pre-trained model for prediction on new test sentences

        model_path : directory of model saved by spacy.to_disk
        """
        nlp = spacy.blank('en')
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner)
        print(f"Loading model {model_path}")
        ner = nlp.from_disk(model_path)
        return ner

    
    def __load_model(self, model_path):
        nlp_model = spacy.load(model_path) 
        return nlp_model


    @staticmethod
    def clean_text(text):
        text = re.sub(r'<\s*=?\s*([0-9.,]+)', r' 0-\1', text, flags=re.I)  # Replace '<[Any number of spaces][Any number with . and ,]' with '0-
        text = text.replace("<", "")  # Remove other <
        text = re.sub(r'less than', r' 0-', text, flags=re.I)  # Replace 'less than' with '0-
        text = re.sub(r'\s+up?\s*to\s+', r' 0-', text, flags=re.I)  # Replace 'upto' with '0-
        text = re.sub(r'\s(\.[0-9]+)\s', r'0\1', text, flags=re.I)  # Change .[decimal] to 0.[decimal]
        text = re.sub(r'([0-9.]+)\s*[-~]\s*([0-9.]+)', r'\1-\2', text)  # Convert [number] [-~] [number] to [number]-[number]

        # text = re.sub(r'\s[^\w%<>]\s', r' ', ' ' + text + ' ')  # Remove single special characters except %<>
        # text = re.sub(r'\s[a-wy-zA-WY-Z_]\s', r' ', text)  # Remove single alphabets except xX
        # text = re.sub(r'\s[a-wy-zA-WY-Z_]\s', r' ', text)  # Remove single alphabets except xX again
        # text = re.sub(r'(\s+)[^\w.<>%]+', r'\1 ', text)  # Remove suffixing special characters except .<>%
        # text = re.sub(r'[^\w%<>]+(\s+)', r' \1', text)  # Remove prefixing special characters except %<>
        # text = re.sub(r'([0-9.,]+)(%)', r' \1 \2', text)  # Add a space between a joined number and '%'
        # text = re.sub(r'\s+', r' ', text).strip()  # Remove extra spaces
        # if len(text.split()) == 1:
        #     text = ''
        return text

    def make_predictions_spacy2(self, text=None, file_name=None, return_json=False, return_json_path=False):

        def escape_text(og_text):
            og_text = og_text.replace('\\', r'\\')
            og_text = og_text.replace('(', r'\(')
            og_text = og_text.replace(')', r'\)')
            og_text = og_text.replace('[', r'\[')
            og_text = og_text.replace('|', r'\|')
            og_text = og_text.replace(']', r'\]')
            og_text = og_text.replace('*', r'\*')
            og_text = og_text.replace('.', r'\.')
            og_text = og_text.replace('}', r'\}')
            og_text = og_text.replace('{', r'\{')
            og_text = og_text.replace('+', r'\+')
            og_text = og_text.replace('?', r'\?')
            og_text = og_text.replace('$', r'\$')
            og_text = og_text.replace('^', r'\^')
            return og_text

        __patient_data = dict()
        __predictions = list()
        doc_words = list()
        doc_tags = list()
        start = time.time()
        page_wise_dict = dict()

        html_data = dict()

        for page_idx, page in enumerate(text.values()):
            lines = ""
            test_idx = 0
            for line in page.splitlines():
                line = self.clean_text(line)
                if line != '':
                    html_line = line
                    offset = 0
                    doc = self.__model(" ".join([item for item in re.split(r'(\s+)', line) if item.strip() != '']))
                    entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
                    get_suggestion_data = dict()
                    for entity in entities:
                        spacious_entity = re.sub(" ", r"\s+", escape_text(entity[0]))
                        re_matches = re.findall(fr"{spacious_entity}", line)
                        space_insensitive_matching = re_matches[0] if len(re_matches) > 0 else '-'
                        if entity[1] == 'TST' or entity[1] == 'CAT' or entity[1] == 'DPR' or entity[1] == 'PER' or \
                            entity[1] == 'AGE' or entity[1] == 'SEX' or entity[1] == 'DTE' or entity[1] == 'DCN':
                            test_idx += 1
                        if entity[1] != 'O':
                            if entity[1] == 'TST' or entity[1] == 'REF' or entity[1] == 'UNT' or entity[1] == 'MTD':
                                get_suggestion_data[entity[1]] = space_insensitive_matching
                            if entity[1] == 'TST' or entity[1] == 'RES' or entity[1] == 'REF' or entity[1] == 'UNT' or \
                                    entity[1] == 'MTD' or entity[1] == 'CAT':
                                if entity[1] == 'TST':
                                    tagged_line = f"""<mark class="noselect html-test-item {test_idx}" id="{test_idx}" data-entity="{entity[1].lower()}"><span class="text">{space_insensitive_matching}</span><div class=\'hover-options\'><div style="display: flex; flex-direction: column;"><i class=\"fas fa-times-circle pb-1\" style=\'color: red;\' onclick=\"deleteTag(this);\"></i><i class=\"fas fa-trash\" style=\'color: red;\' onclick=\"deleteSet(this);\"></i></div></div></mark>"""
                                else:
                                    tagged_line = f"""<mark class="noselect html-test-item {test_idx}" id="{test_idx}" data-entity="{entity[1].lower()}"><span class="text">{space_insensitive_matching}</span><div class=\'hover-options\'><div style="display: flex; flex-direction: column;"><i class=\"fas fa-times-circle\" style=\'color: red;\' onclick=\"deleteTag(this);\"></i></div></div></mark>"""
                            else:
                                tagged_line = f"""<mark class="noselect html-test-item {test_idx}" id="{test_idx}" data-entity="{entity[1].lower()}"><span class="text">{space_insensitive_matching}</span><div class=\'hover-options\'><div style="display: flex; flex-direction: column;"><i class=\"fas fa-times-circle\" style=\'color: red;\' onclick=\"deleteTag(this);\"></i></div></div></mark>"""
                            html_line = html_line[:entity[2] + offset] + html_line[entity[2] + offset:].replace(space_insensitive_matching, tagged_line, 1)
                            offset += len(tagged_line) - len(space_insensitive_matching)

                    if len(get_suggestion_data.items()) > 0:
                        suggs = self.match_finder.find_top_n_matches(get_suggestion_data)
                        if suggs is not None and len(suggs.keys()) > 0:
                            suggs = suggs.head(1)
                            suggs = suggs.to_dict()
                            suggs.pop('LCL', None)
                            suggs.pop('UCL', None)
                            for tag, val in suggs.items():
                                r_text = list(val.values())
                                if len(r_text) > 0 and r_text[0] != '':
                                    html_line = re.sub(fr'(data-entity="{tag.lower()}">)[\w\W]+?(</span>)', fr'\1<span class="text">{r_text[0]}</span>\2', html_line)
                    lines += html_line + "\n"
            html_data[page_idx + 1] = lines

        if len(file_name.rsplit('.', 1)[0].replace("\\", "/").rsplit('/', 1)) > 1:
            folder_structure = file_name.rsplit('.', 1)[0].replace("\\", "/").rsplit('/', 1)[0]
        else:
            folder_structure = ''

        os.makedirs(os.path.join('Reports/extracted_html_files', folder_structure.replace("\\", "/")), exist_ok=True)

        with open(os.path.join('Reports/extracted_html_files', file_name.rsplit('.', 1)[0]).replace("\\", "/") + ".json", 'w') as f:
            json.dump(html_data, f, indent=4)

        os.makedirs(os.path.join('Reports/extracted_html_files_bak', folder_structure), exist_ok=True)

        with open(os.path.join('Reports/extracted_html_files_bak', file_name.rsplit('.', 1)[0]) + ".json", 'w') as f:
            json.dump(html_data, f, indent=4)

        for key, file_content in text.items():
            for line in file_content.splitlines():
                cleaned_line = self.clean_text(line)
                if cleaned_line != '':
                    words_line = ''
                    tags_line = ''
                    doc = self.__model(cleaned_line)
                    entities = [(ent.text, ent.label_) for ent in doc.ents]
                    for entity in entities:
                        word_line, tag_line = entity
                        item_words = word_line.split()
                        if tag_line.strip() == 'O':
                            item_tag = ''
                            item_tag += ' O' * len(item_words)
                        else:
                            item_tag = ' B-' + tag_line
                            item_tag += (' I-' + tag_line) * (len(item_words) - 1)
                        words_line += ' ' + word_line.strip()
                        tags_line += ' ' + item_tag.strip()
                    doc_words.extend(words_line.split())
                    doc_tags.extend(tags_line.split())

            currently_active_tag = None
            active_start_pos = -1
            active_end_pos = -1
            detected_tags = list()
            for idx, tag in enumerate(doc_tags):
                tag_type = tag.split('-')[-1]
                if tag_type != currently_active_tag:
                    if currently_active_tag is not None:
                        detected_tags.append(dict(tagType=currently_active_tag,
                                                  text=" ".join(doc_words[active_start_pos: active_end_pos + 1]),
                                                  startPos=active_start_pos, endPos=active_end_pos))
                    currently_active_tag = tag_type
                    active_start_pos = idx
                    active_end_pos = idx
                else:
                    active_end_pos = idx

            if currently_active_tag is not None:
                detected_tags.append(dict(tagType=currently_active_tag,
                                          text=" ".join(doc_words[active_start_pos: active_end_pos + 1]),
                                          startPos=active_start_pos, endPos=active_end_pos))

            detected_tags = list(filter(lambda x: x['tagType'] != 'O', detected_tags))

            meta_data_tags = ['PER', 'SEX', 'AGE', 'DPR', 'DCN', 'DTE']
            stop_tags = meta_data_tags + ['TST', 'CAT']

            test_list = list()
            test_item = TestItem()
            currently_active_tag = None
            for tag in detected_tags:
                if tag['tagType'] in stop_tags:
                    currently_active_tag = tag['tagType']
                    if test_item is not None and not test_item.is_empty():
                        if test_item.get_category_name() != '':
                            test_item.set_type(TestItemType.CATEGORY)

                        test_item = test_item.to_dict()
                        if test_item is not None:
                            test_list.append(test_item)
                    test_item = TestItem()
                if currently_active_tag in meta_data_tags:
                    self.__process_meta_data(currently_active_tag, tag['text'], __patient_data)
                else:
                    test_item.set_data(tag['tagType'], tag['text'])

            test_item = test_item.to_dict()
            if test_item is not None:
                test_list.append(test_item)

            page_wise_dict[key] = test_list

        end = time.time()
        print(end - start)
        __patient_data['TESTS'] = page_wise_dict
        os.makedirs('Reports/extracted_json_files', exist_ok=True)
        os.makedirs(os.path.join('Reports/extracted_json_files/', folder_structure), exist_ok=True)

        file_path = os.path.join('Reports/extracted_json_files', file_name.rsplit('.', 1)[0] + ".json")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(__patient_data, indent=4))
        if return_json and return_json_path:
            return __patient_data, file_path
        elif return_json:
            return __patient_data
        elif return_json_path:
            return file_path


    def linewise_tags(self, page, ent_log):
        line_tag_dict = {}
        lines = [self.clean_text(line) for line in page.splitlines()]
        line_ind_checked = 0
        tag_found_count = 0
        cummulative_line_start_forent = 0
        for ent in ent_log:
            txt, tag, startind, endind = ent
            
            cummulative_line_startind = cummulative_line_start_forent
            line_no = 0
            for line in lines[line_ind_checked:]:
                temp_ent_list = []
                if line_ind_checked+line_no not in line_tag_dict.keys():
                    line_tag_dict[line_ind_checked+line_no] = [line]
                    
                line_lenght = len(line)
                line_start , line_end = cummulative_line_startind, cummulative_line_startind+line_lenght
                if startind >= line_start and startind <=line_end:
                    rel_starind = startind-cummulative_line_startind
                    rel_endind = rel_starind + len(txt)
                    ent_for_line = (txt, tag, rel_starind, rel_endind)
                    temp_ent_list.append(ent_for_line)
                    tag_found_count +=1
                    cummulative_line_start_forent = line_start                   
                    
                    line_tag_dict[line_ind_checked+line_no].extend(temp_ent_list)
                    line_ind_checked += line_no
                    break

                cummulative_line_startind = line_end+1
                line_no +=1
        
        for i, line in enumerate(lines):
            if i not in line_tag_dict.keys():
                line_tag_dict[i] = [line]

        print(tag_found_count)
        return line_tag_dict


    def make_predictions(self, text=None, file_name=None, return_json=False, return_json_path=False):

        def escape_text(og_text):
            og_text = og_text.replace('\\', r'\\')
            og_text = og_text.replace('(', r'\(')
            og_text = og_text.replace(')', r'\)')
            og_text = og_text.replace('[', r'\[')
            og_text = og_text.replace('|', r'\|')
            og_text = og_text.replace(']', r'\]')
            og_text = og_text.replace('*', r'\*')
            og_text = og_text.replace('.', r'\.')
            og_text = og_text.replace('}', r'\}')
            og_text = og_text.replace('{', r'\{')
            og_text = og_text.replace('+', r'\+')
            og_text = og_text.replace('?', r'\?')
            og_text = og_text.replace('$', r'\$')
            og_text = og_text.replace('^', r'\^')
            return og_text

        # tag_list_model1 = ['PERSON', 'DATE', 'AGE', 'DPR', 'SEX'] 
        # tag_list_model2 = ['CAT', 'TST', 'RES', 'REF', 'UNT', 'DPR', 'MTD']
        
        html_data = dict()

        for page_idx, page in enumerate(text.values()):
            lines = ""
            test_idx = 0
            new_page = page
            value = self.clean_text(page)

            # doc1 = self.model(value)
            # ent_log1 = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc1.ents]

            # for ent1 in ent_log1:
            #     if 'REF' in ent1[1] :
            #         list_of_words = ent1[0].split(' ')
            #         list_wrd_to_replace = []
            #         for word_to_chk in list_of_words:
            #             suggested = self.match_finder.match_finder_ref({'REF': word_to_chk})
            #             if suggested is not None:
            #                 suggested = suggested.to_list()[0]
            #                 list_wrd_to_replace.append(suggested)
            #             else:
            #                 list_wrd_to_replace.append(word_to_chk)
                    
            #         word_to_replace = " ".join(list_wrd_to_replace)

            #         if len(word_to_replace) > len(ent1[0]):
            #             first_half = page[:ent1[2]]
            #             first_half = first_half + (len(word_to_replace)- len(ent1[0]))*' '
            #             new_page = first_half + page[ent1[2]:]
            #             page_partition = list(new_page.partition(page[ent1[2]:ent1[2]+len(word_to_replace)]))
            #             page_partition[1] = word_to_replace
            #             new_page = ''.join(page_partition)
            #         elif len(word_to_replace) < len(ent1[0]):
            #             first_half = page[:ent1[3]]
            #             diff = len(ent1[0]) - len(word_to_replace)
            #             first_half = first_half[:-1*diff]
            #             new_page = first_half + page[ent1[2]:]
            #             page_partition = list(new_page.partition(page[ent1[2]:ent1[2]+len(word_to_replace)]))
            #             page_partition[1] = word_to_replace
            #             new_page = ''.join(page_partition)
            #         else:
            #             page_partition = list(new_page.partition(page[ent1[2]:ent1[3]]))
            #             page_partition[1] = word_to_replace
            #             new_page = ''.join(page_partition)

            # print("new_page-----------", new_page)

            doc = self.model(value)
            ent_log = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]


            dict1 = self.linewise_tags(page, ent_log)

            # for line in page.splitlines():
            #     entities =[]
            #     line = self.clean_text(line)
            #     if line != '':
            #         html_line = line
            #         offset = 0

            #         item_list = [item for item in re.split(r'(\s+)', line) if item.strip() != '']
            #         doc = self.model(" ".join(item_list))
            #         ent_list = [str(i) for i in doc.ents]

            
            # for line_txt, ents in dict1.items():
            
            for line_no , list_line_ents in dict1.items():
                if len(list_line_ents) > 1:
                    ents = list_line_ents[1:]
                else:
                    ents = []
                line_txt = list_line_ents[0]
                offset = 0
                item_list = [item for item in re.split(r'(\s+)', list_line_ents[0]) if item.strip() != '']
                ent_list = [str(i[0]) for i in ents]
                entities = []

                for ent in ents:
                    tagtxt, tag, startind, endind = ent
                    if tag == 'PERSON':
                        label_to_add = 'PER'
                    elif tag == 'DATE':
                        label_to_add = 'DTE'
                    else: 
                        label_to_add = tag

                    if  tag == 'REF' and '-' not in tagtxt.strip():
                        refc = tagtxt.strip().split()
                        txt_ref = ""
                        for i, value in enumerate(refc):
                            if i > 0:
                                if value.isalpha() or refc[i-1].isalpha():
                                    pass
                                else :
                                    txt_ref += "-"
                                    
                            txt_ref += value + " "

                        txt_to_add = txt_ref
                        line_txt = line_txt.replace(tagtxt, txt_to_add)
                    else:
                        txt_to_add = tagtxt

                    t1 = (txt_to_add, label_to_add, startind, endind)

                    entities.append(t1)


                if 'TST' in [i[1] for i in ents] and 'RES' not in [i[1] for i in ents]:
                    for item in item_list:
                        if item not in ent_list and str(item).lower().strip() == 'o':
                            ind = item_list.index(item)
                            strtind = line_txt.index('o')
                            endind = strtind + 1
                            txt_to_add = '0'
                            label_to_add = 'RES'
                            line_txt = line_txt.replace(str(item), txt_to_add)

                            t1 = (txt_to_add, label_to_add, strtind, endind)
                            entities.insert(ind, t1)
                    
                html_line = line_txt

                ############# Micro corrections to the tag and words ##############
                get_suggestion_data = dict()
                for entity in entities:
                    spacious_entity = re.sub(" ", r"\s+", escape_text(entity[0]))
                    re_matches = re.findall(fr"{spacious_entity}", line_txt)
                    space_insensitive_matching = re_matches[0] if len(re_matches) > 0 else '-'
                    if entity[1] == 'TST' or entity[1] == 'CAT' or entity[1] == 'DPR' or entity[1] == 'PER' or \
                        entity[1] == 'AGE' or entity[1] == 'SEX' or entity[1] == 'DTE' or entity[1] == 'DCN':
                        test_idx += 1
                    if entity[1] != 'O':
                        if entity[1] == 'TST' or entity[1] == 'REF' or entity[1] == 'UNT' or entity[1] == 'MTD':
                            get_suggestion_data[entity[1]] = space_insensitive_matching
                        if entity[1] == 'TST' or entity[1] == 'RES' or entity[1] == 'REF' or entity[1] == 'UNT' or \
                                entity[1] == 'MTD' or entity[1] == 'CAT':
                            if entity[1] == 'TST':
                                tagged_line = f"""<mark class="noselect html-test-item {test_idx}" id="{test_idx}" data-entity="{entity[1].lower()}"><span class="text">{space_insensitive_matching}</span><div class=\'hover-options\'><div style="display: flex; flex-direction: column;"><i class=\"fas fa-times-circle pb-1\" style=\'color: red;\' onclick=\"deleteTag(this);\"></i><i class=\"fas fa-trash\" style=\'color: red;\' onclick=\"deleteSet(this);\"></i></div></div></mark>"""
                            else:
                                tagged_line = f"""<mark class="noselect html-test-item {test_idx}" id="{test_idx}" data-entity="{entity[1].lower()}"><span class="text">{space_insensitive_matching}</span><div class=\'hover-options\'><div style="display: flex; flex-direction: column;"><i class=\"fas fa-times-circle\" style=\'color: red;\' onclick=\"deleteTag(this);\"></i></div></div></mark>"""
                        else:
                            tagged_line = f"""<mark class="noselect html-test-item {test_idx}" id="{test_idx}" data-entity="{entity[1].lower()}"><span class="text">{space_insensitive_matching}</span><div class=\'hover-options\'><div style="display: flex; flex-direction: column;"><i class=\"fas fa-times-circle\" style=\'color: red;\' onclick=\"deleteTag(this);\"></i></div></div></mark>"""
                        html_line = html_line[:entity[2] + offset] + html_line[entity[2] + offset:].replace(space_insensitive_matching, tagged_line, 1)
                        offset += len(tagged_line) - len(space_insensitive_matching)

                ############# Check Lavenstein's distances and replace from csv file ##############

                if len(get_suggestion_data.items()) > 0:
                    suggs = self.match_finder.find_top_n_matches(get_suggestion_data)
                    if suggs is not None and len(suggs.keys()) > 0:
                        suggs = suggs.head(1)
                        suggs = suggs.to_dict()
                        suggs.pop('LCL', None)
                        suggs.pop('UCL', None)
                        for tag, val in suggs.items():
                            r_text = list(val.values())
                            if len(r_text) > 0 and r_text[0] != '':
                                html_line = re.sub(fr'(data-entity="{tag.lower()}">)[\w\W]+?(</span>)', fr'\1<span class="text">{r_text[0]}</span>\2', html_line)
                lines += html_line + "\n"
            html_data[page_idx + 1] = lines

        if len(file_name.rsplit('.', 1)[0].replace("\\", "/").rsplit('/', 1)) > 1:
            folder_structure = file_name.rsplit('.', 1)[0].replace("\\", "/").rsplit('/', 1)[0]
        else:
            folder_structure = ''

        os.makedirs(os.path.join('Reports/extracted_html_files', folder_structure.replace("\\", "/")), exist_ok=True)

        with open(os.path.join('Reports/extracted_html_files', file_name.rsplit('.', 1)[0]).replace("\\", "/") + ".json", 'w') as f:
            json.dump(html_data, f, indent=4)
        

        os.makedirs(os.path.join('Reports/extracted_html_files_bak', folder_structure), exist_ok=True)

        with open(os.path.join('Reports/extracted_html_files_bak', file_name.rsplit('.', 1)[0]) + ".json", 'w') as f:
            json.dump(html_data, f, indent=4)



    def __process_meta_data(self, tag, text, __patient_data):
        if tag in ['PER', 'DPR', 'SEX'] and self.__is_number(text):
            return
        elif tag in ['AGE'] and self.__is_number(text) and len(text) > 2:
            return
        elif tag in ['DTE'] and not self.__is_date(text):
            return
        text = re.sub(r'[^A-Z0-9.\s]', '', text.upper())
        text = re.sub(r'\s{0}\s'.format(tag), '', ' ' + text + ' ').strip()
        if text is not None and len(text) > 0 and text != tag:
            if tag not in __patient_data.keys():
                __patient_data[tag] = OrderedDict()
            if text in __patient_data[tag].keys():
                __patient_data[tag][text] += 1
            else:
                __patient_data[tag][text] = 1
            __patient_data[tag] = OrderedDict(
                sorted(__patient_data[tag].items(), key=operator.itemgetter(1), reverse=True))

    @staticmethod
    def __is_number(s):
        try:
            float(re.sub(r'[^\w.]', '', s).strip())
            return True
        except ValueError:
            return False

    @staticmethod
    def __is_date(string, fuzzy=False):
        """
        Return whether the string can be interpreted as a date.

        :param string: str, string to check for date
        :param fuzzy: bool, ignore unknown tokens in string if True
        """
        try:
            parse(str(string), fuzzy=fuzzy)
            return True
        except:
            return False


# class Trainer:

#     def __init__(self, dataset_dir='datasets'):
#         spacy.prefer_gpu()
#         self.dataset_dir = dataset_dir
#         self.nlp = spacy.load('en_core_web_sm')
#         self.ner = self.nlp.get_pipe("ner")
#         self.optimizer = self.nlp.begin_training()
#         self.phrase_list = list()
#         print("Ready to load data...")

#     @staticmethod
#     def find_files_recursively(dir_path):
#         files_to_process = list()
#         for root, dirs, files in os.walk(dir_path):
#             for file in files:
#                 files_to_process.append(os.path.join(root, file))
#         return files_to_process

#     def load_datasets(self, dataset_dirs=None, data_files=None, tags_to_ignore=None):
#         if tags_to_ignore is None:
#             tags_to_ignore = []

#         files = []
#         if dataset_dirs is None and data_files is None:
#             raise ValueError("Please pass either dataset_dirs or data_files")
#         if dataset_dirs is not None:
#             for dataset_dir in dataset_dirs:
#                 files.append(self.find_files_recursively(os.path.join(os.getcwd(), dataset_dir)))
#         if data_files is not None:
#             files = data_files

#         print(f"Found {len(files)} files.")
#         for file in files:
#             all_words_and_tags = open(file.rsplit('.', 1)[0] + ".txt", 'r').readlines()
#             with tqdm(total=len(all_words_and_tags)) as pbar:
#                 for words_tag in all_words_and_tags:
#                     word_tag_split = words_tag.split("    ")
#                     if len(word_tag_split) == 3:
#                         sentence = word_tag_split[0].strip()
#                         word = word_tag_split[1].strip()
#                         tag = word_tag_split[2]
#                         split_words = word.strip().split("   ")
#                         split_tags = tag.strip().split(" ")
#                         start = 0
#                         entities_list = list()
#                         ignore_line = False
#                         for idx, pair in enumerate(zip(split_words, split_tags)):
#                             index = sentence.find(pair[0], start)
#                             last_index = index + len(pair[0])
#                             start = index + len(pair[0])
#                             if pair[1] not in tags_to_ignore:
#                                 entities_list.append((index, last_index, pair[1]))
#                             else:
#                                 ignore_line = True
#                         if not ignore_line:
#                             entities = (sentence, {'entities': entities_list})
#                             self.phrase_list.append(entities)
#                     pbar.update(1)
#         self.__annotate_data()

#     def __annotate_data(self):
#         print("Annotating data...")
#         for _, annotations in self.phrase_list:
#             for ent in annotations.get("entities"):
#                 self.ner.add_label(ent[2])
#         print("Ready to begin training...")

#     @staticmethod
#     def __evaluate(ner_model, examples):
#         scorer = Scorer()
#         for input_, annot in examples:
#             doc_gold_text = ner_model.make_doc(input_)
#             text_entities = []
#             for entity in annot.get('entities'):
#                 text_entities.append(entity)
#             gold = GoldParse(doc_gold_text, entities=text_entities)
#             pred_value = ner_model(input_)
#             scorer.score(pred_value, gold)
#         return scorer.scores

#     @staticmethod
#     def __split_train_test(phrase_list, ratio):
#         train_set = phrase_list[:int(len(phrase_list) * ratio)]
#         test_set = phrase_list[int(len(phrase_list) * ratio) + 1:]
#         return train_set, test_set

#     def train(self, epochs=10, split_ratio=0.90):
#         pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
#         unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

#         with self.nlp.disable_pipes(*unaffected_pipes):
#             for itn in range(epochs):
#                 random.shuffle(self.phrase_list)
#                 train_set, test_set = self.__split_train_test(self.phrase_list, split_ratio)
#                 losses = {}
#                 batches = minibatch(train_set, size=compounding(4.0, 128.0, 1.005))
#                 drop_rate = decaying(0.6, 0.2, 1e-4)
#                 with tqdm(total=len(train_set)) as pbar:
#                     for batch in batches:
#                         texts, annotations = zip(*batch)
#                         try:
#                             self.nlp.update(texts, annotations, sgd=self.optimizer, drop=next(drop_rate), losses=losses)
#                             pbar.update(len(texts))
#                         except Exception as e:
#                             print(e, list(zip(texts, annotations)))
#                 eval_metrics = self.__evaluate(self.nlp, test_set)
#                 entity_metrics = eval_metrics['ents_per_type']
#                 print("Evaluating EPOCH {itn + 1} on {len(test_set)} records...\nEvaluation metrics:")
#                 for tag, nth_entity_metric in entity_metrics.items():
#                     print(f"{tag} ---> p: {round(nth_entity_metric['p'], 2)}, r: {round(nth_entity_metric['r'], 2)}, f1: {round(nth_entity_metric['f'], 2)}")
#                 print(f"EPOCH {itn + 1} completed.")
#         model_name = str(int(time.time()))
#         self.nlp.to_disk(f"spacy_model/{model_name}")