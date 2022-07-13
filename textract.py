import boto3
import time
from configparser import ConfigParser
import numpy as np
import os
import cv2 as cv2
from PIL import Image
import mimetypes

configuration = ConfigParser()
configuration.read('config.ini')

pages_dict = {}
lines =[]
imagelist = []

textract = boto3.client(
    'textract',
    region_name = str(configuration.get('textract', 'textract_region_name')),
    aws_access_key_id = str(configuration.get('textract', 'textract_access_key_id')),
    aws_secret_access_key = str(configuration.get('textract', 'textract_aws_secret_access_key'))
)

s3 = boto3.client(
    's3',
    region_name = str(configuration.get('s3', 'region_name')),
    aws_access_key_id = str(configuration.get('s3', 'aws_access_key_id')),
    aws_secret_access_key = str(configuration.get('s3', 'aws_secret_access_key'))
)

s3_ = boto3.resource('s3',
    region_name = str(configuration.get('s3', 'region_name')),
    aws_access_key_id = str(configuration.get('s3', 'aws_access_key_id')),
    aws_secret_access_key = str(configuration.get('s3', 'aws_secret_access_key'))
)


def startJob(objectName):
    response = None
    response = textract.start_document_text_detection(
    DocumentLocation={
        'S3Object': {
            'Bucket': str(configuration.get('s3', 'bucket_name')),
            'Name': objectName
        }
    })

    return response["JobId"]

def isJobComplete(jobId):
    time.sleep(5)
    response = textract.get_document_text_detection(JobId=jobId)
    status = response["JobStatus"]
    print("Job status: {}".format(status))

    while(status == "IN_PROGRESS"):
        time.sleep(5)
        response = textract.get_document_text_detection(JobId=jobId)
        status = response["JobStatus"]
        print("Job status: {}".format(status))

    return status

def getJobResults(jobId):
    pages = []
    time.sleep(5)
    response = textract.get_document_text_detection(JobId=jobId)
    
    pages.append(response)
    print("Resultset page recieved: {}".format(len(pages)))
    nextToken = None
    if('NextToken' in response):
        nextToken = response['NextToken']

    while(nextToken):
        time.sleep(5)

        response = textract.get_document_text_detection(JobId=jobId, NextToken=nextToken)

        pages.append(response)
        print("Resultset page recieved: {}".format(len(pages)))
        nextToken = None
        if('NextToken' in response):
            nextToken = response['NextToken']

    return pages


# def extract_main(file_path):
#     global pages_dict
#     pages_dict = {}
#     prev = None
#     jobId = startJob(file_path)
#     print("Started job with id: {}".format(jobId))
#     if(isJobComplete(jobId)):
#         response = getJobResults(jobId)
#         for resultPage in response:
#             for item in resultPage["Blocks"]:
#                 curr = item
#                 if item["BlockType"] == "LINE":
#                     # pages_dict[int(item['Page'])] = printlines(curr,prev,int(item['Page']))
#                     # print ('\033[94m' +  item["Text"] + '\033[0m')
#                     if int(item['Page']) not in pages_dict.keys():
#                         pages_dict[int(item['Page'])] = ''
#                     # pages_dict[int(item['Page'])] = pages_dict[int(item['Page'])] + item["Text"] + '\n'
#                     CurrBox = curr['Geometry']['BoundingBox']['Top']
#                     if prev is not None:
#                         PrevBox = prev['Geometry']['BoundingBox']['Top']
#                         if(abs(round((CurrBox-PrevBox),2))) == 0:
#                             pages_dict[int(item['Page'])] = pages_dict[int(item['Page'])] + ' ' +curr["Text"]
#                         else :
#                             pages_dict[int(item['Page'])] = pages_dict[int(item['Page'])] + '\n' + curr["Text"] 
#                     prev = curr
#         print("PAGES DICT ----> ",pages_dict)
#         return


def extract_main(file_path):
    global pages_dict
    pages_dict = {}
    prev = None

    newFilePath = rotatePdf(file_path)
    if newFilePath is not None:
        pages_dict = {}
        print("pages dict length (extract_main) -------- > ")
        print(len(pages_dict))
        print('')
        jobId=startJob(newFilePath)

        print("Started job with id: {}".format(jobId))
        if(isJobComplete(jobId)):
            response = getJobResults(jobId)
            for resultPage in response:
                for item in resultPage["Blocks"]:
                    curr = item
                    if item["BlockType"] == "LINE":
                        if int(item['Page']) not in pages_dict.keys():
                            pages_dict[int(item['Page'])] = ''

                        CurrBox = curr['Geometry']['BoundingBox']['Top']
                        if prev is not None:
                            PrevBox = prev['Geometry']['BoundingBox']['Top']
                            if(abs(round((CurrBox-PrevBox),2))) == 0:
                                pages_dict[int(item['Page'])] = pages_dict[int(item['Page'])] + ' ' +curr["Text"]
                            else :
                                pages_dict[int(item['Page'])] = pages_dict[int(item['Page'])] + '\n' + curr["Text"] 
                        prev = curr

            print("pages dict length (After rotation) -------- > ")
            print(len(pages_dict))
            print('')
            print("PAGES DICT (After rotation) ---------***********----------> ",pages_dict)           

    return 


def rotatePdf(file_path):
    pages = dict()
    prev = None
    global pages_dict
    pages_dict = {}

    jobId=startJob(file_path)

    pdf_name = os.path.basename(file_path)
    pdf_name_wo_ext = pdf_name.split(".")[0]
    folder_key = 'Reports/extracted_png/' + pdf_name_wo_ext
    print("folder_key -> ",folder_key)

    bucketname = str(configuration.get('s3', 'bucket_name'))
    print("Bucketname -> ",bucketname)

    try:
        result = s3.list_objects_v2(Bucket=configuration.get('s3', 'bucket_name'), Prefix=folder_key)
    except Exception as e:
        print("exception in list_object_v2 ", e)
    
    if 'Contents' not in result.keys():
        print("Png path -------- > ", os.path.join(configuration.get('digitization','default_reporting_directory'),configuration.get('digitization', 'extracted_pages_path')))
        upload_dir_to_s3(os.path.join(configuration.get('digitization','default_reporting_directory'),configuration.get('digitization', 'extracted_pages_path')))

        print("Pngs uploaded to S3--------------------")

    if(isJobComplete(jobId)):
        response = getJobResults(jobId)
        for resultPage in response:
            for item in resultPage["Blocks"]:
                curr = item
                if item["BlockType"] == "LINE":
                    if int(item['Page']) not in pages_dict.keys():
                        pages_dict[int(item['Page'])] = ''

                    CurrBox = curr['Geometry']['BoundingBox']['Top']
                    if prev is not None:
                        PrevBox = prev['Geometry']['BoundingBox']['Top']
                        if(abs(round((CurrBox-PrevBox),2))) == 0:
                            pages_dict[int(item['Page'])] = pages_dict[int(item['Page'])] + ' ' +curr["Text"]
                        else :
                            pages_dict[int(item['Page'])] = pages_dict[int(item['Page'])] + '\n' + curr["Text"] 
                    prev = curr


                    if int(item['Page']) not in pages.keys():
                        pages[int(item['Page'])] = ''
                        angles = []
                        filename = configuration.get('digitization', 'default_reporting_directory') + '/' +configuration.get('digitization', 'extracted_pages_path') \
                                + '/' + os.path.split(file_path)[-1].split('.')[0] + '/' + os.path.split(file_path)[1].split('.')[0] + '-' + str(int(item['Page'])) + '.png'

                        pngPath = os.path.join("Reports", "downloaded_pngs")
                        try:
                            os.makedirs(pngPath, exist_ok=True)
                        except Exception as e:
                            print(e)

                        file_png = os.path.join(pngPath, os.path.split(filename)[-1])
                        print("File_png Path", file_png)
                        print("Key file Path ", filename)
                        getPngFromS3(filename, file_png)
                        print("Saved png file   ")
                        img = cv2.imread(file_png)

                    X1, Y1 = getCoordinates(img, item['Geometry']['Polygon'][0]['X'], item['Geometry']['Polygon'][0]['Y']) 
                    X2, Y2 = getCoordinates(img, item['Geometry']['Polygon'][1]['X'], item['Geometry']['Polygon'][1]['Y']) 

                    angle = slope(X1, Y1, X2, Y2)
                
                    angles.append(round(angle,3))
                    pages[int(item['Page'])] = angles
        

        print("pages dict length (Before rotation) -------- > ")
        print(len(pages_dict))
        print('')
        print("PAGES DICT (Before rotation) ---------***********----------> ",pages_dict)         
        

    if max([abs(max(pg[1])) for pg in pages.items()]) == 0:
        return None
    else: 
        pages_dict.clear()
        print("pages dict Cleared -------- > ")
        print("Length --- ",len(pages_dict))
        print('')

        global imagelist
        imagelist = []
        
        for page in pages.items():
            theta = max(page[1], key=page[1].count)
            
            print(page[0],"--", theta)
            rotateImg(theta, file_path, page[0])                

        imglist = imagelist[1:]
        newPDF = "Modified_" + os.path.split(file_path)[-1]
        imagelist[0].save(newPDF, save_all= True, append_images = imglist)

        key = configuration.get('digitization', 'default_reporting_directory') + '/' + configuration.get('digitization', 'output_processed_path')+ '/' + newPDF
        s3.upload_file(newPDF, Bucket = configuration.get('s3', 'bucket_name'), Key = key)
        os.remove(newPDF)

        for f in os.listdir(pngPath):
            os.remove(os.path.join(pngPath, f))
            
        return key


def slope(x1,y1,x2,y2):
    if x1 == x2:
        return 0
    else:
        slope = (y2-y1)/(x2-x1)
        theta = np.rad2deg(np.arctan(slope))
        return theta


def getCoordinates(img, xNorm, yNorm):
    hPage, wPage, _ = img.shape
    if hPage is not None and wPage is not None:
        x = xNorm*wPage
        y = yNorm*hPage

    return x, y


def getPngFromS3(key,file_png_path):
    s3_.Bucket(configuration.get('s3','bucket_name')).download_file(Key = key, Filename=file_png_path)
    

def rotateImg(theta, path, page_num):
    filename = os.path.split(path)[-1].split('.')[0] + "-" + str(page_num) + ".png"

    s3Path = configuration.get('digitization', 'default_reporting_directory') + '/' +configuration.get('digitization', 'extracted_pages_path') + '/' + os.path.split(path)[-1].split('.')[0] + '/' + filename
    pngPath = os.path.join("Reports", "downloaded_pngs")
    file_png = os.path.join(pngPath, os.path.split(s3Path)[-1])

    temp_path = os.path.join("Reports", "modified_pngs")  
    os.makedirs(temp_path, exist_ok=True)
   
    for f in os.listdir(temp_path):
        os.remove(os.path.join(temp_path, f))
    
    newFile = "Rotated_" + os.path.split(path)[-1].split('.')[0] + ".png"
    savePath = os.path.join(temp_path, newFile)

    img = cv2.imread(file_png)
    img = cv2.resize(img, dsize=None, fx = 0.1, fy= 0.1)

    h, w = img.shape[0:2]
    center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, angle=theta, scale=1)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(savePath, rotated)

    png2pdf(savePath)


def png2pdf(image):
    img = Image.open(image)
    img.thumbnail((765,990))
    im = img.convert('RGB')
    imagelist.append(im)

def upload_dir_to_s3(path=None):
        for subdir, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(subdir, file)
                file_mime = mimetypes.guess_type(file)[0] or 'binary/octet-stream'
                with open(full_path, 'rb') as data:
                    s3.put_object(Key=full_path, Body=data, ContentType=file_mime, Bucket=configuration.get('s3', 'bucket_name'))
                    os.remove(full_path)