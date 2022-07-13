import schedule
import time
import database_credentials
from digitization import ConvertToTxt, ExtractToJson
import MySQLdb

from queue import Queue
from threading import Thread
import traceback


# TODO - Change these SPs for the new server
reports_get_by_status = "CALL reports_get_by_status('{status}');"
# process_progress_insert = "SET @x = 0; CALL process_progress_insert(@x, {report_id}, '{status}', '{time_taken}', '{comment}'); SELECT @x;"

# These have been changed
# report_upsert = "SET @x = {rec_id}; CALL reports_upsert(@x, '{report_name}', {page_count}, '{stage}'); SELECT @x;"
reports_update_page_count = "CALL reports_update_page_count({report_id}, {no_of_pages});"
report_update_status = "CALL report_update_status({report_id}, '{stage}', {user_id}, '{time_taken}', '{comment}');"

process_queue = []
process_batch = 1
under_process = []

ctxt = ConvertToTxt(directory_path=r'Reports', thread_limit=1)
print("Object created for ConvertToTxt")
ejson = ExtractToJson(file_directory='Reports/extracted_text_files')
print("Object created for ExtractToJson")


def exec_query(sql, params):
    try:
        sql = sql.format(**params)
        print(sql)
        db = MySQLdb.connect(host=database_credentials.host, user=database_credentials.user,
                  passwd=database_credentials.password, db=database_credentials.database, port=database_credentials.port)
        db.autocommit(1)
        mysql_response = []
        cursor = db.cursor()
        for statement in sql.split(';'):
            if len(statement.strip()) > 0:
                cursor.execute(statement.strip() + ';')
                mysql_response.append(cursor.fetchall())
        cursor.close()
        return mysql_response
    except Exception as e:
        print(e)


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
        # print("Adding task -", args)
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()

##################### To reprocess errored out files ###################
sql_response = exec_query(reports_get_by_status, {'status': 'Under Processing'})

if len(sql_response) !=0 :
    for i in sql_response[0]:
        exec_query(report_update_status, {'report_id': i[0], 'stage': 'In Queue', 'user_id': 0, 'time_taken': '0', 'comment': ''})
##################### To reprocess errored out files ###################

thread_pool = ThreadPool(process_batch)


def process_report(item):

    # exec_query(report_upsert,
    #            {'rec_id': item[0], 'report_name': item[1], 'page_count': 0, 'stage': 'Under Processing'})
    exec_query(report_update_status,
               {'report_id': item[0], 'stage': 'Under Processing', 'user_id': 0, 'time_taken': '0', 'comment': ''})
    under_process.append(item)
    start = time.time()
    try:
        files = ctxt.get_files_by_name_or_count_aws_s3(file_name=item[1] + ".pdf")
    except Exception as e:
        print(traceback.format_exc())
        print({"error":"Error getting files from S3, please try again", "raw": str(e)})
        return
    try:
        filename, txt, page_count = ctxt.process_files(files[0], save_to_txt=True, return_page_count=True)
        ejson.make_predictions(text=txt, file_name=filename + '.txt', return_json_path=True)
        end = time.time()
        ctxt.upload_results_to_s3(filename=filename)
        exec_query(reports_update_page_count,
                {'report_id': item[0], 'no_of_pages': page_count})
        exec_query(report_update_status,
                {'report_id': item[0], 'stage': 'Digitized', 'user_id': 0, 'time_taken': str(end - start), 'comment': ''})
        under_process.remove(item)
    
    ##################### To reprocess errored out files ###################
    except Exception as e:
        under_process.remove(item)
        print("Error in processing report {}".format(files[0]))
        print("Exception : ",e)


def get_queue():
    global under_process
    try:
        if len(process_queue) < process_batch:
            mysql_response = exec_query(reports_get_by_status, {'status': 'In Queue'})
            process_queue.extend(mysql_response[0])
        if len(process_queue) > 0 and len(under_process) < process_batch:
            thread_pool.add_task(process_report, process_queue.pop(0))
            thread_pool.wait_completion()
    except Exception as e:
        print({"error": "Could not get queue, will try again.", "raw": str(e)})


schedule.every(1).seconds.do(get_queue)

while True:
    schedule.run_pending()
    time.sleep(1)
