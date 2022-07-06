#encoding=utf-8
"""
Providing the operation on database
"""
import sqlite3
import os, sys, logging

DBNAME = ""
l = logging.getLogger("db_operation.py")
l.setLevel(logging.WARNING)
l.addHandler(logging.StreamHandler())
l.addHandler(logging.FileHandler("DbOp.log"))
l.handlers[0].setFormatter(logging.Formatter("%(filename)s : %(message)s"))
l.handlers[0].setLevel(logging.INFO)
l.handlers[1].setLevel(logging.ERROR)
root = os.path.dirname(os.path.abspath(__file__))
class DBOP():
    def __init__(self, db_path=None):
        self._db_path = db_path
        if not self._db_path or self._db_path=="":
            self.init_db_name()
        self.open_db()
    
    def __del__(self):
        self.db.commit()
        self.db.close()

    def init_db_name(self):
        '''
        To initialize the path of database
        '''
        dir = os.path.dirname(os.path.abspath(__file__))
        self._db_path = os.path.join(dir,"all.sqlite")

    def open_db(self):
        db = sqlite3.connect(self._db_path)
        db.text_factory = str
        db.row_factory = sqlite3.Row
        self.db = db
        self.create_schema()
        db.execute("analyze") #To boost the query

    def create_schema(self):
        '''
        :return:
        '''
        print("Create schema.")
        cur = self.db.cursor()
        sql = """create table if not exists function (
                        func_name varchar(255),
                        bin_path varchar(255),
                        bin_name varchar(255),
                        func_pick_dumps text,
                        func_embedding text,
                        is_cve integer not null default 0, 
                        cve_id varchar(20),
                        cve_description text,
                        cve_references text,
                        constraint pk primary key (func_name, bin_path)
                    )"""
        cur.execute(sql)
        #create index
        cur.close()
        l.info("created table function")
    
    def insert_function(self, func_name, bin_path, bin_name, func_pick_dump):
        '''
        :param func_name: str
        :param bin_path: str
        :param func_pick_dump: str
        :return:
        '''
        sql = """insert into function(func_name, bin_path, bin_name, func_pick_dumps, is_cve)
                    values (?,?,?,?,?)"""
        values =(func_name, bin_path, bin_name, func_pick_dump, 0) 
        # print(values)
        try:
            cur = self.db.cursor()
            cur.execute(sql, values)
            cur.close()
            l.info("bin_path:function_name %s:%s" % (bin_path, func_name))
        except Exception as e:
            # l.error("[E]insert error. args : %s\n %s" % (str(values), Exception.message))
            l.error("Function %s's info insertion to database fails" % (func_name))
            # print(func_name, bin_path)
            l.error(e)
    
    def insert_vul_function(self, func_name, bin_path, bin_name, func_pick_dump, cve_id=None, cve_desc=None, cve_references=None):
        '''
        :param func_name: str
        :param bin_path: str
        :param func_pick_dumps: str
        :param cve_id: str
        :param cve_desc: str
        :param cve_references: str
        :return:
        '''
        sql = """insert into function(func_name,  bin_path, bin_name, func_pick_dumps, is_cve, cve_id, cve_description, cve_references)
                    values (?,?,?,?,?,?,?,?)"""
        values =(func_name, bin_path, bin_name, func_pick_dump, 1, cve_id, cve_desc, cve_references) 
        try:
            cur = self.db.cursor()
            cur.execute(sql, values)
            cur.close()
            l.info("bin_path:vul_function_name %s:%s" % (bin_path, func_name))
        except :
            # l.error("[E]insert error. args : %s\n %s" % (str(values), Exception.message))
            l.error("Vul Function %s's info insertion to database fails" % (func_name))

