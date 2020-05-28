#-*- coding=utf-8 -*-
import pymysql
import pymysql.cursors


class mysql_connection:
    def __init__(self):
        self.initialized = False
        self.conn = None
    def connect(self, host, user, password, db, port=3306, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor ):
        self.conn = pymysql.connect(host=host, user=user, password=password, db=db, port=port, charset=charset, cursorclass=cursorclass)
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.port = port
        self.charset=charset
        self.cursorclass = cursorclass
        self.initialized = True
    def reconnect(self):
        try:
            self.connect(self.host, self.user, self.password, self.db, self.charset, self.cursorclass)
        except Exception as e :
            print("there is an exception for connect: {}".format(e))
    #just for query , not change anything in db
    def query(self, sql):
        if not self.initialized:
            print("failed due to not initialized")
            return None
        try:
            self.conn.ping(reconnect=True)
            cursor = self.conn.cursor()
            cursor.execute(sql)
            return cursor
        except pymysql.OperationalError:
            self.reconnect()
            cursor = self.conn.cursor()
            cursor.execute(sql)
            return cursor
        except Exception as e:
            print("exception occur, sql{}, exception:{}".format(sql, e))
            return None

    def get_raw_conn(self):
        if not self.initialized:
            print("failed due to not initialized")
            return None
        try:
            self.conn.ping(reconnect=True)
        except Exception as e:
            print("exception occur,  exception::{}".format(e))
            return None
        return self.conn

    def execute(self, sql):
        if not self.initialized:
            print("failed due to not initialized")
            return -1
        try:
            self.conn.ping(reconnect=True)
            cursor = self.conn.cursor()
            row_count = cursor.execute(sql)
            self.conn.commit()
            return row_count
        except pymysql.OperationalError:
            self.reconnect()
            cursor = self.conn.cursor()
            row_count = cursor.execute(sql)
            self.conn.commit()
            return row_count
        except Exception as e:
            self.conn.rollback()
            print("exception occur, sql:{}, exception::{}".format(sql, e))
            return -2

    def execute_insert(self, sql):
        if not self.initialized:
            print("failed due to not initialized")
            return -1
        try:
            self.conn.ping(reconnect=True)
            cursor = self.conn.cursor()
            row_count = cursor.execute(sql)
            self.conn.commit()
            # return row_count
            return cursor.lastrowid
        except pymysql.OperationalError:
            self.reconnect()
            cursor = self.conn.cursor()
            row_count = cursor.execute(sql)
            self.conn.commit()
            return cursor.lastrowid
            # return row_count
        except Exception as e:
            self.conn.rollback()
            print("exception occur, sql:{}, exception::{}".format(sql, e))
            return -2

    @staticmethod
    def Binary(content):
        try:
            return pymysql.Binary(content)
        except Exception as e:
            print("exception on reconnect")
            return None



