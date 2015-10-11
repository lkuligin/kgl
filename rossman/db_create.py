#!/usr/bin/env python
# coding: utf8
"""
Create a sqlite db, supports easy migration between different db versions
"""
import sqlite3
import os
import traceback
import csv

from config import SQLITE_DB_NAME
#paths to files with data
PATH_STORES = 'data\\store.csv'
PATH_TRAIN = 'data\\train.csv'
PATH_TEST = 'data\\test.csv'
#mapping  external fields to internal
FIELDS_TRAIN = {'Store': ['Store', 'int'], 'Date': ['DayRef', 'date'], 'Sales': ['Sales', 'int'], 'Customers': ['Customers', 'int'], 'Open': ['Open', 'int'], \
'Promo': ['Promo', 'int'], 'StateHoliday': ['StateHoliday ', 'text'], 'SchoolHoliday': ['SchoolHoliday', 'int']}
FIELDS_TEST = {'Id': ['ID', 'int'], 'Store': ['Store', 'int'], 'Date': ['DayRef', 'date'], 'Open': ['Open', 'int'], \
'Promo': ['Promo', 'int'], 'StateHoliday': ['StateHoliday ', 'text'], 'SchoolHoliday': ['SchoolHoliday', 'int']}
FIELDS_STORES = {'Store': ['Store', 'int'], 'StoreType': ['Type', 'text'], 'Assortment': ['Assortment', 'text'], 'CompetitionDistance': ['CompDist', 'int'], \
'CompetitionOpenSinceMonth': ['CompOpenMonth', 'int'], 'CompetitionOpenSinceYear': ['CompOpenYear', 'int'], 'Promo2': ['Promo2', 'text'], \
'Promo2SinceWeek': ['Promo2Week', 'text'], 'Promo2SinceYear': ['Promo2Year', 'text']}
FIELDS_STORES_MONTHS = {'Store': ['Store', 'int'], 'PromoInterval': ['Month', int]}
#amount of rows to INSERT by one query
BUFF_SIZE = 500
MONTHES = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def process_line_months(headers, row):
	sql_text = ''
	store_id = row[headers.index('Store')]
	months = row[headers.index('PromoInterval')]
	if months <> '':
		months = months.split(',')
		for month in months:
			month = month.strip()
			if month in MONTHES.keys():
				sql_text = sql_text + store_id + ', ' + str(MONTHES[month]) + '), ('
			else:
				print months, month
	return sql_text

def upload_general_data(db_name, table_name, path, field_stores, buff_size, plf = None):
	try:
		con = sqlite3.connect(db_name)
		cur = con.cursor()
		with open(path) as f:
			datareader = csv.reader(f, delimiter = ',')
			headers = datareader.next()
			columns_add = []
			sql_text_header = 'INSERT INTO ' + table_name + ' ('
			i = 0
			for title in headers:
				if title in field_stores.keys():
					sql_text_header += field_stores[title][0] + ', '
					columns_add.append(title)
			sql_text_header = sql_text_header[:-2] + ') VALUES ('
			lines_done = 0
			sql_text = ''
			for row in datareader:
				if len(row) == len(headers):
					lines_done += 1
					if plf == None:
						for title in columns_add:
							indx = headers.index(title)
							if row[indx] <> '':
								if field_stores[title][1] == 'text':
									sql_text += '\'' + row[indx] + '\', '
								elif field_stores[title][1] == 'date':
									sql_text += '\'' + row[indx] + '\', '
								else:
									sql_text += row[indx] + ', '
							else: 
								sql_text += 'NULL, '
						sql_text = sql_text[:-2] +'), ('
					elif plf == 'process_line_months':
						sql_text += process_line_months(headers, row)
					if lines_done % buff_size == 0:
						sql_text = sql_text[:-3]
						sql_text = sql_text_header + sql_text
						cur.execute(sql_text)
						con.commit()
						sql_text = ''
						#print lines_done
			if lines_done % buff_size <> 0:
				sql_text = sql_text[:-3]
				sql_text = sql_text_header + sql_text
				cur.execute(sql_text)
				con.commit()
	except:
		print sql_text
		print lines_done
		print row
		print "error: ", traceback.format_exc()

if __name__ == "__main__":
	if not os.path.exists(SQLITE_DB_NAME):
		con = sqlite3.connect(SQLITE_DB_NAME)
		cur = con.cursor()
		cur.execute('CREATE TABLE Stores (Store INTEGER PRIMARY KEY, Type NVACHAR(3), Assortment NVARCHAR(3), CompDist INTEGER, CompOpenMonth INTEGER, \
		CompOpenYear INTEGER, Promo2 NVACHAR(3), Promo2Week INTEGER, Promo2Year INTEGER)')
		con.commit()
		cur.execute('CREATE TABLE TrainData (Store INTEGER, DayRef date, Sales INTEGER, Customers INTEGER, Open NVARCHAR(3), Promo NVARCHAR(3), \
		StateHoliday NVARCHAR(3), SchoolHoliday nvarchar(3), primary key (Store, DayRef))')
		con.commit()
		cur.execute('CREATE TABLE StoresPromo (Store INTEGER, Month nvarchar(10), primary key (Store, Month))')
		con.commit()
		cur.execute('CREATE TABLE TestData (ID INTEGER, Store INTEGER, DayRef date, Open NVARCHAR(3), Promo NVARCHAR(3), \
		StateHoliday NVARCHAR(3), SchoolHoliday nvarchar(3), primary key (ID))')
		con.commit()
		con.close()
	upload_general_data(SQLITE_DB_NAME, 'Stores', PATH_STORES, FIELDS_STORES, BUFF_SIZE)
	upload_general_data(SQLITE_DB_NAME, 'TrainData', PATH_TRAIN, FIELDS_TRAIN, BUFF_SIZE)
	upload_general_data(SQLITE_DB_NAME, 'TestData', PATH_TEST, FIELDS_TEST, BUFF_SIZE)
	upload_general_data(SQLITE_DB_NAME, 'StoresPromo', PATH_STORES, FIELDS_STORES_MONTHS, BUFF_SIZE, 'process_line_months')

