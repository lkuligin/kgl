#!/usr/bin/env python
#coding: utf8

import sqlite3
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from matplotlib.dates import DateFormatter
import sys
from config import SQLITE_DB_NAME

#plot sales for one particular store
def sales_store(store_id):
	con = sqlite3.connect(SQLITE_DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES)
	cur = con.cursor()
	store_id = 262
	cur.execute('SELECT Store, DayRef, Sales FROM TrainData WHERE Sales > 0 AND Store = ?', (store_id,))
	res = cur.fetchall()
	con.close()
	days = []
	values = []
	for row in res:
		days.append(row[1])
		values.append(row[2])
	return days, values
	
if __name__ == "__main__":
	days, values = sales_store(262)
	days = mdates.date2num(days)
	fig, ax = plt.subplots()
	ax.plot_date(days, values, '-')
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b'%y"))
	ax.xaxis.set_major_locator(mdates.AutoDateLocator())
	ax.autoscale_view()
	fig.autofmt_xdate()
	#ays=datetime.strptime(
	# = mdates.date2num(days)
	#ax.plot_date(x=x, y=y)
	fig.savefig('test.png')
	plt.close(fig)
	