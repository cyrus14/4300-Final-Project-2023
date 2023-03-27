import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from return_songs import find_nonzero_indices

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
SQL_CREDS = json.load(open('./conf.json', 'r+'))

MYSQL_USER = SQL_CREDS['sql_user']
MYSQL_USER_PASSWORD = SQL_CREDS['sql_user_pwd']
MYSQL_PORT = SQL_CREDS['sql_port']
MYSQL_DATABASE = SQL_CREDS['sql_db']

# mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# # Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this


def sql_search(episode):
    query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
    keys = ["id", "title", "descr"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys, i)) for i in data])


@app.route("/<path:path>")
def home(path):
    if "/key=" in path and "results" not in path:
        return render_template('home.html', title="sample html")
    else:
        my_link()


@app.route('/results')
def my_link():
    current_url = request.url
    city = current_url[current_url.index('key=') + 4:]
    cityClean = city.replace('_', ' ')
    content = find_nonzero_indices(cityClean)
    return render_template('results.html', data=content)


@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)


'''
@app.route('/results')
def results():
    return render_template('results.html')
'''


# app.run(debug=True)
