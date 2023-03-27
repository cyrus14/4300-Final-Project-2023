from flask import Flask, render_template, request
from return_songs import find_nonzero_indices
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('base.html')

@app.route('/my-link')
def my_link():
  current_url = request.url
  city = current_url[current_url.index('key=') + 4:]
  city = city.replace('_', ' ')
  return find_nonzero_indices(city)

if __name__ == '__main__':
  app.run(debug=True)