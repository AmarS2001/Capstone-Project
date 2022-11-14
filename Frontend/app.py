from flask import Flask, render_template
from flask import request
import os
import requests
app = Flask(__name__)

@app.route("/")
def home():
   return render_template("home.html")


@app.route("/dash")
def Dashboard():
   if request.method == 'POST':
      months = request.form.get('months')
      return f'{months} months'

   return render_template("Dashboard.html")

@app.route("/about")
def aboutus():
   return render_template("about.html")

@app.route("/show")
def showdata():
   return render_template("showdata.html")


if __name__ == '__main__':
   app.run()