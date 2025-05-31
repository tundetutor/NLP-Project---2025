from flask import Flask

app = Flask(__name__)

from text_app_project.app import routes