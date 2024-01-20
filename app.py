from flask import Flask
from flask_restful import Api

from resources.main_sim import textApp

app = Flask(__name__)
api = Api(app)

api.add_resource(textApp, "/textApp")

if __name__ == "__main__":
  app.run(debug=True)