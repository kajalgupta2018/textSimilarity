from flask_restful import Resource
from flask import Flask, request
from model.model import TextTokenizer,ModelArgs,SimilarityModel
from model.model import pipeline
import os

root = os.path.dirname(os.path.abspath(__file__))
nltk_dir = os.path.join(root, '../my_nltk_dir')  # Your folder name here
os.environ['NLTK_DATA'] = nltk_dir
print("nltk dir:", nltk_dir)
tokenizer = TextTokenizer()
current_path = os.path.dirname(__file__)
print("current path:", current_path) 
model_path = os.path.join(current_path, "../data", "model.sim")
model = pipeline(tokenizer, ModelArgs, train=False, model_name=model_path)


class textApp(Resource):

  def get(self):
    text1 = request.args.get("text1")
    text2 = request.args.get("text2")
    tokens1  = tokenizer.tokenizer(text1)
    tokens2  = tokenizer.tokenizer(text2)
    score = model.similarityIndex(tokens1, tokens2)
    print(score)
    pp={}
    pp["similarity score"] = score 
    return pp, 200

  def put(self, tex1):
    return "put", 200
    