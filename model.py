
"""
The main code which runs the entire review analysis
"""

import os
import sys
import numpy as np
from . import HAC
import math
from . import FileCreationWithBigrams
from . import AdjScore
import operator
import collections
from textblob import TextBlob
from texttable import Texttable
from . import confusion_matrix
#Get the review filename as command line argument
# filename = sys.argv[1]

#reviewTitle is the list containing title of all reviews
model_vars = {
	"filename": "",
	"reviewTitle" : [],

	#model_vars["reviewContent"] is the list containing all reviews
	"reviewContent" : [],

	"posCount" : 0,
	"negCount" : 0,
	"neutCount" : 0,
	"curIndex" : -1,

	#Arrays holding the index of the reviews
	"posActIndex" : [],
	"negActIndex" : [],
	"neutActIndex" : []
}

#Extract review title and content from the file
def extract_reviews(path, filename):
	global model_vars
	result = {"reviews": [], "scores": []}
	filepath = os.path.join(path, filename)
	with open(filepath) as f:
		review = []
		for line in f:
			if line[:6] == "[+][t]":							#Incase the line starts with [t], then its the title of review
				if review:
					model_vars["reviewContent"].append(review)
					review = []
				model_vars["reviewTitle"].append(line.split("[+][t]")[1].rstrip("\r\n"))
				model_vars["reviewTitle"].append(line.split("[+][t]")[1].rstrip("\r\n"))
				model_vars["posCount"] += 1
				model_vars["curIndex"] += 1
				model_vars["posActIndex"].append(model_vars["curIndex"])
			elif line[:6] == "[-][t]":
				if review:
					model_vars["reviewContent"].append(review)
					review = []
				model_vars["reviewTitle"].append(line.split("[-][t]")[1].rstrip("\r\n"))
				model_vars["reviewTitle"].append(line.split("[-][t]")[1].rstrip("\r\n"))
				model_vars["negCount"] += 1
				model_vars["curIndex"] += 1
				model_vars["negActIndex"].append(model_vars["curIndex"])
			elif line[:6] == "[N][t]":
				if review:
					model_vars["reviewContent"].append(review)
					review = []
				model_vars["reviewTitle"].append(line.split("[N][t]")[1].rstrip("\r\n"))
				model_vars["reviewTitle"].append(line.split("[N][t]")[1].rstrip("\r\n"))
				model_vars["neutCount"] += 1
				model_vars["curIndex"] += 1
				model_vars["neutActIndex"].append(model_vars["curIndex"])
			else:
				if "##" in line:								#Each line in review starts with '##'
					x = line.split("##")
					for i in range(1, len(x)):			#x[0] is the feature given the file.Its been ignored here as its not a part of the review
						review.append(x[i].rstrip("\r\n"))
				else:
					continue
		model_vars["reviewContent"].append(review)

	#Creating a File attaching Bigrams
	FileCreationWithBigrams.fileCreation(model_vars["reviewContent"],filepath)
	from . import MOS
	from . import WithNgrams



	#The HAC algorithm to extract features and adjectives in the review
	adjDict = HAC.findFeatures(model_vars["reviewContent"],filepath)
	featureList = WithNgrams.getList()

	#Get adjective scores for each adjective
	adjScores = AdjScore.getScore(adjDict,filepath)

	#MOS algorithm to get feature score and review score
	posPredIndex, negPredIndex, neutPredIndex, avgFeatScore = MOS.rankFeatures(adjScores, featureList,
		model_vars["reviewTitle"], model_vars["reviewContent"])

	# outputDir = "Results_" + filename.split("/")[1]
	# if not os.path.exists(outputDir):
	#     os.makedirs(outputDir)

	#Write the predicted positive reviews to a file
	# with open(outputDir + "/positiveReviews.txt", "w") as filePos:
	for i in posPredIndex:
		r = ""
		for k in range(len(model_vars["reviewContent"][i])):
			# filePos.write(model_vars["reviewContent"][i][k])
			r += model_vars["reviewContent"][i][k]
		# filePos.write("\n")
		result["reviews"].append((i, r, 1))

	#Write the predicted negative reviews to a file
	# with open(outputDir + "/negativeReviews.txt", "w") as fileNeg:
	for i in negPredIndex:
		r = ""
		for k in range(len(model_vars["reviewContent"][i])):
			# fileNeg.write(model_vars["reviewContent"][i][k])
			r += model_vars["reviewContent"][i][k]
		# fileNeg.write("\n")
		result["reviews"].append((i, r, -1))

	#Write the predicted neutral reviews to a file
	# with open(outputDir + "/neutralReviews.txt", "w") as fileNeut:
	for i in neutPredIndex:
		r = ""
		for k in range(len(model_vars["reviewContent"][i])):
			# fileNeut.write(model_vars["reviewContent"][i][k])
			r += model_vars["reviewContent"][i][k]
		# fileNeut.write("\n")
		result["reviews"].append((i, r, 0))

	#Write the predicted neutral reviews to a file
	# with open(outputDir + "/featureScore.txt", "w") as fileFeat:
	# t = Texttable()
	# lst = [["Feature", "Score"]]
	for tup in avgFeatScore:
		# lst.append([tup[0], tup[1]])
		result["scores"].append((tup[0], round(tup[1], 2)))
	# t.add_rows(lst)
	# fileFeat.write(str(t.draw()))

	# print("The files are successfully created in the dir '" + outputDir + "'")

	#Evaluation metric
	PP = len(set(model_vars["posActIndex"]).intersection(set(posPredIndex)))
	PNe = len(set(model_vars["posActIndex"]).intersection(set(negPredIndex)))
	PN = len(set(model_vars["posActIndex"]).intersection(set(neutPredIndex)))

	NeP = len(set(model_vars["negActIndex"]).intersection(set(posPredIndex)))
	NeNe = len(set(model_vars["negActIndex"]).intersection(set(negPredIndex)))
	NeN = len(set(model_vars["negActIndex"]).intersection(set(neutPredIndex)))

	NP = len(set(model_vars["neutActIndex"]).intersection(set(posPredIndex)))
	NNe = len(set(model_vars["neutActIndex"]).intersection(set(negPredIndex)))
	NN = len(set(model_vars["neutActIndex"]).intersection(set(neutPredIndex)))

	#Draw the confusion matrix table
	# t = Texttable()
	# t.add_rows([['', 'Pred +', 'Pred -', 'Pred N'], ['Act +', PP, PNe, PN], ['Act -', NeP, NeNe, NeN], ['Act N', NP, NNe, NN]])
	cm = np.array([[PP, PNe, PN], [NeP, NeNe, NeN], [NP, NNe, NN]])
	confusion_matrix.plot_confusion_matrix(cm, ["Positive", "Negative", "Neutral"], filename)
	# print("Evaluation metric - Confusion matrix:")
	# print("=====================================")
	# print("Dataset source:", filename)
	# print(t.draw())
	# print(result)
	model_vars = {
		"filename": "",
		"reviewTitle" : [],

		#model_vars["reviewContent"] is the list containing all reviews
		"reviewContent" : [],

		"posCount" : 0,
		"negCount" : 0,
		"neutCount" : 0,
		"curIndex" : -1,

		#Arrays holding the index of the reviews
		"posActIndex" : [],
		"negActIndex" : [],
		"neutActIndex" : []
	}
	print(result["scores"])
	return result
