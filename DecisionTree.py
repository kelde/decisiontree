#! /usr/bin/python
import csv
import math
from itertools import *
import operator
import numpy

class DecisionTree:
	
	def __init__(self, filePath, outcomeField):
		self.filePath = filePath
		self.excludedFields = []
		self.setConditions = []
		self.setOutcomeField(outcomeField)

	def setFieldNames(self, fieldNames):
		self.fieldNames = fieldNames

	def loadFile(self):
		with open(self.filePath, 'rb') as csvfile:
			records = csv.DictReader(csvfile, delimiter=',', fieldnames=self.fieldNames)
			self.records = [x for x in records]
			self.currentSet = self.records
	
	def binField(self, column):
		vals = [x[column] for x in self.records]
		minim = math.floor(float(min(vals)))
		maxim = math.ceil(float(max(vals)))
		diff = int(maxim) - int(minim)
		stepsize = diff / 5
		bins = range(minim, maxim + (1 * stepsize), stepsize)
		vals = numpy.array([float(x[column]) for x in self.records])
		inds = numpy.digitize(vals,bins)
		for n in range(vals.size):
			self.records[n][column] = str(bins[inds[n]-1]) + "-" + str(bins[inds[n]])

	def printSet(self):
		print str(self.records)

	def setOutcomeField(self, outcomeField):
		self.outcomeField = outcomeField
		self.excludedFields.append(outcomeField)

	def setExcludedFields(self, excludedFields):
		self.excludedFields = excludedFields

	def setSetConditions(self, setConditions):
		self.setConditions = setConditions

	def setCurrentSet(self):
		self.currentSet = self.records
		for condition in self.setConditions:
			self.currentSet = [x for x in self.currentSet if x[condition['name']] == condition['value']]

	def calculateSetEntropy(self):
		groups = []
		pos = 0
		neg = 0
		records = sorted(self.currentSet, key = lambda k: k[self.outcomeField])
		for k, g in groupby(records, key = lambda k: k[self.outcomeField]):
			groups.append(list(g))

		for group in groups:
			value = group[0][self.outcomeField]
			if value == 'T':
				pos = float(len(group))
			else:
				neg = float(len(group))

		if pos == 0 or neg == 0:
			print "May have encountered a leaf, percentage of positive and/or negative outcomes is zero"
		
		total = len(self.currentSet)

		self.pos = pos
		self.neg = neg

		pos = pos / total
		neg = neg / total

		entropy = ((-pos) * (math.log(pos) / math.log(2))) - (neg * (math.log(neg) / math.log(2))) 

		self.entropy = entropy

	def printExcludedFields(self):
		print "Fields currently excluded as potential nodes: " + str(self.excludedFields)

	def printSetConditions(self):
		print "Conditions placed on current set: "
		print str(self.setConditions)

	def printEntropy(self):
		print "Positive outcomes in given set: " + str(self.pos)
		print "Negative outcomes in given set: " + str(self.neg)
		print "Entropy of given set is: " + str(self.entropy)

	def getHighestGain(self):

		gains = []
		fields = [x for x in self.fieldNames if x not in self.excludedFields]
		
		for field in fields:
			gain = self.calculateGain(field)
			gains.append( { 'name' : field, 'gain' : gain } )
		maxGain = sorted(gains, key = lambda k: k['gain'], reverse=True)[0]
		
		self.maxGain = maxGain

	def calculateGain(self, field):
		total = float(len(self.currentSet))
		valueVars = []
		records = sorted(self.currentSet, key = lambda k: k[field])
		#print "Total records in set: " + str(total)
		print "Attribute is: " + str(field)
		for valueKey, valueGroup in groupby(records, key = lambda k: k[field]):
			pos = 0
			neg = 0
			valueGroup = sorted(list(valueGroup), key = lambda k: k[self.outcomeField])
			#print "Total number with value " + valueKey + ": " + str(len(valueGroup))
			for outcomeKey, outcomeGroup in groupby(valueGroup, key = lambda k: k[self.outcomeField]):
				outcomeGroup = list(outcomeGroup)
				value = outcomeGroup[0][self.outcomeField]
				if value == 'T':
					pos = float(len(outcomeGroup)) / len(valueGroup)
					#print "Total number with value " + valueKey + " and outcome T: " + str(len(outcomeGroup))
				if value == 'F':
					neg = float(len(outcomeGroup)) / len(valueGroup)
					#print "Total number with value " + valueKey + " and outcome F: " + str(len(outcomeGroup))
			if pos == 0:
				posComponent = 0
			else:
				posComponent =  (-pos * (math.log(pos) / math.log(2)))
			if neg == 0:
				negComponent = 0
			else:
				negComponent = neg * (math.log(neg) / math.log(2))
			entropySv = posComponent - negComponent
			#print "Entropy of the set with value: " + valueKey + ": " + str(entropySv)
			valueAttributes = {'sv' : len(valueGroup), 's' : total, 'entropy' : entropySv}
			valueVars.append(valueAttributes)
		gainSa = self.entropy
		for val in valueVars:
			gainSa = gainSa + (-(val['sv'] / val['s']) * val['entropy'])
		print "Gain for this attribute: " + str(gainSa)
		return gainSa

	def printMaxGain(self):
		print "Attribute with maximum gain is: " + str(self.maxGain)

	def setNodeBranches(self, attributeName):
		self.branches = list(set([x[attributeName] for x in self.records]))

	def printBranches(self):
		print str(self.branches)

	def processBranch(self, branch):
		print "Calculating nodes for branch: " + branch
		self.printExcludedFields()
		self.printSetConditions()
		self.setCurrentSet()
		if len(self.currentSet) == 0:
			print "This branch leads to a default category leaf node"
			self.pos, self.neg, self.maxGain, self.entropy = (None,)*4
			return
		elif len(list(set([x[self.outcomeField] for x in self.currentSet]))) == 1:
			classification = [x[self.outcomeField] for x in self.currentSet][0]
			print "This branch leads to a classification node: " + classification
			self.pos, self.neg, self.maxGain, self.entropy = (None,)*4
			return
		else:
			self.calculateSetEntropy()
			self.getHighestGain()