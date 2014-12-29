#! /usr/bin/python
import csv
import math
from itertools import *
import operator
import numpy

class DecisionTree:
	
	def __init__(self, filePath):
		self.nodes = []
		self.fieldNames = ['Diagnosis', 'FVC', 'FEV1', 'PerformanceStatus', 'Pain', 'Haemoptyisis', 'Dyspnoea', 'Cough', 'Weakness', 'SizeOfTumor', 'Diabetes', 'MI', 'PAD', 'Smokes', 'Asthma', 'Age', 'Risk1Y']
		self.preProcessData(filePath)
		self.outcomeColumn = 'Risk1Y'
		self.entropy = self.setEntropy(self.records)
		self.excludedFields = [self.outcomeColumn]
		self.currentAttribute = None
		self.setRootNode()
		self.processBranches(self.rootNode)
		print str(self.nodes)

	def preProcessData(self, filePath):
		with open('thoracic_surgery.csv', 'rb') as csvfile:
			records = csv.DictReader(csvfile, delimiter=',', fieldnames=self.fieldNames)
			self.records = [x for x in records]
			for field in self.fieldNames:
				if field == 'FVC' or field == 'FEV1' or field == 'Age':
					self.binField(field)

	def binField(self, column):
		minim = math.floor(self.getMin(column))
		maxim = math.ceil(self.getMax(column))
		bins = self.createBins(int(minim), int(maxim))
		vals = numpy.array([float(x[column]) for x in self.records])
		inds = numpy.digitize(vals,bins)
		for n in range(vals.size):
			self.records[n][column] = str(bins[inds[n]-1]) + "-" + str(bins[inds[n]])

	def getMin(self, column):
		vals = [x[column] for x in self.records]
		minim = float(min(vals))
		return minim

	def getMax(self, column):
		vals = [x[column] for x in self.records]
		maxim = float(max(vals))
		return maxim

	def createBins(self, start, stop):
		diff = int(stop) - int(start)
		stepsize = diff / 5
		bins = range(start, stop + (1 * stepsize), stepsize)
		return bins

	def setEntropy(self, records):
		groups = []
		pos = 0
		neg = 0
		
		records = sorted(records, key = lambda k: k[self.outcomeColumn])
		
		for k, g in groupby(records, key = lambda k: k[self.outcomeColumn]):
			groups.append(list(g))
		
		for group in groups:
			value = group[0][self.outcomeColumn]
			if value == 'T':
				pos = float(len(group))
			else:
				neg = float(len(group))
		
		if pos == 0 or neg == 0:
			print "May have encountered a leaf, percentage of positive and/or negative outcomes is zero"
		
		total = len(records)
		pos = pos / total
		neg = neg / total
		entropy = ((-pos) * (math.log(pos) / math.log(2))) - (neg * (math.log(neg) / math.log(2))) 
		
		return entropy

	def calculateGain(self, records, field, entropy):
		total = float(len(records))
		valueVars = []
		records = sorted(records, key = lambda k: k[field])
		for valueKey, valueGroup in groupby(records, key = lambda k: k[field]):

			pos = 0
			neg = 0

			valueGroup = sorted(list(valueGroup), key = lambda k: k[self.outcomeColumn])
			
			for outcomeKey, outcomeGroup in groupby(valueGroup, key = lambda k: k[self.outcomeColumn]):
				outcomeGroup = list(outcomeGroup)
				value = outcomeGroup[0][self.outcomeColumn]
				if value == 'T':
					pos = float(len(outcomeGroup)) / len(valueGroup)
				if value == 'F':
					neg = float(len(outcomeGroup)) / len(valueGroup)
			if pos == 0:
				posComponent = 0
			else:
				posComponent =  (-pos * (math.log(pos) / math.log(2)))
			if neg == 0:
				negComponent = 0
			else:
				negComponent = neg * (math.log(neg) / math.log(2))
			entropySv = posComponent - negComponent
			valueAttributes = {'key' : valueKey, 'sv' : len(valueGroup), 's' : total, 'pos' : pos, 'neg' : neg, 'entropy' : entropySv}
			valueVars.append(valueAttributes)
		gainSa = entropy
		for val in valueVars:
	 		valueComponent = -(val['sv'] / val['s']) * val['entropy']
			gainSa = gainSa + valueComponent
		return gainSa

	def getHighestGain(self, records):
		#print "Placeholder"
		gains = []
		fields = [x for x in self.fieldNames if x not in self.excludedFields]
		for field in fields:
			gain = self.calculateGain(records, field, self.entropy)
			gains.append({ 'name' : field, 'gain' : gain})
		maxGain = sorted(gains, key = lambda k: k['gain'], reverse=True)[0]
		return maxGain

	def setRootNode(self):
		#print "Placeholder"
		maxGain = self.getHighestGain(self.records)
		node = maxGain['name']
		self.rootNode = node
		self.nodes.append({ 'name' : node, 'branch' : 'Root' })
		self.setRootNodeBranches()



	def setDefaultCategory(self):
		print "Placeholder"

	def setLeaf(self):
		print "Placeholder"

	def setNode(self, nodename, branchName, childNode):
		for i in range(len(self.nodes)):
			if self.nodes[i]['name'] == nodename:
				print "Adding childnode here: " + childNode
				try:
					self.nodes[i]['childnodes'].append({'name': childNode, 'branch': branchName})
					self.setNodeBranches(nodename, childNode)
				except KeyError:
					self.nodes[i]['childnodes'] = []
					self.nodes[i]['childnodes'].append({'name':childNode, 'branch':branchName})
					self.setNodeBranches(nodename, childNode)




		
	def processBranches(self, nodename):
		print "Placeholder"

		for node in self.nodes:
			if node['name'] == nodename:
				branches = node['branches']
				for branch in branches:
					# calculate the sv set
					sv = self.calculateSv(branch['name'], nodename)
					numExamples = len(sv);
					print "Branch " + branch['name'] + " has: " + str(len(sv)) + "examples"
					# if sv is empty
					if len(sv) == 0:
						# set the default category here
						self.setDefaultCategory(branch['name'], nodename)
						continue
					#if sv only contains examples from one classification
					elif len(list(set([x[self.outcomeColumn] for x in sv]))) == 1:
						classification = [x[self.outcomeColumn] for x in sv][0]
						self.setBranchLeaf(branch['name'], nodename, classification)
						continue
					else:
						# remove the attribute from the set 
						self.excludedFields.append(nodename)
						maxGain = self.getHighestGain(sv)
						print str(maxGain)
						childNode = maxGain['name']
						branchName = branch['name']
						print "Adding childNode: " + childNode + ", Nodename: " + nodename
						self.setNode(nodename, branchName, childNode)
						# self.calculateSv() with the new node


	def calculateSv(self, branchname, nodename):
		print str(nodename)
		sv = [x for x in self.records if x[nodename] == branchname]
		return sv
		# get the set of examples from self.records that have value v for attribute a
		
	def setDefaultCategory(self, nodename, branchname):
		for i in range(len(self.nodes)):
			if self.nodes[i]['name'] == nodename:
				branches = self.nodes[i]['branches']
				for j in range(len(branches)):
					if self.nodes[i]['branches'][j]['name'] == branchname:
						self.nodes[i]['branches'][j]['leaf'] = 'Default Category'

			





	def setRootNodeBranches(self):
		root = self.nodes[0]
		root['branches'] = []
		branches = list(set([x[self.rootNode] for x in self.records]))
		for branch in branches:
			root['branches'].append({'name': branch})

	def setNodeBranches(self, nodename, childNode):
		for i in range(len(self.nodes)):
			if self.nodes[i]['name'] == nodename:
				childNodes = self.nodes[i]['childnodes']
				for j in range(len(childNodes)):
					if self.nodes[i]['childnodes'][j]['name'] == childNode:
						self.nodes[i]['childnodes'][j]['branches'] = []
						branches = list(set([x[childNode] for x in self.records]))
						self.nodes[i]['childnodes'][j]['branches'].append(branches)





	def setBranchLeaf(self, branchname, nodename, classification):
		for i in range(len(self.nodes)):
			if self.nodes[i]['name'] == nodename:
				branches = self.nodes[i]['branches']
				for j in range(len(branches)):
					if self.nodes[i]['branches'][j]['name'] == branchname:
						self.nodes[i]['branches'][j]['leaf'] = classification

