from collections import OrderedDict
import os
import json
#import string
import numpy
import gc
from new_tree import Tree

__author__ = 'Behnaz Bostanipour'


def build_sem_tag_tree():
  tag_tree_file = open("fs_tag_tree.txt", 'r')
  tree=Tree("root")
  for line in tag_tree_file:
    parent, child = str.split(line.rstrip('\r\n'), ',')
    parent = parent.replace(" ", "_")
    child = child.replace(" ","_")
    tree.add_node(child,parent)
  tag_tree_file.close()
  return tree

def get_semantic_tags(input_dir):
    traces_file = open(input_dir + "/traces.txt", "r")
    sem_tags = set()
    for line in traces_file:
        if line == "":
            break
        event = str.split(line.rstrip('\r\n'), ' ')
        sem_tags.add(event[3])
    traces_file.close()
    return sem_tags

class MapInfo:
  
  def __init__(self,city_name):
    self.input_dir="input_" + city_name
    self.sem_tag_tree = build_sem_tag_tree()
    self.load_map_info()
    self.load_region_to_venues_from_traces()
    self.load_all_region_to_venues()

  def load_map_info(self):
    self.regions = OrderedDict()
    self.region_to_semantics = dict()
    self.region_to_venues = dict()
    regions_file = open(self.input_dir + "/regions.txt", "r")
    self.region_size = str.split(regions_file.readline(), " ")
    self.region_size = { 'width': int(self.region_size[0]), 'height': int(self.region_size[1]) }

    for line in regions_file:
      line = line.rstrip('\r\n')
      self.region_values = str.split(line, " ")[:-1]
      region_id = int(self.region_values[0])
      self.regions[region_id] = {'lat': float(self.region_values[1]), 'lon': float(self.region_values[2])}
      self.region_to_semantics[region_id] = []
      self.region_to_venues[region_id] = []

      for val in range(3, len(self.region_values), 1):
        self.region_to_semantics[region_id].append(self.region_values[val])
        self.region_to_venues[region_id].append(self.region_values[val])

      self.region_to_semantics[region_id].append("_")
      self.region_to_semantics[region_id] = set(self.region_to_semantics[region_id])
      self.region_to_venues[region_id].append("_")

    regions_file.close()
    self.semantics = []
    semantics_file = open(self.input_dir + "/fs_tags.txt", "r")

    for line in semantics_file:
      line = line.rstrip('\r\n')
      if line != "":
        self.semantics.append(line)

    semantics_file.close()
    self.this_region_sem_tags_set=get_semantic_tags(self.input_dir)

  def load_region_to_venues_from_traces(self):
    
    traces_file = open(self.input_dir + "/traces.txt", "r")
    self.region_to_venues_from_traces = dict()
    for line in traces_file:
      if line == "":
        break
      event = str.split(line.rstrip('\r\n'), ' ')
      region_id = int(event[2])
      if region_id not in self.region_to_venues_from_traces.keys():
        self.region_to_venues_from_traces[region_id] = []
      self.region_to_venues_from_traces[region_id].append(event[3])
    traces_file.close()

  def load_all_region_to_venues(self):
    self.all_region_to_venues=dict()
    for region_id in self.region_to_venues_from_traces.keys():
      self.all_region_to_venues[region_id]=self.region_to_venues[region_id]+self.region_to_venues_from_traces[region_id]