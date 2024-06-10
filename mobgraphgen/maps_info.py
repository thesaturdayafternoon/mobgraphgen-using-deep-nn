from map_info import MapInfo,build_sem_tag_tree

__author__ = 'Behnaz Bostanipour'

def merge_dicts_with_duplicates(dicts):
  super_dict = {}
  for k in set(k for d in dicts for k in d):
    super_dict[k] = [d[k] for d in dicts if k in d]
    super_dict[k] = list([item for sublist in super_dict[k] for item in sublist])
  return super_dict

class MapsInfo:

  def __init__(self,city_names):
    self.load_all_cities_to_venues(city_names)
    self.sem_tag_tree = build_sem_tag_tree()

  def load_all_cities_to_venues(self,city_names):
    dicts=[]
    for city_name in city_names:
      dicts.append(MapInfo(city_name).region_to_venues_from_traces)
    self.all_cities_to_venues=merge_dicts_with_duplicates(dicts)