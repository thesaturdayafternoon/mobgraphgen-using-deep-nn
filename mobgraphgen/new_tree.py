__author__ = 'Behnaz Bostanipour'

class Node:
  def __init__(self, label,parent=None):
    self.label=label
    self.parent=parent
    self.children=[]
    
  def add_child(self,child):
    self.children.append(child)
  
    
class Tree:
  def __init__(self, root_label):
    self.root=Node(root_label)
    self.nodes={}
    self.nodes[root_label]=self.root

  def add_node(self,label,parent_label):
    parent_node=self.nodes[parent_label]
    self.nodes[label]=Node(label,parent_node)
    parent_node.add_child(self.nodes[label])
  
  def iter_ancestor_labels(self,label):
    node=self.nodes[label]
    ancestors=[]
    while node.label != "root":
      node=node.parent
      yield node.label

  def get_ancestor_labels(self,label):
    l=list(self.iter_ancestor_labels(label))
    if "root" in l:
      l.remove("root")
    return l

  def get_sibling_labels(self,label):
    node=self.nodes[label]
    if node.parent is None or label=="root":
        return []
    else:
        return [sibling.label for sibling in node.parent.children if sibling is not node]

  def get_descendants(self,node,descendants):
    for child in node.children:
      descendants.append(child)
      self.get_descendants(child,descendants)

  def get_descendant_labels(self,label):
    descendants=[]
    self.get_descendants(self.nodes[label],descendants)
    return [node.label for node in descendants]