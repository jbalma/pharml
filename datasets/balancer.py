#!/usr/bin/python

import os 
import random


############################################################


class train_item(object):
    def __init__(self,inputs=None,outputs=None,tags=None):
        self.inputs  = inputs  if inputs  else []
        self.outputs = outputs if outputs else []
        self.tags    = tags    if tags    else []


class category(object):
    def __init__(self,name=None,frac=None,children=None,parent=None,items=None, shuffle=None):
        self.name     = name
        self.frac     = frac
        self.parent   = parent
        self.children = children if children else set()
        self.items    = items if items else []
        self.gitems   = []
        self.count    = len(self.items)
        self.bcount   = 0
        self.shuffle = shuffle 
        if parent:
            parent.children.add(self)


    def write(self,balanced=True,path="./data"):
        # Open output map file for writing
        map_dir="%s/map"%(path)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)

        with open("%s/map/%s.map"%(path,self.name), 'w') as mapfile:
            # Decide which items to write
            if balanced:
                items = self.gitems
            else:
                items = self.items if len(self.items) > len(self.gitems) else self.gitems
            # Write each item's inputs, outputs, and tags
            for item in items:
                mapfile.write(str(len(item.inputs))+" ")
                for i in item.inputs:
                    mapfile.write(str(i)+" ")
                mapfile.write(str(len(item.outputs))+" ")
                for o in item.outputs:
                    mapfile.write(str(o)+" ")
                mapfile.write(str(len(item.tags))+" ")
                for t in item.tags:
                    mapfile.write(str(t)+" ")
                mapfile.write("\n")
            # Close file    
            mapfile.close()


    def balance(self):
        # Balance the data set with self at the root
        min_max_count = min_max_category_size(self)
        balance_node(self,min_max_count)


############################################################


def min_max_category_size(node):
    node.gfrac = node.frac * node.parent.gfrac if node.parent else node.frac
    counts = [ count for count in map(min_max_category_size, node.children) if count ]
    min_max_count = min(counts) if len(counts) else 0.0
    if not len(node.children):
        min_max_count = node.count / node.gfrac if node.gfrac else 0.0
        pt = (node.name, node.gfrac, node.count, min_max_count)
        # print("%20s:  gf:%-10.2f  count:%-10.2f  max_gcount:%-10.2f"%pt)
    return min_max_count


def balance_node(node,count):
    if len(node.children):
        node.bcount = sum(map(balance_node, node.children, [count for c in node.children]))
        for c in node.children:
            node.gitems += c.gitems
    else:
        node.bcount = int(node.gfrac*count)
        node.gitems = node.items[:node.bcount]
    if node.shuffle:
        random.shuffle(node.gitems)
    return node.bcount


############################################################


def main():
    print("main(): Stub!")


if __name__ == "__main__":
    main()


############################################################
