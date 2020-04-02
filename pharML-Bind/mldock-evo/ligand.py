#!/usr/bin/python
############################################################


import numpy as np
from rdkit import RDLogger
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw


############################################################


SMILES_SYMS = [
    "H", "C", "c", "N", "O", "S", "P", "F", "I", "B",
    "Cl", "Br",
    ".", "=", "-", "#", ":", "$", "%", "/", "\\",
    "(C)", "[C]", "@", "+",
    "0", "1", "2", "3", "4", #"5", "6", "7", "8", "9",
    "C", "C", "CC", "CCC", "CCCC", "H", "cc", "ccc",
    "c", "c1ccccc1"
]


############################################################


def mutate_smiles(smiles,mu):
    # !!av: Stub.
    smiles = list(smiles)
    mus = np.random.rand(len(smiles))
    for ndx,m in enumerate(mus):
        if m < mu:
            m = m / mu
            if m < 0.70:
                if smiles[ndx] == "[" or smiles[ndx] == "]" or smiles[ndx] == "(" or smiles[ndx] == ")":
                    continue
                if smiles[ndx] in list(range(10)):
                    continue
                smiles[ndx] = SMILES_SYMS[np.random.randint(0,len(SMILES_SYMS))]
            elif m < 0.85:
                if len(smiles) < 2:
                    continue
                if smiles[ndx] == "(":
                    c = 0
                    for i in range(ndx,len(smiles)):
                        if smiles[i] == "(":
                            c = c + 1
                        if smiles[i] == ")":
                            c = c - 1
                        if c == 0:
                            smiles[i] = None
                            break
                elif smiles[ndx] == "[":
                    c = 0
                    for i in range(ndx,len(smiles)):
                        if smiles[i] == "[":
                            c = c + 1
                        if smiles[i] == "]":
                            c = c - 1
                        if c == 0:
                            smiles[i] = None
                            break
                elif smiles[ndx] == ")":
                    c = 0
                    for i in range(ndx,len(smiles),-1):
                        if smiles[i] == "(":
                            c = c + 1
                        if smiles[i] == ")":
                            c = c - 1
                        if c == 0:
                            smiles[i] = None
                            break
                elif smiles[ndx] == "]":
                    c = 0
                    for i in range(ndx,len(smiles),-1):
                        if smiles[i] == "[":
                            c = c + 1
                        if smiles[i] == "]":
                            c = c - 1
                        if c == 0:
                            smiles[i] = None
                            break
                elif smiles[ndx] in list(range(10)):
                    ov = smiles[ndx]
                    smiles[ndx] = None
                    count = smiles.count("ov")
                    if count == 0:
                        continue
                    si = np.random.randint(0,count)+1
                    count = 0
                    for i in range(len(smiles)):
                        if smiles[i] == ov:
                            count = count + 1
                        if count == si:
                            smiles[i] = None
                            break
                else:
                    smiles[ndx] = None
            else:
                smiles.insert(ndx,SMILES_SYMS[np.random.randint(0,len(SMILES_SYMS))])
                if smiles[ndx] in list(range(10)):
                    si = np.random.randint(0,len(smiles))
                    smiles.insert(si,smiles[ndx])
    smiles = [ sym for sym in smiles if sym != None ]
    smiles = ''.join(smiles)
    return smiles


############################################################


class Ligand():
    def __init__(self,smiles):
        # Save the smiles string.
        self.smiles = smiles
        return

    def quiet(self):
        # Silence everything but critical errors.
        self.rdk_lg = RDLogger.logger()
        self.rdk_lg.setLevel(RDLogger.CRITICAL)

    def clone(self):
        # Return a copy of this ligand.
        return Ligand(self.smiles)

    def mutate(self,mu):
        # Mutate the ligand.
        self.smiles = mutate_smiles(self.smiles,mu)
        return

    def mate(self,ind):
        # Shorter names.
        smiles_a = self.smiles
        smiles_b = ind.smiles
        # Pick crossover point and recombine.
        xp = np.random.randint(0,min(len(smiles_a),len(smiles_b)))
        cuts_a = ( smiles_a[:xp], smiles_a[xp:] )
        cuts_b = ( smiles_b[:xp], smiles_b[xp:] )
        smiles_a = cuts_a[0] + cuts_b[1]
        smiles_b = cuts_b[0] + cuts_a[1]
        # Create children and return them.
        child_a = Ligand(smiles_a)
        child_b = Ligand(smiles_b)
        return ( child_a, child_b )

    def get_dict(self):
        # Convert the SMILES string into a mol object.
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            return None
        atoms = mol.GetAtoms();
        # Convert the mol object to a data dict.
        v,e,s,r,g = ([],[],[],[],[0.0])
        # Atoms / nodes.
        for atom in atoms:
            v.append( [atom.GetAtomicNum(),atom.GetFormalCharge()] )
        # Sanity check nodes.
        if len(v) == 0:
            return None
        # Bonds / edges.
        for ndx_a in range(len(atoms)):
            for ndx_b in range(len(atoms)):
                bond = mol.GetBondBetweenAtoms(ndx_a, ndx_b)
                if bond != None:
                    if bond.GetIsAromatic():
                        btype = 4
                    else:
                        btype = str(bond.GetBondType())
                        if btype == "SINGLE":
                            btype = 1
                        elif btype == "DOUBLE":
                            btype = 2
                        elif btype == "TRIPLE":
                            btype = 3
                        elif btype == "AROMATIC":
                            btype = 4
                        else:
                            btype = 0
                    e.append([btype])
                    s.append( ndx_a )
                    r.append( ndx_b )
        # Sanity check edges.
        if len(e) == 0:
            return None
        # Globals (stub).
        g = [0.0]
        # Build the actual dict of numpy arrays.
        v = np.array(v,dtype=np.float32)
        e = np.array(e,dtype=np.float32)
        s = np.array(s,dtype=np.float32)
        r = np.array(r,dtype=np.float32)
        g = np.array(g,dtype=np.float32)
        ldict = {}
        ldict['nodes'], ldict['n_node'] = (v, v.shape[0])
        ldict['edges'], ldict['n_edge'] = (e, e.shape[0])
        ldict['senders'], ldict['receivers'] = (s, r)
        ldict['globals'] = g
        # Return ligand graph as data dict.
        return ldict

    def write_image(self,path):
        # Convert to mol object, write, return success status.
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            return False
        Draw.MolToFile(mol,path)
        return True


############################################################
