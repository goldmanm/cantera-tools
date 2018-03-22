# -*- coding: utf-8 -*-

def make_string_labels_independent(species):
    """
    This method accepts a list of species objects, converts their label to SMILES,
    and makes sure none of their labels conflict
    If a conflict occurs, the second occurance will have '-2' added, third '-3'...
    """
    from rmgpy.molecule import Molecule
    
    labels = set()
    for spec in species:
        duplicate_index = 1
        if spec.molecule: # use smiles string
            # see if the label is already valid smiles
            try:
                Molecule().fromSMILES(spec.label)
                potential_label = spec.label
            except:
                potential_label = spec.molecule[0].toSMILES()
        else:
            potential_label = spec.label
        unnumbered_label = potential_label
        while potential_label in labels:
            duplicate_index += 1
            potential_label = unnumbered_label + '-{}'.format(duplicate_index)
        spec.label = potential_label
        labels.add(potential_label)

def obtain_cti_file_nicely_named(chemkinfilepath, readComments = True, 
                                 original_ck_file = 'chem_annotated.inp'):
    """
    Given a chemkin file path, this method reads in the chemkin file and species
    dictionary into RMG objects, renames the species more intuitively, and then
    saves the output as 'input_nicely_named.cti' to the same file path.
    """
    from rmgpy.chemkin import loadChemkinFile
    import os
    import soln2cti
    import cantera as ct

    chemkinPath = os.path.join(chemkinfilepath, original_ck_file)
    speciesDictPath = os.path.join(chemkinfilepath,'species_dictionary.txt')
    species, reactions = loadChemkinFile(chemkinPath, speciesDictPath, readComments = readComments, useChemkinNames=False)
    make_string_labels_independent(species)
    for spec in species:
        if len(spec.molecule) == 0:
            print(spec)
    # convert species
    ct_species = [spec.toCantera(useChemkinIdentifier = False) for spec in species]
    # convert reactions
    # since this can return a rxn or list of reactions, this allows to make a list based on the returned type
    ct_reactions = []
    for rxn in reactions:
        ct_rxn = rxn.toCantera(useChemkinIdentifier = False)
        if isinstance(ct_rxn, list):
            ct_reactions.extend(ct_rxn)
        else:
            ct_reactions.append(ct_rxn)

    # save new file
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                  species=ct_species, reactions=ct_reactions)
    new_file = soln2cti.write(gas)
    # move the file to new location
    os.rename(new_file, os.path.join(chemkinfilepath,'input_nicely_named.cti'))
    # save species dictionary
    dictionary = ''
    for spec in species:
        dictionary += spec.toAdjacencyList() + '\n\n'
    f = open(os.path.join(chemkinfilepath,'species_dictionary_nicely_named.txt'),'w')
    f.write(dictionary)
    f.close()