    #!/usr/bin/env python
    # coding: utf-8
    
    # In[1]:
    
    
import time
t1 = time.time()
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
#from pyodide.ffi import to_js
import pickle
import math
from scipy.spatial.distance import cdist
from itertools import combinations
import numpy as np
from pymatgen.core.structure import Structure, Lattice, IStructure
from pymatgen.io.xyz import XYZ
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import periodic_table as PT
import os
    
def mlcp(structure,sysname,sf=1.0): 
    # In[2]:
    
    
    #files=['CaFe4Sb12.cif','1601417.cif','Clathrate-I.cif',#'EntryWithCollCode17347.cif',#@'2129946.cif',
    # %'Cu.cif','Cu7PS6.cif','Ag8SnSe6.cif','EntryWithCollCode80292.cif',#'EntryWithCollCode25.cif',
    # 'EntryWithCollCode137778.cif',#'EntryWithCollCode280375.cif','SnSe_optimized.cif','InTe.cif',
    # 'Co9S8.cif', 'CoS2.cif', 'CoSi.cif', 'CuFeS2.cif', 'TiS2.cif','TiS2.cif','VFeSb.cif',
    # #'Li2Ge11Sb2Te15.cif', 'YbCd2Sb2.cif', #'Ge12Sb2Te15.cif', #'Ag2Se.cif', 'MgAgSb.cif',
    # 'Sb2Si2Te6.cif', #'Li2Ge3Sb2Te7.cif', 'Tl4SnTe3.cif', 'Bi2Te3.cif', 'LiCoO2.cif','GeTe.cif',
    # #'FeVSb.cif', 'CoGeTe.cif', 'YbSi2.cif', #'NbCoSn.cif', #'Ag8SiSe6_off.cif', #'Yb9Ca4BaMgSb11_off.cif',
    # 'Cu12Sb4S13.cif', #'As2Te2Se.cif', 'BaBiTe3.cif', @'LaCoO3.cif', #'Eu2ZnSb2.cif',
    # #'Cu11ZnSb4S13.cif']
    #restrictions: no partial occupancies#, no particular elements like La @, more than one element %.
    
    
    # In[3]:
    
    
    #line inputs:
    
    # #file being used
    # cif=cif
    # #scale factor
    # sf=1.0
    # #template
    temp=None
    
    
    # In[4]:
    
    
    #check contacts within this distance
    radius = 5.0
    
    
    # In[5]:
    
    
    def modifile(filename, action, lines=None):
        # Reads, writes, or appends a file
        if action == "r":
            with open(filename, "r") as file:
                lines = file.readlines()
            return lines
        else:
            if lines is not None:
                with open(filename, action) as file:
                    file.writelines(lines)
                return 1
            else:
                return 0
    
    
    # In[371]:
    
    
    #modifile("cif.cif", "w", cif)
    df2 = modifile("./element_data.txt", "r")
    for i in range(len(df2)):
        df2[i] = [float(j) for j in df2[i].split("\t")]
    
    
    # In[372]:
    
    
    #scales
    structure.scale_lattice(structure.volume*(float(sf)**3))
    #some rounding, making things 0-1
    for i in range(len(structure)):
        c = structure[i].frac_coords
        for j in range(3):
            if c[j] > -1e-5 and c[j] < -1e-25:
                structure[i].frac_coords[j] = 0
            if c[j] == 1:
                structure[i].frac_coords[j] = 0
            if c[j] < -1e-5 and c[j] > -1e-1:
                structure[i].frac_coords[j] = c[j]+1
            if c[j] < 1.1 and c[j] > 1:
                structure[i].frac_coords[j] = c[j]-1
    
    
    # In[373]:
    
    
    #structure
    
    
    # In[374]:
    
    
    #unit cell as matrix zero values??? yes
    cell = [list(i) for i in structure.lattice.matrix]
    #summary of crystal characteristics
    summary = str(structure).split("\n")[:4]
    
    
    # In[375]:
    
    
    full_formula=summary[0].split("(")[1].split(")")[0].replace(" ","") 
    
    
    # In[376]:
    
    
    if temp != None:
        #modifile("cif.xyz", "w", temp)
        xyz = XYZ.from_file(temp).as_dataframe()
        species = xyz['atom'].tolist()
        x = xyz['x'].tolist()
        y = xyz['y'].tolist()
        z = xyz['z'].tolist()
        xyzs = [[x[i], y[i], z[i]] for i in range(len(x))]
        xyzs_template = [[species[i], x[i], y[i], z[i]] for i in range(len(x))]
        substructure = Structure(structure.lattice, species, xyzs, coords_are_cartesian=True)
    
    
    # In[377]:
    
    
    #time so far
    t2 = time.time()
    
    
    # In[378]:
    
    
    #making a bunch of shifts to the coordinates
    hkls = []
    for h in range(-1,2):
        for k in range(-1,2):
            for l in range(-1,2):
                hkls.append([h,k,l])
    coord_shifts = []
    cell_array = np.array(cell)
    for k in range(len(hkls)):
        coord_shifts.append(np.dot(hkls[k], cell_array))
    coord_shifts = np.array(coord_shifts)
    
    
    # In[379]:
    
    
    #making the supercell with all the extra stuff
    sup = structure.copy()
    sup.make_supercell(2)
    #neighbors in the periodic supercell
    nn = structure.get_all_neighbors(radius)
    
    
    # In[380]:
    
    
    # get voronoi data for all relevant atoms
    if temp != None:
        allvnn = []
        getvoro = [i for i in xyzs]
        for i in range(len(substructure)):
            for j in range(len(structure)):
                test_struct = np.transpose(np.repeat(structure[j].coords, 27).reshape(3, 27))
                test_sub = np.transpose(np.repeat(substructure[i].coords, 27).reshape(3, 27))
                sub = test_struct - test_sub + coord_shifts
                w = np.where(np.bincount(np.where(np.logical_and(sub < 0.001, sub > -0.001))[0])==3)[0]
                if len(w) == 1:
                    for k in range(len(nn[j])):
                        getvoro.append(nn[j][k].coords)
                    break
        getvoro = np.unique(np.array(getvoro).round(decimals=6), axis=0)
        getvoro_frac = np.dot(getvoro, np.linalg.inv(np.array(cell)))
        getvoro_where = np.where(np.logical_and(getvoro_frac>-0.001, getvoro_frac<1), 1, 0)
        getvoro_in_uc = np.where(np.sum(getvoro_where, axis=1)==3)[0]
        getvoro = np.unique(getvoro[getvoro_in_uc], axis=0)
        for i in range(len(structure)):
            for j in range(len(getvoro)):
                if np.all(np.isclose(getvoro[j], structure[i].coords)):
                    allvnn.append([value for key, value in VoronoiNN().get_voronoi_polyhedra(structure,i).items()])
                    break
            else:
                allvnn.append([])
    else:
        allvnn = VoronoiNN().get_all_nn_info(structure)
    
    
    # In[381]:
    
    
    #shows voronoi polynomial for each one
    #allvnn
    
    
    # In[382]:
    
    
    nn_options = []
    neighbor_weights = []
    neighbors = []
    neighbors2 = []
    cns = []
    stoich = []
    elems = []
    data = []
    used_for = []
    vectors = []
    geo = []
    geo2 = []
    struct_coords = []
    template = []
    template2 = []
    template3 = []
    vols = []
    s_tot = []
    en = []
    rad = []
    en2 = []
    rad2 = []
    v = 0
    
    
    # In[383]:
    
    
    for i in range(len(allvnn)):
        if structure[i].specie.Z in stoich:
            #adding onto the cumulative total of that element
            s_tot[stoich.index(structure[i].specie.Z)] += 1
            #cube of electronegativity???
            en.append(PT.Element(elems[-1]).X**3)
            #cube of radius
            rad.append(PT.Element(elems[-1]).atomic_radius_calculated**3)
        else:
            #which atomic number
            stoich.append(structure[i].specie.Z)
            #element total
            s_tot.append(1)
            #symbol of element
            elems.append(structure[i].specie.symbol)
            #electroneg cubed??? jonathan
            en.append(PT.Element(elems[-1]).X**3)
            #radius cubed
            rad.append(PT.Element(elems[-1]).atomic_radius_calculated**3)
    ensum = sum(en)/len(en) #avg cubed en
    radsum = sum(rad)/len(rad) #avg cubed rad 
    for i in range(len(en)):
        en2.append(en[i]/ensum) #cubed en normalized to average
        rad2.append(rad[i]/radsum) #cubed rad normalized to average
    
    
    # **La did not work--https://pymatgen.org/pymatgen.core.html#pymatgen.core.periodic_table.Element.atomic_radius_calculated
    # weird thing with wiki reference, using calculated column; Le, Ce, H have values but no reference
    # 
    # look into psuedo manually adding them?
    # 
    # ordered sites only--no partial occupancy? do we do the max prob? choose one probabalistically etc?
    
    # In[384]:
    
    
    for i in range(len(allvnn)): #each atom
        neighbors.append([]) #new list for each one
        neighbors2.append([])
        vols.append([])
        cns.append([0,0])
        for j in range(len(allvnn[i])): #each neighbor
            neighbors2[i].append(allvnn[i][j]['site'].to_unit_cell()) #translated to unit cell
            neighbors[i].append(allvnn[i][j]['site']) #general space?
            if temp != None:
                vols[i].append(round(allvnn[i][j]['volume'], 6)) #assigned vol to each
            else:
                vols[i].append(round(allvnn[i][j]['poly_info']['volume'], 6)) #assigned vol to each
            if allvnn[i][j]['site'].specie.Z == structure[i].specie.Z: #is it a like interaction? !!!
                cns[i][0] += 1 #yes
            else:
                cns[i][1] += 1 #no
    
    
    # In[385]:
    
    
    t3 = time.time()
    
    sn = sup.get_all_neighbors(radius) #neigbors of each 8n in the supercell
    
    dx = 2 # search x angstroms beyond the unit cell.
    def within_range(d, dx, a, b, c):
        if -dx <= d[0] and d[0] <= a+dx:
            if -dx <= d[1] and d[1] <= b+dx:
                if -dx <= d[2] and d[2] <= c+dx:
                    return(True)
                else:
                    return(False)
            else:
                return(False)
        else:
            return(False)
    
    
    # In[386]:
    
    
    for i in range(len(sup)):
        a = structure.lattice.a
        b = structure.lattice.b
        c = structure.lattice.c
        d = sup[i].coords
        for j in range(len(sn[i])):
            d = sn[i][j].coords
            if within_range(d, dx, a, b, c):
                template.append([sn[i][j].specie.symbol, d[0], d[1], d[2]]) #adding on the ones that are in the range
    
    
    # In[387]:
    
    
    [template2.append(x) for x in template if x not in template2]
    template3 = template2.copy() #taking unique ones
    
    
    # In[388]:
    
    
    for i in range(len(structure)):
        s = structure[i] #each element in the structure added to list of counts
        if s.specie.Z in stoich:
            s_tot[stoich.index(s.specie.Z)] += 1
        else:
            stoich.append(s.specie.Z)
            s_tot.append(1)
            elems.append(s.specie.symbol)
        geo.append([s.specie.symbol, round(s.coords[0],4), round(s.coords[1],4), round(s.coords[2],4)]) #element and coords
        geo2.append([round(s.frac_coords[0],4), round(s.frac_coords[1],4), round(s.frac_coords[2],4)]) #coords
        struct_coords.append(s.coords)
    struct_coords = np.array(struct_coords) 
    
    
    # In[389]:
    
    
    cell_voxel = np.array([np.array(i, dtype=np.float16) for i in cell])
    positions = []
    x = []
    y = []
    z = []
    
    
    # In[390]:
    
    
    for i in range(len(geo2)):
        positions.append(np.dot(geo2[i], cell_voxel))
    
    
    # In[391]:
    
    
    if temp != None:
        coords = np.array([i.frac_coords for i in substructure]) 
        mincv = np.min(coords, axis=0)
        maxcv = np.max(coords, axis=0)
        for i in range(3):
            if mincv[i] < 0:
                mincv[i] = 0
            if maxcv[i] > 1:
                maxcv[i] = 1
        coords_first_uc = []
        for i in range(len(xyzs)):
            c = coords[i]
            for j in range(3):
                if c[j] < 0:
                    c[j] = c[j] + 1
                if c[j] >= 1:
                    c[j] = c[j] - 1
                if c[j] < 1e-5:
                    c[j] = 0
            coords_first_uc.append([round(c[0],4), round(c[1],4), round(c[2],4)])
        ucfuc = []
        [ucfuc.append(p) for p in coords_first_uc if p not in ucfuc]
        ucfuc = np.array(ucfuc)
        xyzs_first_uc = np.dot(ucfuc, cell_voxel)
        xyzs_first_uc = np.array([[round(i[0],4), round(i[1],4), round(i[2],4)] for i in xyzs_first_uc])
        for i in range(len(xyzs_first_uc)):
            x.append(xyzs_first_uc[i][0])
            y.append(xyzs_first_uc[i][1])
            z.append(xyzs_first_uc[i][2])
        vca_small = []
        vca_weights_small = []
        cubic_grid_small = []
        filter_indicies = []
        geo_list = [list(l) for l in geo2]
        ucfuc_list = [list(l) for l in ucfuc]
        for i in range(len(ucfuc_list)):
            if ucfuc_list[i] in geo_list:
                filter_indicies.append(geo_list.index(ucfuc_list[i]))
    else:
        for i in range(len(positions)): #separating out each dimension
            x.append(positions[i][0])
            y.append(positions[i][1])
            z.append(positions[i][2])
        filter_indicies = [i for i in range(len(structure))] #number of sites
    
    
    # In[392]:
    
    
    unique = []
    #!!!
    el_data={}
    for i in range(len(s_tot)):
        el_data[i]=df2[stoich[i]]
    #el1_data = df2[stoich[0]]
    #el2_data = df2[stoich[1]] #why only the first two??? only used for binarys??? !!!
        
    e_data = []
    
    
    # In[393]:
    
    
    #!!!
    for l in range(14):
        value=0
        for i in range(len(s_tot)):
            value+=el_data[i][l]*s_tot[i]
        e_data.append(value/sum(s_tot)) # weighted average of each element's data, whole crystal
        value2=0
        for i in range(len(s_tot)):
            value2+=(el_data[i][l]-e_data[-1])**2*s_tot[i]
        e_data.append(np.sqrt(value2/sum(s_tot)))
        #e_data.append((el1_data[l]*s_tot[0] + el2_data[l]*s_tot[1])/(s_tot[0]+s_tot[1])) # weighted average of each element's data, whole crystal
        #e_data.append(np.sqrt((s_tot[0]*(el1_data[l]-e_data[-1])**2 + (s_tot[1]*(el2_data[l]-e_data[-1])**2))/(s_tot[0]+s_tot[1]))) # weighted square deveation of each element's data
    
    
    
    #e_data.append((el1_data[l]*s_tot[0] + el2_data[l]*s_tot[1])/(s_tot[0]+s_tot[1])) # weighted average of each element's data, whole crystal
    #e_data.append(np.sqrt((s_tot[0]*(el1_data[l]-e_data[-1])**2 + (s_tot[1]*(el2_data[l]-e_data[-1])**2))/(s_tot[0]+s_tot[1]))) # weighted square average of each element's data
    
    
    # In[394]:
    
    
    for i in range(len(structure)):
        if i in filter_indicies:
            for j in range(len(neighbors[i])): #for each neighbor atom of each one
                ct1_data = df2[structure[i].specie.Z] #the element in the structure info
                ct2_data = df2[neighbors[i][j].specie.Z] # the other one info
                length = round(np.linalg.norm(structure[i].coords - neighbors[i][j].coords), 4) #gets magnitude of length diff
                test = np.transpose(np.repeat(neighbors2[i][j].coords, len(structure)).reshape(3, len(structure))) #repeat unit cell translated coords n times
                sub = test - struct_coords #difference of everything to the contact one
                k = np.where(np.bincount(np.where(np.isclose(sub,np.zeros(sub.shape),rtol=0.001,atol=0.000001))[0])==3)[0][0] #the one the contact is
                cns_neighbor = cns[k] #interaction characters (like/unlike)
                sc = structure[i].frac_coords.tolist() #location of the first
                nc = neighbors[i][j].frac_coords.tolist() #location of the neighbor
                if [i, k, sc, nc] in unique or [k, i, nc, sc] in unique: #so as to not repeat a contact twice from each's perspective
                    continue
                else:
                    vven = round(vols[i][j]*(en2[i]+en2[k]), 4) #vol of contact times sum of cubed averaged electronegativity
                    neighbor_weights.append(vven*0.1*(1/length)) #weighting neighbor by the vol-EN and closer ones more
                    is_same = 0 if (neighbors2[i][j].coords == neighbors[i][j].coords).all() else 1 #same unit cell?
                    #adding in element properties of interest
                    sum_met_rad = (ct1_data[2] + ct2_data[2])/100
                    sum_met_vol = round((ct1_data[2]/100)**3 + (ct2_data[2]/100)**3, 4)
                    length_cubed = round(length**3, 4)
                    en_range = round(abs(ct1_data[6] - ct2_data[6]), 4)
                    #comparisons between 'expected' and the structural ones
                    diff_len = round(sum_met_rad - length, 4)
                    diff_vorovol = round(vven - length_cubed, 4)
                    diff_metvol = round(sum_met_vol - length_cubed, 4)
                    extras = [length, vven, sum_met_rad, sum_met_vol, length_cubed, en_range, diff_len, diff_vorovol, diff_metvol] #storing these
                    c_data = []
                    cns_data = []
                    for l in range(14): #for all 14 base properties
                        c_data.append((ct1_data[l]+ct2_data[l])/2) #average
                        c_data.append(abs(ct1_data[l]-ct2_data[l])/2) #half diff
                    for l in range(2):
                        cns_data.append((cns[i][l]+cns_neighbor[l])/2) #avg of the like/unlike of both
                        cns_data.append(abs(cns[i][l]-cns_neighbor[l])/2) #same thing? should be neg??? (changed)
                    data.append(extras + e_data + c_data + cns_data) #setting up feature list
                    vectors.append(neighbors[i][j].coords - structure[i].coords) # adding on the spatial distance
                    if is_same == 0:
                        used_for.append([i,k]) #if in the unit cell, this is the represenative one??
                    else:
                        used_for.append([i,len(structure)]) #just saying it's something else for now??
                    unique.append([i, k, structure[i].frac_coords.tolist(), neighbors[i][j].frac_coords.tolist()]) #recording the one done
    
    
    # In[395]:
    
    
    npdata = np.array([np.array(line) for line in data]) #into numpy
    npdata3 = [list(a) for a in npdata] #just data again
    assignments = []
    unique_npdata = []
    
    
    # In[396]:
    
    
    for i in range(len(npdata3)):
        if npdata3[i] not in unique_npdata:
            #if npdata3[i][:28]+npdata3[i][42:56]+npdata3[i][28:42]+npdata3[i][56:] not in unique_npdata:
            if npdata3[i][:37]+npdata3[i][51:65]+npdata3[i][37:51]+npdata3[i][65:] not in unique_npdata: #swap to diff before of avgs. why??? ranges wrong??? changed
                unique_npdata.append(npdata3[i]) #unique data one
                assignments.append(len(unique_npdata)-1)
            else:
                assignments.append(unique_npdata.index(npdata3[i][:37]+npdata3[i][51:65]+npdata3[i][37:51]+npdata3[i][65:])) #which one it's using
                #assignments.append(unique_npdata.index(npdata3[i][:28]+npdata3[i][42:56]+npdata3[i][28:42]+npdata3[i][56:])) #changed
        else:
            assignments.append(unique_npdata.index(npdata3[i])) #record which assignment used
    
    
    # In[397]:
    
    
    unique_npdata = [np.array(a) for a in unique_npdata]
    
    
    # In[398]:
    
    
    f = open('./rfc.pickle', 'rb') #loading classifier
    rfc = pickle.load(f)
    f.close()
    
    
    # In[399]:
    
    
    pred_rfc = rfc.predict(unique_npdata) #which group they are in: 0 neg, 1 int, 2 pos
    predictions_rfc = np.reshape(pred_rfc, (pred_rfc.shape[0],1)) #reshape into arrays
    npdata2 = np.hstack((unique_npdata, predictions_rfc)) #adding predictions to each
    
    
    # In[400]:
    
    
    rfc = 0 #closing the model
    
    
    # In[401]:
    
    
    f = open('./rfr.pickle', 'rb') #opening regressor
    rfr = pickle.load(f)
    f.close()
    
    
    # In[402]:
    
    
    unique_predictions_rfr = rfr.predict(npdata2) #making the predictions
    
    
    # In[403]:
    
    
    rfr = 0 #closing the model
    
    
    # In[404]:
    
    
    predictions_rfr = []
    for i in range(len(assignments)):
        predictions_rfr.append(unique_predictions_rfr[assignments[i]]) #adding the predicted value for each contact
    predictions_rfr = np.array(predictions_rfr)
    geo2 = np.array([np.array(i) for i in geo2]) #each location in fractional 
    neighborr = unique.copy() #each unique neighbor connection
    
    
    # In[405]:
    
    
    t4 = time.time()
    
    
    # In[406]:
    
    
    for j in range(len(neighborr)):
        for i in range(3): #xyz
            if neighborr[j][2][i] >= 0 and neighborr[j][2][i] < 1: #if main is in unit cell
                neighborr[j][2][i] = 0 #the unit cell
            elif neighborr[j][2][i] >= -1 and neighborr[j][2][i] < 0: #if main is in in one below
                neighborr[j][2][i] = -1 #the one below
            else: # neighbor should be greater than one
                neighborr[j][2][i] = 1 #the one above
            if neighborr[j][3][i] >= 0 and neighborr[j][3][i] < 1: #if neighbor is in unit cell
                neighborr[j][3][i] = 0
            elif neighborr[j][3][i] >= -1 and neighborr[j][3][i] < 0: #if neighbor is in one below
                neighborr[j][3][i] = -1
            else: # neighbor should be greater than one
                neighborr[j][3][i] = 1 #if neighbor is in one above
    
    
    # In[407]:
    
    
    contact_verts = []
    contact_atoms = []
    
    buffer = 7
    voxels_per_angstrom = 5
    
    
    # In[408]:
    
    
    def make_cubic_grid(voxels_per_angstrom, mins=[0,0,0], maxs=[1,1,1], buff=0):
        # determine the cell vector lengths
        lena = np.sqrt(cell_voxel[0][0]**2 + cell_voxel[0][1]**2 + cell_voxel[0][2]**2) #a
        lenb = np.sqrt(cell_voxel[1][0]**2 + cell_voxel[1][1]**2 + cell_voxel[1][2]**2) #b
        lenc = np.sqrt(cell_voxel[2][0]**2 + cell_voxel[2][1]**2 + cell_voxel[2][2]**2) #c
        # calculate the length needed for increments, which is (max-min) + 2 * buffer, or len, whichever is less.
        lengths = [lena, lenb, lenc]
        start = [0,0,0]
        stop = [1,1,1]
        for i in range(3):
            # along a, b, and c, if the substructure is entirely contained within the unit cell, we don't need the whole range.
            # note: if the substructure lies along the unit cell edge on one side, but not the other, we can't do this.
            if mins[i]-(buff/lengths[i]) > 0: #the min (0) is more than the buffer as a percent of the side
                if maxs[i]+(buff/lengths[i]) < 1: #and the max (1) plus the puffer and side length is over 1 (usually true)
                    start[i] = mins[i]-(buff/lengths[i]) #start and stop get buffered
                    stop[i] = maxs[i]+(buff/lengths[i]) 
            if lengths[i] > ((maxs[i]-mins[i]) * lengths[i] + buff * 2): #length larger than max-min times range with buffer
                lengths[i] = ((maxs[i]-mins[i]) * lengths[i] + buff * 2) #use lower value for new lengths
        # calculate the number of increments along each vector
        na = int(math.ceil(lengths[0] * voxels_per_angstrom)) + 1
        nb = int(math.ceil(lengths[1] * voxels_per_angstrom)) + 1
        nc = int(math.ceil(lengths[2] * voxels_per_angstrom)) + 1
        na_max = int(math.ceil(lena * voxels_per_angstrom)) #position of the unit cell in the increment units
        nb_max = int(math.ceil(lenb * voxels_per_angstrom))
        nc_max = int(math.ceil(lenc * voxels_per_angstrom))
        # create the fractional coordinates along each vector
        a_space = np.linspace(start[0], stop[0], na, dtype="float16")
        b_space = np.linspace(start[1], stop[1], nb, dtype="float16")
        c_space = np.linspace(start[2], stop[2], nc, dtype="float16")
        # center the coordinates (ex.: [0.0, 0.1, ... 0.8, 0.9] ==> [0.05, 0.015, ... 0.85, 0.95]
        a_space = (a_space[:-1] + a_space[1:]) / 2  # gets centers of the voxels
        b_space = (b_space[:-1] + b_space[1:]) / 2  # gets centers of the voxels
        c_space = (c_space[:-1] + c_space[1:]) / 2  # gets centers of the voxels
        # combine a, b, and c coordinates
        grid_positions = np.array([np.array([x, y, z], dtype="float16") for x in a_space for y in b_space for z in c_space])
        return(np.dot(grid_positions, cell_voxel), [na_max, nb_max, nc_max]) #matrix product
    
    
    # In[409]:
    
    
    if temp != None:
        cubic_grid,voxel_max = make_cubic_grid(voxels_per_angstrom, mincv, maxcv, buffer) #more complex
    else:
        cubic_grid,voxel_max = make_cubic_grid(voxels_per_angstrom) #only a function of vpa
    vmt = voxel_max[0]*voxel_max[1]*voxel_max[2] #area of the unit cell in those units as a prism
    #vca = [] # added as [index of p1, index of p2, [h,k,l] of p1, [h,k,l] of p2]
    
    
    # In[410]:
    
    
    offset = []
    p_test_hkl = []
    p_test_xyz = []
    p_test_hkl2 = []
    p_test_xyz2 = []
    p_test_hkl3 = []
    p_test_xyz3 = []
    for i in range(len(positions)):
        for h in range(-2,3):
            for k in range(-2,3):
                for l in range(-2,3):
                    p_test = []
                    for j in range(3):
                        p_test.append(positions[i][j] + cell_voxel[0][j]*h + cell_voxel[1][j]*k + cell_voxel[2][j]*l) #getting the vlaues for each one when transferred to the next unit cell
                    p_test_hkl2.append([i,h,k,l])
                    p_test_xyz2.append(p_test)
    
    
    # In[411]:
    
    
    t5 = time.time()
    
    minmax = [[min(x)-buffer, max(x)+buffer],[min(y)-buffer, max(y)+buffer],[min(z)-buffer, max(z)+buffer]] #buffer the positions
    
    
    # In[412]:
    
    
    for i in range(len(p_test_hkl2)): #for each shift of each atom
        pp = p_test_xyz2[i] #the location of it
        if pp[0] > minmax[0][0] and pp[0] < minmax[0][1]: #in x range
            if pp[1] > minmax[1][0] and pp[1] < minmax[1][1]: #in y range
                if pp[2] > minmax[2][0] and pp[2] < minmax[2][1]: #in z range
                    p_test_hkl3.append(p_test_hkl2[i]) #adding on the inlimit ones
                    p_test_xyz3.append(np.array(p_test_xyz2[i], dtype=np.float32)) #samedeal
    
    
    # In[413]:
    
    
    if temp != None:
        # remove atoms from superstructure which are more than <buffer> angstroms away from any one atom in the substructure
        distances = cdist(p_test_xyz3, xyzs_first_uc)
        mins = np.min(distances, axis=1)
        maybe = np.where(mins<buffer,1,0)
        for i in range(len(p_test_hkl3)):
            if maybe[i] == 1:
                p_test_hkl.append(p_test_hkl3[i])
                p_test_xyz.append(p_test_xyz3[i])
    else:
        p_test_xyz = p_test_xyz3
        p_test_hkl = p_test_hkl3
    
    
    # In[414]:
    
    
    p_test_xyz = np.array(p_test_xyz, dtype="float32")
    
    
    # In[415]:
    
    
    if temp != None:
        cell_voxel = np.array(cell_voxel, dtype="float32")
        newmin = np.dot(np.min(p_test_xyz, axis=0), np.linalg.inv(cell_voxel))
        newmax = np.dot(np.max(p_test_xyz, axis=0), np.linalg.inv(cell_voxel))
        for i in range(3):
            if newmin[i] < 0:
                newmin[i] = 0
            if newmax[i] > 1:
                newmax[i] = 1
        newmin = np.dot(newmin, cell_voxel)
        newmax = np.dot(newmax, cell_voxel)
        august = np.where(np.logical_and(cubic_grid > newmin, cubic_grid < newmax))[0]
        sept = np.bincount(august)
        octo = np.where(sept==3)[0]
        new_grid = cubic_grid[octo]
        cubic_grid = new_grid
    
    
    # In[416]:
    
    
    q1 = int(len(cubic_grid)/4) #getting quartiles of the sample points
    q2 = int(len(cubic_grid)/2)
    q3 = int(3*len(cubic_grid)/4)
    
    
    # In[417]:
    
    
    summ1 = cdist(cubic_grid[:q1],p_test_xyz) #distance of each grid point to the atoms
    summ2 = cdist(cubic_grid[q1:q2],p_test_xyz) #second quartile
    summ3 = cdist(cubic_grid[q2:q3],p_test_xyz) #3rd
    summ4 = cdist(cubic_grid[q3:],p_test_xyz) 
    summ5 = np.concatenate((summ1,summ2), axis=0, dtype="float16") #combined first half
    summ1, summ2 = [], []  #clear
    summ6 = np.concatenate((summ3,summ4), axis=0, dtype="float16") #second half
    summ3, summ4 = [], [] #clear
    summ = np.concatenate((summ5,summ6),axis=0) #full version
    
    
    # In[418]:
    
    
    fs = np.min(summ, axis=1) #lowest distance value in each set (reflects closest voxel!)
    ss = np.partition(summ, 1)[:,1] + 1/voxels_per_angstrom #all the second closest distances, plus a voxel size
    
    
    # In[419]:
    
    
    vca = []
    vca_weights = []
    cubic_grid_big = []
    
    
    # In[420]:
    
    
    apv = 0.5/voxels_per_angstrom #half of a voxel length
    summlen = len(summ[0]) #total atoms
    
    
    # In[421]:
    
    
    for i in range(len(summ)): #for each grid point
        bi = np.where(summ[i] <= ss[i])[0] #(two) lowest values
        b = summ[i][bi] #those lowest distances
        c = [a+b for a,b in combinations(b,2)] #sum of pairs of distances among them
        ci = [[a,b] for a,b in combinations(bi,2)] #pairs of indices
        d = min(c) + apv #min sum distance plus half a cell length
        e = [ci[j] for j in range(len(c)) if c[j] <= d]# within 0.5 voxlen of the lowest summed distance ones
        for j in range(len(e)): #for each of the ties
            p1 = p_test_hkl[e[j][0]] #which atom & hkl the lowest refers to
            p2 = p_test_hkl[e[j][1]] #which atomh & hkl the 2nd lowest refers to
            vca.append([p1[0]] + [p2[0]] + p1[1:] + p2[1:]) #the two closest atoms and their hkl coordinates
            cubic_grid_big.append(cubic_grid[i]) #making the cubic grid of points
            vca_weights.append(1/len(e)) #for how many close ones, that's the weight
    
    
    # In[422]:
    
    
    summ = [] #clear
    
    t6 = time.time()
    
    unique_contacts = []
    unique_contacts_array = []
    voxel_groups = []
    uca, reconstruct = np.unique(np.array(vca), axis=0, return_inverse=True) #unique contacts and the label for each
    
    
    # In[423]:
    
    
    for i in range(len(uca)):
        voxel_groups.append(np.where(reconstruct == i)[0]) #adding the indices for each of the unique contacts
        unique_contacts.append([uca[i][0], uca[i][1], list(uca[i][2:5]), list(uca[i][5:])]) #adding in the unique ones as a separated list
        unique_contacts_array.append(np.concatenate((uca[i][0], uca[i][1], uca[i][2:5], uca[i][5:]), axis=None)) #all one list
    
    
    # In[424]:
    
    
    unique_contacts2 = unique_contacts.copy()
    voxel_groups2 = voxel_groups.copy()
    
    
    # In[425]:
    
    
    hklsnp = np.array(hkls) # as array
    
    
    # In[426]:
    
    
    uc2 = np.array([unique_contacts[i][2] for i in range(len(unique_contacts))]) #the first atom's hkl
    uc3 = np.array([unique_contacts[i][3] for i in range(len(unique_contacts))]) #the second one's hkl
    
    
    # In[427]:
    
    
    #making each unique contact (atoms and hkl) and translating it in all directions with hkl
    uc_options = np.array([np.concatenate((unique_contacts[i][:2], uc2[i]+j, uc3[i]+j), axis=None)for i in range(len(unique_contacts)) for j in hklsnp])
    
    
    # In[428]:
    
    
    paired_groups = []
    paired_groups_hkl = []
    
    
    # In[429]:
    
    
    for i in range(len(unique_contacts)):
        test = np.transpose(np.repeat(unique_contacts_array[i], 27*len(unique_contacts)).reshape(8, 27*len(unique_contacts))) #making it match the many dimension one
        sub = test - uc_options #seeing if it matches
        w = np.where(np.bincount(np.where(sub==0)[0])==8)[0] #seeing if it matches fully
        paired_groups.append(w//27) #which unique contact gets the match
        paired_groups_hkl.append(hklsnp[w%27]) #which coordinate shift matches the one
    
    
    # In[430]:
    
    
    for i in range(len(paired_groups)):
        new = []
        newhkl = []
        for j in range(len(paired_groups[i])): #each index of the same contacts for each contact
            if paired_groups[i][j] not in new:
                new.append(paired_groups[i][j]) #add in the index of the unique contact that gets there
                newhkl.append(paired_groups_hkl[i][j]) #hkl shift of it
        paired_groups[i] = new #putting in the unique indexes
        paired_groups_hkl[i] = newhkl #putting in the unique hkls
    
    
    # In[431]:
    
    
    unique_pg = []
    unique_pg_hkl = []
    for i in range(len(paired_groups)):
        pg = sorted(paired_groups[i]) #putting things in order
        if pg not in unique_pg:
            unique_pg.append(paired_groups[i]) #unique paired groups
            unique_pg_hkl.append(paired_groups_hkl[i]) #hkls
    
    
    # In[432]:
    
    
    unique_contacts3 = []
    voxel_groups3 = []
    
    
    # In[433]:
    
    
    for i in range(len(unique_pg)):
        upgi = unique_pg[i][0] #the first one
        voxel_groups3.append(voxel_groups[upgi]) #which voxel groups near it, for all of them
        unique_contacts3.append(unique_contacts[upgi]) #which contacts, for all of them
        if len(unique_pg[i]) > 1: #if theres more than one unique index
            for j in range(1,len(unique_pg[i])):
                upgij = unique_pg[i][j] #the index
                hkl = unique_pg_hkl[i][j] #the hkl
                vg = voxel_groups[upgij] #the contact indices near it
                grid_shift = np.dot(hkl, cell) #product with the cell
                points = []
                weights = []
                l = len(cubic_grid_big)
                count = 0
                for k in range(len(vg)): #for each contact
                    points.append(cubic_grid_big[vg[k]]) #the coordinates of each contact
                    weights.append(vca_weights[vg[k]]) #weight of each point
                    voxel_groups3[-1] = np.append(voxel_groups3[-1], l+count) #adding however many over the grid size to the end
                    count += 1
                cubic_grid_big = np.append(cubic_grid_big, points + grid_shift, axis=0) #adding on the shifted coordinates 
                vca_weights = vca_weights + weights #appending the weights of the used ones to the previous weights
    
    
    # In[434]:
    
    
    uc = np.array([[0,0,0,0,0,0,0,0]]) #eight
    for i in range(len(unique_contacts3)): #for each unique one
        u = unique_contacts3[i] #the unique contact
        uc = np.vstack([uc, np.concatenate((np.array(u[0:2]), np.array(u[2]), np.array(u[3])))]) #rearranging into a 8 item list
    uc = uc[1:] #taking all the ones with data
    
    
    # In[435]:
    
    
    voxel_groups4 = voxel_groups3.copy()
    unique_contacts4 = unique_contacts3.copy()
    
    
    # In[436]:
    
    
    hkls
    
    
    # In[437]:
    
    
    cell_voxel
    
    
    # In[438]:
    
    
    for i in range(len(neighborr)):  #each unique neighbor contact
        if neighborr[i] in unique_contacts3: #they match, the unique contact is the neighbor one
            pass
        elif [neighborr[i][1], neighborr[i][0], neighborr[i][3], neighborr[i][2]] in unique_contacts3: #atom order swapped
            pass
        else: # quite often, a symmetrically equivalent bond has been found. For instance, missing the [0,4,[0,0,0],[0,0,-1]], but we have the [0,4,[0,0,1],[0,0,0]]
            x = neighborr[i]
            y = np.array(neighborr[i][2]) #first hkl
            z = np.array(neighborr[i][3]) #second
            for hkl in hkls:
                s1 = np.concatenate((np.array([x[0], x[1]]), y+hkl, z+hkl)) #the contacts with hkls shifted
                w = np.bincount(np.where(uc==s1)[0]) #find the matching ones
                try:
                    a = max(w)
                    if max(w) == 8: #all match
                        m = np.argmax(w)
                        vg = voxel_groups3[m] #find the appropriate voxel of the match
                        unique_contacts4.append(x) #add neighbor as unique contact
                        voxel_groups4.append([])
                        grid_shift = -np.dot(hkl, cell_voxel) #negative times ??? jonathan
                        points = []
                        weights = []
                        l = len(cubic_grid_big)
                        count = 0
                        for j in range(len(vg)):
                            points.append(cubic_grid_big[vg[j]]) #the coordinates of each contact
                            weights.append(vca_weights[vg[j]]) #weight of each point
                            voxel_groups4[-1].append(l+count) #adding however many over the grid size to the end
                            count += 1
                        cubic_grid_big = np.append(cubic_grid_big, points + grid_shift, axis=0) #adding on the shifted coordinates 
                        vca_weights = vca_weights + weights #appending the weights of the used ones to the previous weights
                        break
                except:
                    pass
                s2 = np.concatenate((np.array([x[1], x[0]]), z+hkl, y+hkl)) #if that doesnt work, swap the hkls
                w = np.bincount(np.where(uc==s2)[0]) #see if they match
                try:
                    if max(w) == 8: #and do the same stuff
                        m = np.argmax(w)
                        vg = voxel_groups3[m]
                        unique_contacts4.append(x)
                        voxel_groups4.append([])
                        grid_shift = -np.dot(hkl, cell_voxel)
                        points = []
                        weights = []
                        l = len(cubic_grid_big)
                        count = 0
                        for j in range(len(vg)):
                            points.append(cubic_grid_big[vg[j]])
                            weights.append(vca_weights[vg[j]])
                            voxel_groups4[-1].append(l+count)
                            count += 1
                        cubic_grid_big = np.append(cubic_grid_big, points + grid_shift, axis=0)
                        vca_weights = vca_weights + weights
                        break
                except:
                    pass
    
    
    
    # In[439]:
    
    
    voxel_groups5 = []
    unique_contacts5 = []
    
    
    # In[440]:
    
    
    for i in range(len(unique_contacts4)): #ALL THE CONTACTS we had and we found
        if unique_contacts4[i] in neighborr: #adding on the ones that match, contacts and voxels
            unique_contacts5.append(unique_contacts4[i])
            voxel_groups5.append(voxel_groups4[i])
        elif [unique_contacts4[i][1], unique_contacts4[i][0], unique_contacts4[i][3], unique_contacts4[i][2]] in neighborr: #adding on the ones if swapped
            unique_contacts5.append(unique_contacts4[i])
            voxel_groups5.append(voxel_groups4[i])
        else:
            pass
    #now we have all and only the ones that match neighborr
    
    
    # In[441]:
    
    
    vca_weights = np.array(vca_weights) #weight of each grid point
    atom_voxels = []
    #groups = []
    #group_weights = []
    #group_scales = []
    coeffs = []
    bptable = []
    cubic_grid_big = np.array(cubic_grid_big)
    parts = []
    bptableguide=[]
    
    
    # In[442]:
    
    
    for i in range(len(structure)): #for each atom in the structure
        atom_voxels.append(0)
        coeffs.append([0 for j in range(49)]) #48 spots for coeffs per atom, why the extra 24???
        if i in filter_indicies: # for each atom
            for j in range(len(predictions_rfr)): #each contact
                if i in used_for[j]: #if the atom we are looking at in is the contact
                    neigh = neighborr[j] #the relation there
                    try:
                        index = unique_contacts5.index(neigh) #find the index of the contact
                    except:
                        try:
                            index = unique_contacts5.index([neigh[1], neigh[0], neigh[3], neigh[2]]) #index of the contact if swapped
                        except:
                            break
                    vg = voxel_groups5[index] #the voxel groups there
                    points = np.take(cubic_grid_big, vg, axis=0) - structure[i].coords #getting the nearby points as how far they are from the atom
                    weights = np.take(vca_weights, vg) #getting the number of voxels for each
    
                    if i == neigh[0]: #regularizing the order
                        s0 = structure[neigh[0]]
                        s1 = structure[neigh[1]]
                    else:
                        s0 = structure[neigh[1]]
                        s1 = structure[neigh[0]]
                    # While it seems unintuitive to multiply the weighted pressure prediction by sum(weights)/len(weights),
                    # this is entirely necessary for the bptable values to be correct.
                    prediction = sum(weights)*neighbor_weights[j]*predictions_rfr[j]/len(vg) #weighted average of pressure prediction
                    bptable.append([s0.specie.symbol, s0.coords, s1.specie.symbol, s1.coords, prediction, sum(weights), 0.5*sum(weights)*structure.volume/vmt])
                    bptableguide.append([i,j,prediction])
                    #values here: first atom/coords, second, ml-cp, number of points nearby, half (bc of two contacts for 1-sum??) of the volume of the contact taken up
                    
                    """
                    # This code is used for diagnostics. Don't delete it, but
                    # don't run it unless necessary either, it slows the main
                    # for loop down by about a factor of 3.
                    l1 = list(neighborr[j][:2])
                    l2 = list(np.flip(neighborr[j][:2]))
                    if l1 in groups:
                        group_weights[groups.index(l1)].append(np.sum(weights)) #add on the points of this one to its associate
                        group_scales[groups.index(l1)].append(predictions_rfr[j]) #same deal with predictions
                    elif l2 in groups:
                        group_weights[groups.index(l2)].append(np.sum(weights))
                        group_scales[groups.index(l2)].append(predictions_rfr[j])
                    else:
                        groups.append(l1) #each contact
                        group_weights.append([np.sum(weights)]) #adding on the points that match here
                        group_scales.append([predictions_rfr[j]]) #adding on the prediction
                    """
    
                    atom_voxels[-1] += sum(weights) #add on the number of voxels there to the one for the atom
    
                    scale = weights*neighbor_weights[j]*predictions_rfr[j]/len(vg) #getting a micro prediction for each voxel, basically, adds up to prediction
                    h = np.hypot(np.hypot(points[:,0], points[:,1]), points[:,2]) #distance of each point to atom
                    phi = np.arctan2(points[:,1], points[:,0]) #angle in xy, azimuthal
                    theta = np.arccos(points[:,2]/h) #angle off of the z, polar angle
                    cost = np.cos(theta) #trig functions
                    sint = np.sin(theta)
                    cosp = np.cos(phi)
                    sinp = np.sin(phi)
    
                    #adding all these onto the first position for the atom
                    #REAL spherical harmonics
                    coeffs[i][0] += np.sum(np.ones(len(points))*scale*0.5*(1/math.pi)**0.5) 
                    #sum of 0,0 harmonic times each contributing prediction, with a unit array in there for some reason
    
                    coeffs[i][1] += np.sum(scale*0.5*(3/(math.pi))**0.5*cost) #sum of 1,0 harmonic times contributing predictions
                    coeffs[i][2] += np.sum(-scale*(3/(8*math.pi))**0.5*sint*cosp*(2)**0.5) 
                    #sum of 1,1 harmonic times contributing predictions, negative from odd rule ??? jonathan
                    coeffs[i][3] += np.sum(-scale*(3/(8*math.pi))**0.5*sint*sinp*(2)**0.5)
                    #sum of 1,-1 harmonic times contributing predictions, why negative??
    
                    coeffs[i][4] += np.sum(scale*0.25*(5/(math.pi))**0.5*(3*cost**2-1)) #sum of 2,0 harmonic times contributing predictions
                    coeffs[i][5] += np.sum(-scale*0.5*(15/(math.pi))**0.5*sint*cosp*cost) #2,1 why neg??
                    coeffs[i][6] += np.sum(-scale*0.5*(15/(math.pi))**0.5*sint*sinp*cost) #2,-1 why neg??
                    coeffs[i][7] += np.sum(scale*0.25*(15/(math.pi))**0.5*sint**2*np.cos(2*phi)) #2,2
                    coeffs[i][8] += np.sum(scale*0.25*(15/(math.pi))**0.5*sint**2*np.sin(2*phi)) #2,-2
    
                    coeffs[i][9] += np.sum(scale*0.25*(7/(math.pi))**0.5*(5*cost**3-3*cost)) #3,0
                    coeffs[i][10] += np.sum(-scale*0.125*(21/(math.pi))**0.5*sint*(5*cost**2-1.0)*cosp*(2)**0.5) #3,1 i suppose
                    coeffs[i][11] += np.sum(-scale*0.125*(21/(math.pi))**0.5*sint*(5*cost**2-1.0)*sinp*(2)**0.5) #and so on. trust.
                    coeffs[i][12] += np.sum(scale*0.25*(105/(2*math.pi))**0.5*sint**2*(cost)*np.cos(2*phi)*(2)**0.5)
                    coeffs[i][13] += np.sum(scale*0.25*(105/(2*math.pi))**0.5*sint**2*(cost)*np.sin(2*phi)*(2)**0.5)
                    coeffs[i][14] += np.sum(-scale*0.125*(35/(math.pi))**0.5*sint**3*np.cos(3*phi)*(2)**0.5)
                    coeffs[i][15] += np.sum(-scale*0.125*(35/(math.pi))**0.5*sint**3*np.sin(3*phi)*(2)**0.5)
    
                    coeffs[i][16] += np.sum(scale*(3/16)*(1/(math.pi))**0.5*(35*cost**4-30*cost**2+3)) #4,0
                    coeffs[i][17] += np.sum(-scale*(3/8)*(5/(math.pi))**0.5*sint*(7*cost**3-3*cost)*cosp*(2)**0.5)
                    coeffs[i][18] += np.sum(-scale*(3/8)*(5/(math.pi))**0.5*sint*(7*cost**3-3*cost)*sinp*(2)**0.5)
                    coeffs[i][19] += np.sum(scale*(3/8)*(5/(2*math.pi))**0.5*sint**2*(7*cost**2-1)*np.cos(2*phi)*(2)**0.5)
                    coeffs[i][20] += np.sum(scale*(3/8)*(5/(2*math.pi))**0.5*sint**2*(7*cost**2-1)*np.sin(2*phi)*(2)**0.5)
                    coeffs[i][21] += np.sum(-scale*(3/8)*(35/(math.pi))**0.5*sint**3*cost*np.cos(3*phi)*(2)**0.5)
                    coeffs[i][22] += np.sum(-scale*(3/8)*(35/(math.pi))**0.5*sint**3*cost*np.sin(3*phi)*(2)**0.5)
                    coeffs[i][23] += np.sum(scale*(3/16)*(35/(2*math.pi))**0.5*sint**4*np.cos(4*phi)*(2)**0.5)
                    coeffs[i][24] += np.sum(scale*(3/16)*(35/(2*math.pi))**0.5*sint**4*np.sin(4*phi)*(2)**0.5)
    
    
    
    # In[443]:
    
    
    data = []
    cpvalues=[]
    sitenames=[]
    net_pressure = 0
    
    
    # In[444]:
    
    
    for i in range(len(atom_voxels)): #for each atom
        net_pressure += atom_voxels[i] * 29421.0265*coeffs[i][0]*np.sqrt(4*math.pi) / 2 
        #calculating total pressure--what is this from??? jonathan
        data.append([structure[i].specie.symbol, '{0:.2f} GPa'.format(round(29421.0265*coeffs[i][0]*np.sqrt(4*math.pi),2))]) #reporting the pressure
        cpvalues.append(29421.0265*coeffs[i][0]*np.sqrt(4*math.pi))
        sitenames.append(structure[i].specie.symbol)
    # only the first value contributes. is this a problem???
    if temp == None:
        data.append(["Total", '{0:.2f} GPa'.format(round(net_pressure / sum(atom_voxels),2))]) #normalizing pressure to cell size
        #net pressure is each site's pressure coefficient weighted by the number of voxels it involves
    else:
        data.append(["Total", "undeterminable"])
    
    
    # In[445]:
    
    
    #sysname #file name
    # qps = []
    # qp_lmax = []
    # qps_total = []
    # qp_lines = ["_"*85+"\n", "  CP Quadrupole Report\n"]
    
    
    # In[446]:
    
    
    # import os
    
    # Specify the directory name
    if sysname==full_formula:
        directory_name=sysname
    else:
        directory_name = sysname+"_"+full_formula
    
    # # Create the directory
    # try:
    #     os.mkdir(directory_name)
    #     #print(f"Directory '{directory_name}' created successfully.")
    # except FileExistsError:
    #     pass
    #     #print(f"Directory '{directory_name}' already exists.")
    # except PermissionError:
    #     print(f"Permission denied: Unable to create '{directory_name}'.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    
    # In[447]:
    
    
    # for i in range(len(coeffs)): #for each atom
    #     qps.append([0,0,0,0,0])
    #     for j in range(5):
    #         for k in range(j**2,(j+1)**2): #an increasing odd number for each, as in the harmonics
    #             qps[i][j] += coeffs[i][k]**2 #combining each level squared, for l=0 to 4 for each atom
    
    
    # In[448]:
    
    
    # for i in range(len(qps)): #for each atom
    #     qp_lmax.append([0,0,0])
    #     qps_total.append([])
    #     for j in range(3):
    #         for k in range(j+3):
    #             qp_lmax[i][j] += qps[i][k] #summing the combined coeff values, starting with first 3, up to all.
    #     for j in range(3):
    #         try: 
    #             qps_total[i].append(qps[i][2]/qp_lmax[i][j]) #how much the l=2 level comb coeff (d basically) is of the total added coeffs, three here too, how quadrupolar it is
    #         except:
    #             qps_total[i].append(0)
    #     for j in range(5):
    #         try:
    #             qps_total[i].append(qps[i][j]/qp_lmax[i][2]) #new, contribution of each
    #         except:
    #             qps_total[i].append(0)
    #     try:
    #         qps_total[i].append(coeffs[i][4]**2/qp_lmax[i][2]) #new, dz^2 contribution
    #     except:
    #         qps_total[i].append(0)
    
    
    # In[449]:
    
    
    # for i in range(len(qps_total)): #for each atom
    #     qp_lines.append("    Atom {0:3n} ({1:2s})    (2/2max): {2:.6f}  (2/3max): {3:.6f}  (2/4max): {4:.6f}  (l = 0): {5:.6f}  (l = 1): {6:.6f}  (l = 2): {7:.6f}  (l = 3): {8:.6f}  (l = 4): {9:.6f}  (z2): {10:.6f}\n".format(i+1, geo[i][0], qps_total[i][0], qps_total[i][1], qps_total[i][2],qps_total[i][3],qps_total[i][4],qps_total[i][5],qps_total[i][6],qps_total[i][7],qps_total[i][8]))
    #values here: atom number, element, l2 contribution to 0-2, 0-3, 0-4
    
    
    # In[450]:
    
    
    #qp_lines.append("_"*85+"\n") #formatting
    
    
    # In[451]:
    
    
    #pm = ["0p", "1p", "1m", "2p", "2m", "3p", "3m", "4p", "4m", "5p", "5m", "6p", "6m"]
    
    
    # In[452]:
    
    
    #coeff_labels = ["l_"+str(i)+"m_"+pm[j]+"=" for i in range(7) for j in range(2*i+1)] #???
    
    
    # In[453]:
    
    
    #coeff_labels[0] = coeff_labels[0].replace("0p=", "0=")
    
    
    # In[454]:
    
    
    #pcell = "\n".join(["    " + "    ".join(["{:.6f}".format(j) for j in i]) for i in cell]) #cell dimensions
    
    
    # In[455]:
    
    
    #pgeo = "\n".join(["      ".join([i[0]]+["{:.6f}".format(j) for j in i[1:]]) for i in geo]) #atom positions
    
    
    # In[456]:
    
    
    #pcoeffs = "\n".join(["\n".join([coeff_labels[j]+"      {:.14f}".format(i[j]) for j in range(len(i))]) for i in coeffs]) #labeled coeffs. these wrong???
    
    
    # In[457]:
    
    
    #os.chdir('/Users/aboth/Desktop/gordon/mlcpproject/mlcp code/'+directory_name)
    
    
    # In[458]:
    
    
    # modifile(sysname+"_MLCP-cell", "w", pcell) #writes the files
    # modifile(sysname+"_MLCP-geo", "w", pgeo)
    # modifile(sysname+"_MLCP-coeff", "w", pcoeffs)
    
    
    # In[459]:
    
    
    t7 = time.time()
    
    """
    print(round(t2-t1,3))
    print(round(t3-t2,3))
    print(round(t4-t3,3))
    print(round(t5-t4,3))
    print(round(t6-t5,3))
    print(round(t7-t6,3))
    print(round(t7-t1,3))
    """
    
    
    # In[460]:
    
    
    # f = open(sysname+"_MLCP-data","w") #the data file
    
    
    # In[461]:
    
    
    # f.write("Based on: "+cif+"\nRan at Scale Factor: "+str(sf)+"\n") #adds the data/param info
    
    
    # In[462]:
    
    
    #print("\n")
    
    
    # In[463]:
    
    
    # for i in summary: #more of the data things
    #     #print("   ",i)
    #     f.write(i+"\n")
    
    
    # In[464]:
    
    
    # for i in qp_lines: #adding quadrupole report
    #     f.write(i)
    
    
    # In[465]:
    
    
    # for i in data: #pressure data
    #     #print("   ", i[0], ": ", i[1])
    #     f.write(i[0]+": "+i[1]+"\n")
    
    
    # In[ ]:
    
    
    
    
    
    # In[466]:
    
    
    #print("    Total time:", t7-t1, "sec.")
    # f.write("Total time: "+str(round(t7-t1, 2))+" sec.\n") #time report
    
    
    # In[467]:
    
    
    # f.close()
    #print("\n")
    
    
    # In[468]:
    
    
    # os.chdir('/Users/aboth/Desktop/gordon/mlcpproject/mlcp code/')
    
    
    # In[469]:
    
    
    bptable_lines = []
    
    
    # In[470]:
    
    
    # for i in range(len(bptable)):
    #     bptable_lines.append(bptable[i][0]+" ") #the first atom
    #     for j in range(len(bptable[i][1])):
    #         bptable_lines[i] += '{0:.6f} '.format(bptable[i][1][j]) #the coords
    #     bptable_lines[i] += "to " + bptable[i][2] + " " #to the second atom 
    #     for j in range(len(bptable[i][3])):
    #         bptable_lines[i] += '{0:.6f} '.format(bptable[i][3][j]) #at its coords
    #     bptable_lines[i] += "dist = {0:.6f}".format(np.linalg.norm(bptable[i][1]-bptable[i][3])) #distance
    #     bptable_lines[i] += " pressure = {0:.6f} voxels: {1:.6f} volume of contact: {2:.6f}\n".format(bptable[i][4], bptable[i][5], bptable[i][6])
    # pressure, voxels, volume, added last 2 labels
    
    
    # In[471]:
    
    
    # os.chdir('/Users/aboth/Desktop/gordon/mlcpproject/mlcp code/'+directory_name)
    
    
    # In[472]:
    
    
    # modifile(sysname+"_MLCP-bptable","w",bptable_lines) #writing the file
    
    
    # In[473]:
    
    
    # os.chdir('/Users/aboth/Desktop/gordon/mlcpproject/mlcp code/')
    
    
    # In[474]:
    qps=[]
    for i2 in range(len(sitenames)): #for each atom
        qps.append([0,0,0,0,0])
        for j in range(5):
            for k in range(j**2,(j+1)**2): #an increasing odd number for each, as in the harmonics
                qps[i2][j] += coeffs[i2][k]**2 #combining each level squared, for l=0 to 4 for each atom
    qps0=[]
    for i3 in range(len(qps)):
        qpssum=sum(qps[i3])
        try:
            qps0.append(qps[i3][0]/qpssum)
        except:
            qps0.append(0)
    
    compoundname=sysname.split('_')[0]
    result={}
    result['sysname']=compoundname #compound name
    result['directory']=directory_name
    result['name']=structure.composition.reduced_formula #reduced formula
    result['elements']=stoich #the element identifier of each
    result['counts']=s_tot #cumulutaive total of each element, with some factor
    result['abc']=structure.lattice.abc #abc lattice lengths
    result['cell']=cell #cell matrix
    result['e_data']=e_data #the structure-wide data inputs
    result['totalCP']=float(data[-1][1][:-4]) #total CP (GPa)
    result['sites']=sitenames #atom at each site
    result['geo']=geo #coords of each atom
    result['CP']=cpvalues # predictions at l=0, the number we've been working with (GPa)
    result['Xs']=qps0 #l=0 contribution
    result['neighbors']=neighbors #each site's neighbors
    result['contacts']=[row[0:2] for row in unique] #unique contacts
    result['rfrpredictions']=predictions_rfr #for each in unique contacts
    result['coeffs']=coeffs #coefficient outputs
    result['allvnn']=allvnn #info on all voronoi cells
    result['bptable']=bptable #values here: first atom/coords, second, ml-cp, #points nearby, (half of) the volume of the contact taken up
    result['bptablekey']=bptableguide #indices of the first atom, and the number of the associated contact
    
    return result
    
    
    
    # In[ ]:
    
    
    
    
    
    # In[475]:
    
    
    
    
    # In[ ]:
    
    
    
    
