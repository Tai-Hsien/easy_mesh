import os
import numpy as np
import pandas as pd
import vtk
import vedo
import time

# easy mesh is a simple library for mesh processing based on vedo and vtk
# it is designed to be easy to use and easy to understand
# current vedo version: 2023.4.6

def meshsegnet_feature_process(target_mesh, need_decimate=False, target_ncells=10000):
    '''
    input: a vedo mesh object
    output: a Nx15 numpy array for the input of MeshSegNet/iMeshSegNet in either training or inference phases,
            where 15 features are 9 points of a triangle, 3 barycenters, 3 normals
    '''
    # move mesh to origin
    mesh = target_mesh.clone()

    if mesh.is_manifold() == False:
        print('trying to repair...')
        mesh.clean()
        mesh.non_manifold_faces(tol=0)
        if mesh.is_manifold():
            print('Repair completed')
        else:
            print('Repair failed')
            return None
        
    if need_decimate == True: 
        target_num = target_ncells
        ratio = target_num/mesh.ncells # calculate ratio
        mesh = mesh.decimate(fraction=ratio)

    points = mesh.points()
    mean_cell_centers = mesh.center_of_mass()
    points[:, 0:3] -= mean_cell_centers[0:3]

    ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((mesh.ncells, -1))[:, 1:]
    cells = points[ids].reshape(mesh.ncells, 9).astype(dtype='float32')
    mesh.compute_normals()
    normals = mesh.celldata['Normals']
    barycenters = mesh.cell_centers()
    barycenters -= mean_cell_centers[0:3]

    #normalized data
    maxs = points.max(axis=0)
    mins = points.min(axis=0)
    means = points.mean(axis=0)
    stds = points.std(axis=0)
    nmeans = normals.mean(axis=0)
    nstds = normals.std(axis=0)

    for i in range(3): # x, y, z direction
        cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
        cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
        cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i]-mins[i])
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

    return np.column_stack((cells, barycenters, normals)), mesh




def expand_selection(main_mesh, partial_mesh, n_loop=1):
    '''
    input:
        main_mesh: a vedo mesh object
        partial_mesh: a vedo mesh object
        n_loop: number of loops to expand the selection; each loop will expand the selection by one connection
    output: a list of selected vertex indices
    '''
    for i_loop in range(n_loop):
        start_time = time.time()

        # find the cell ids of partial_mesh on the main mesh
        partial_cell_centers = partial_mesh.cell_centers()
        main_cell_centers = main_mesh.cell_centers()
        main_cell_centers = vedo.Points(main_cell_centers)

        original_cell_ids_of_partial_mesh = []
        for i in range(len(partial_cell_centers)):
            i_original_selected_cell_ids = main_cell_centers.closest_point(partial_cell_centers[i], n=1, return_point_id=True)
            original_cell_ids_of_partial_mesh.append(i_original_selected_cell_ids)
        original_cell_ids_of_partial_mesh = np.array(original_cell_ids_of_partial_mesh, dtype=np.int32).squeeze()

        # find the boundary points of partial mesh and their ids on original mesh
        boundary_pt_ids = partial_mesh.boundaries(return_point_ids=True)
        boundary_pts = partial_mesh.points()[boundary_pt_ids]

        original_selected_boudnary_pt_ids = []
        for i in range(len(boundary_pts)):
            i_original_selected_boudnary_pt_ids = main_mesh.closest_point(boundary_pts[i], n=1, return_point_id=True)
            original_selected_boudnary_pt_ids.append(i_original_selected_boudnary_pt_ids)
        original_selected_boudnary_pt_ids = np.array(original_selected_boudnary_pt_ids, dtype=np.int32).squeeze()

        # create a tmp cell array ['selection'] to store the selection
        main_mesh.celldata['tmp_selection'] = np.zeros(main_mesh.ncells)
        main_mesh.celldata['tmp_selection'][original_cell_ids_of_partial_mesh] = 10
        main_mesh.celldata.select('tmp_selection')

        # expand selection
        expand_selection = []
        for i in range(len(original_selected_boudnary_pt_ids)):
            # i_original_selected_boudnary_pt_ids = mesh.closest_point(boundary_pts[i], n=1, return_cell_id=True)
            connected_cells = main_mesh.connected_cells(original_selected_boudnary_pt_ids[i], return_ids=True)
            for j in connected_cells:
                expand_selection.append(j)
        expand_selection = np.array(expand_selection, dtype=np.int32).squeeze()

        main_mesh.celldata['tmp_selection'][expand_selection] = 10

        partial_mesh = main_mesh.clone().threshold('tmp_selection', above=9.5, below=10.5, on='cells')

        end_time = time.time()
        total_time = end_time - start_time
        print("---Loop {}: {} seconds ---".format(i_loop, total_time))

    # vedo_bd_pts = vedo.Points(boundary_pts)
    # vedo.show(main_mesh, vedo_bd_pts)
    main_mesh.celldata.remove('tmp_selection')

    return partial_mesh


if __name__ == '__main__':

    upper_mesh = vedo.load('Example_01.vtp')
    
    # expand_selection example
    # teeth_mesh = upper_mesh.clone().threshold('Label', above=0.5, below=16.5, on='cells').c('red')
    # expanded_teeth_mesh = expand_selection(upper_mesh, teeth_mesh, n_loop=5).c('blue').alpha(0.5)
    # vedo.show(teeth_mesh, expanded_teeth_mesh).close()

    # meshsegnet example
    X, mesh_d = meshsegnet_feature_process(upper_mesh, need_decimate=True, target_ncells=10000)
    print(X.shape)