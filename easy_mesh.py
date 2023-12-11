import os
import numpy as np
import pandas as pd
import vtk
import vedo
import time

# easy mesh is a simple library for mesh processing based on vedo and vtk
# it is designed to be easy to use and easy to understand
# current vedo version: 2023.4.6

def meshsegnet_feature_process(target_mesh, check_manifold=False, need_decimate=False, target_ncells=10000, feature_options=['cells', 'barycenters', 'cell_normals']):
    '''
    input:
        target_mesh: a vedo mesh object
        check_manifold: boolean; whether to check if the mesh is manifold
        need_decimate: boolean; whether to decimate the mesh to target_ncells
        target_ncells: int; target number of cells after decimation
        feature_options: ['cells', 'barycenters', 'cell_normals'] for MeshSegNet; either 'points' or 'cells' musts be the first option
                         current options: 'points' (NPx3), 'cells' (NCx9), 'barycenters' (NCx3), 'cell_normals' (NCx3), 'cell_curvatures' (NCx1), 'cell_densities (NCx3)'
    output: a numpy array for the input features of segmentation model (e.g., MeshSegNet) in either training or inference phases based on the feature_scheme
    '''
    # move mesh to origin
    mesh = target_mesh.clone()

    if check_manifold == True:
        if mesh.is_manifold() == False:
            print('trying to repair...')
            mesh.clean()
            mesh.non_manifold_faces(tol=0)
            if mesh.is_manifold():
                print('Repair completed')
            else:
                print('Repair failed')
        
    if need_decimate == True: 
        target_num = target_ncells
        ratio = target_num/mesh.ncells # calculate ratio
        mesh = mesh.decimate(fraction=ratio)

    # move points into origin
    # mesh.points(mesh.points() - mean_cell_centers[0:3])
    mean_cell_centers = mesh.center_of_mass()
    points = mesh.points() - mean_cell_centers[0:3]

    if 'points' in feature_options:
        points -= points.mean(axis=0)
        points /= points.std(axis=0)

        if 'point_normals' in feature_options:
            mesh.compute_normals(points=True, cells=False)
            point_normals = mesh.pointdata['Normals']
            point_normals -= point_normals.mean(axis=0)
            point_normals /= point_normals.std(axis=0)
        
        if 'point_curvatures' in feature_options:
            mesh.compute_curvature(method=1)
            point_curvatures = mesh.pointdata['Mean_Curvature']
            point_curvatures -= point_curvatures.mean(axis=0)
            point_curvatures /= point_curvatures.std(axis=0)

        if 'point_densities' in feature_options:
            # features used in https://ieeexplore.ieee.org/abstract/document/10063862
            barycenters = mesh.cell_centers()
            
            # make barycenters within [0, 1]
            unity_points = points.copy()
            unity_points -= unity_points.min(axis=0)
            unity_points /= (unity_points.max(axis=0) - unity_points.min(axis=0))

            from scipy.spatial import distance_matrix
            PM1 = np.zeros([mesh.npoints, mesh.npoints], dtype=np.float32)
            PM2 = np.zeros([mesh.npoints, mesh.npoints], dtype=np.float32)
            PM3 = np.zeros([mesh.npoints, mesh.npoints], dtype=np.float32)
            D = distance_matrix(unity_points, unity_points) # distance matrix
            PM1[D<0.05] = 1
            PM2[D<0.1] = 1
            PM3[D<0.2] = 1
            pm1 = np.sum(PM1, axis=1)
            pm2 = np.sum(PM2, axis=1)
            pm3 = np.sum(PM3, axis=1)
            point_densities = np.concatenate([pm1.reshape(-1, 1), pm2.reshape(-1, 1), pm3.reshape(-1, 1)], axis=1)
            point_densities -= point_densities.min(axis=0)
            point_densities /= (point_densities.max(axis=0) - point_densities.min(axis=0))

    if 'cells' in feature_options:
        ids = vedo.vtk2numpy(mesh.polydata().GetPolys().GetData()).reshape((mesh.ncells, -1))[:, 1:]
        cells = points[ids].reshape(mesh.ncells, 9).astype(dtype='float32')
        cells[:, 0:3] -= points.mean(axis=0) # point 1
        cells[:, 0:3] /= points.std(axis=0)
        cells[:, 3:6] -= points.mean(axis=0) # point 2
        cells[:, 3:6] /= points.std(axis=0)
        cells[:, 6:9] -= points.mean(axis=0) # point 3
        cells[:, 6:9] /= points.std(axis=0)

        if 'cell_normals' in feature_options:
            mesh.compute_normals(points=False, cells=True)
            cell_normals = mesh.celldata['Normals']
            cell_normals -= cell_normals.mean(axis=0)
            cell_normals /= cell_normals.std(axis=0)

        if 'barycenters' in feature_options:
            barycenters = mesh.cell_centers()
            # make barycenters within [0, 1]
            barycenters[:, 0:3] -= barycenters[:, 0:3].min(axis=0)
            barycenters[:, 0:3] /= (barycenters[:, 0:3].max(axis=0) - barycenters[:, 0:3].min(axis=0))

        if 'cell_curvatures' in feature_options:
            # mesh.compute_curvature(method=0)
            # mesh.map_points_to_cells(arrays=(['Gauss_Curvature']), move=True)
            # cell_curvatures = mesh.celldata['Gauss_Curvature']
            mesh.compute_curvature(method=1)
            mesh.map_points_to_cells(arrays=(['Mean_Curvature']), move=True)
            cell_curvatures = mesh.celldata['Mean_Curvature']
            cell_curvatures -= cell_curvatures.mean(axis=0)
            cell_curvatures /= cell_curvatures.std(axis=0)

        if 'cell_densities' in feature_options:
            # features used in https://ieeexplore.ieee.org/abstract/document/10063862
            barycenters = mesh.cell_centers()
            
            # make barycenters within [0, 1]
            barycenters[:, 0:3] -= barycenters[:, 0:3].min(axis=0)
            barycenters[:, 0:3] /= (barycenters[:, 0:3].max(axis=0) - barycenters[:, 0:3].min(axis=0))

            from scipy.spatial import distance_matrix
            M1 = np.zeros([mesh.ncells, mesh.ncells], dtype=np.float32)
            M2 = np.zeros([mesh.ncells, mesh.ncells], dtype=np.float32)
            M3 = np.zeros([mesh.ncells, mesh.ncells], dtype=np.float32)
            D = distance_matrix(barycenters, barycenters) # distance matrix
            M1[D<0.05] = 1
            M2[D<0.1] = 1
            M3[D<0.2] = 1
            m1 = np.sum(M1, axis=1)
            m2 = np.sum(M2, axis=1)
            m3 = np.sum(M3, axis=1)
            cell_densities = np.concatenate([m1.reshape(-1, 1), m2.reshape(-1, 1), m3.reshape(-1, 1)], axis=1)
            cell_densities -= cell_densities.min(axis=0)
            cell_densities /= (cell_densities.max(axis=0) - cell_densities.min(axis=0))

    # 'points' or 'cells' is the mandatory feature
    if 'points' in feature_options:
        X = points
        for i_feature in feature_options:
            if i_feature == 'points':
                continue
            elif i_feature == 'point_normals':
                X = np.column_stack((X, point_normals))
                mesh.pointdata['Normalized_Normals'] = point_normals
            elif i_feature == 'point_curvatures':
                X = np.column_stack((X, point_curvatures))
                mesh.pointdata['Normalized_Curvatures'] = point_curvatures
            elif i_feature == 'point_densities':
                X = np.column_stack((X, point_densities))
                mesh.pointdata['Normalized_Densities'] = point_densities

    elif 'cells' in feature_options:
        X = cells
        for i_feature in feature_options:
            if i_feature == 'cells':
                continue
            elif i_feature == 'barycenters':
                X = np.column_stack((X, barycenters))
                mesh.celldata['Normalized_Barycenters'] = barycenters
            elif i_feature == 'cell_normals':
                X = np.column_stack((X, cell_normals))
                mesh.celldata['Normalized_Normals'] = cell_normals
            elif i_feature == 'cell_curvatures':
                X = np.column_stack((X, cell_curvatures))
                mesh.celldata['Normalized_Curvatures'] = cell_curvatures
            elif i_feature == 'cell_densities':
                X = np.column_stack((X, cell_densities))
                mesh.celldata['Normalized_Densities'] = cell_densities

    return X, mesh
    

def mesh_grah_cut_optimization(target_mesh, label_probability, lambda_c=30, label_name='Label'):
    '''
    required packages: pygco; this function has been optimized by vecteroization
    inputs:
        target_mesh: a vedo mesh object
        label_probability: a numpy array of shape (num_cells, num_classes)
        num_classes: int; number of classes
        lambda_c: float; parameter for the edge weight
        label_name: string; name of the label array in the target_mesh
    '''
    from pygco import cut_from_graph

    start_time = time.time()

    label_probability = label_probability.squeeze()
    num_classes = label_probability.shape[1]

    round_factor = 100
    label_probability[label_probability < 1.0e-6] = 1.0e-6

    # unaries
    unaries = -round_factor * np.log10(label_probability)
    unaries = unaries.astype(np.int32)
    unaries = unaries.reshape(-1, num_classes)

    # parawise
    pairwise = 1 - np.eye(num_classes, dtype=np.int32)

    #edges
    try:
        normals = target_mesh.celldata['Normals']
    except:
        target_mesh.compute_normals()
        normals = target_mesh.celldata['Normals']
        
    barycenters = target_mesh.cell_centers()

    # find all edges [vedo build-in function is wrong]
    all_cells = np.array(target_mesh.cells())
    edge1 = np.sort(all_cells[:, [0, 1]])
    edge2 = np.sort(all_cells[:, [0, 2]])
    edge3 = np.sort(all_cells[:, [1, 2]])
    all_unique_edges = np.unique(np.concatenate((edge1, edge2, edge3), axis=0), axis=0)

    # find shared edges
    shared_edges = []
    for i_edge in all_unique_edges:
        p1_id = i_edge[0]
        p2_id = i_edge[1]
        p1_connected_cells_ids = target_mesh.connected_cells(p1_id, return_ids=True)
        p2_connected_cells_ids = target_mesh.connected_cells(p2_id, return_ids=True)
        connected_cells_ids = np.intersect1d(p1_connected_cells_ids, p2_connected_cells_ids)
        if len(connected_cells_ids) == 2:
            shared_edges.append(connected_cells_ids)
    shared_edges = np.array(shared_edges) # share_edge: [cell1_id, cell2_id]

    cos_theta = np.dot(normals[shared_edges[:, 0], 0:3], normals[shared_edges[:, 1], 0:3].transpose()).diagonal()/np.linalg.norm(normals[shared_edges[:, 0], 0:3], axis=1)/np.linalg.norm(normals[shared_edges[:, 1], 0:3], axis=1)
    cos_theta[cos_theta >= 1.0] = 0.9999
    theta = np.arccos(cos_theta)
    phi = np.linalg.norm(barycenters[shared_edges[:, 0], :] - barycenters[shared_edges[:, 1], :], axis=1)
    
    beta = 1 + np.linalg.norm((np.dot(normals[shared_edges[:, 0], 0:3], normals[shared_edges[:, 1], 0:3].transpose()).diagonal()).reshape(-1, 1), axis=1)
    edges = -beta*np.log10(theta/np.pi)*phi
    edges2 = -np.log10(theta/np.pi)*phi
    theta_mask = theta > np.pi/2.0
    edges[theta_mask] = edges2[theta_mask]
    edges = np.concatenate([shared_edges, edges.reshape(-1, 1)], axis=1)
    edges[:, 2] *= lambda_c*round_factor
    edges = edges.astype(np.int32)

    refine_labels = cut_from_graph(edges, unaries, pairwise)
    refine_labels = refine_labels.reshape([-1, 1])

    new_mesh = target_mesh.clone()
    new_mesh.celldata['Label'] = refine_labels
    end_time = time.time()
    total_time = end_time - start_time
    print("---Label has been refined: {} seconds ---".format(total_time))

    return new_mesh


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
    upper_mesh_d = vedo.load('Example_02_d.vtp')
    # upper_mesh = vedo.load('Example_03.vtp')

    # meshsegnet example
    # cell-based
    X, mesh_d = meshsegnet_feature_process(upper_mesh_d, check_manifold=True, need_decimate=True, target_ncells=10000, feature_options=['cells', 'cell_curvatures', 'cell_normals', 'cell_densities', 'barycenters'])
    print(X.shape, mesh_d.ncells)
    mesh_d.write(('tmp_Example_01_d_cell_based.vtp'))
    # point-based
    upper_mesh_d = upper_mesh.decimate(n=10000)
    print('# of cells and points after decimating', upper_mesh_d.ncells, upper_mesh_d.npoints)
    X, mesh_d = meshsegnet_feature_process(upper_mesh_d, check_manifold=True, need_decimate=False, target_ncells=10000, feature_options=['points', 'point_normals', 'point_curvatures', 'point_densities'])
    print(X.shape, mesh_d.npoints)
    mesh_d.write(('tmp_Example_02_d_point_based.vtp'))

    # X, mesh_d = meshsegnet_feature_process(upper_mesh, check_manifold=True, need_decimate=True, target_ncells=10000, feature_options=['cells', 'cell_normals'])
    # print(X.shape)
    # X, mesh_d = meshsegnet_feature_process(upper_mesh, check_manifold=True, need_decimate=True, target_ncells=10000, feature_options=['cells'])
    # print(X.shape)

    # graph-cut example
    # label_probability = np.zeros([upper_mesh_d.ncells, 17])
    # # init probability; 0.9 for each cell with the current label
    # for i_label in range(17):
    #     label_probability[upper_mesh_d.celldata['Label']==i_label, i_label] = 0.9
    # refined_label_mesh = mesh_grah_cut_optimization(upper_mesh_d, label_probability, lambda_c = 30, label_name='Label')
    # upper_mesh_d.show().close()
    # refined_label_mesh.show().close()
    
    # expand_selection example
    # teeth_mesh = upper_mesh.clone().threshold('Label', above=0.5, below=16.5, on='cells').c('red')
    # expanded_teeth_mesh = expand_selection(upper_mesh, teeth_mesh, n_loop=5).c('blue').alpha(0.5)
    # vedo.show(teeth_mesh, expanded_teeth_mesh).close()

    # density feature
    # cell_centers = mesh_d.cell_centers()
    # cell_centers[:, 0:3] -= cell_centers[:, 0:3].min(axis=0)
    # cell_centers[:, 0:3] /= (cell_centers[:, 0:3].max(axis=0) - cell_centers[:, 0:3].min(axis=0))
    # from scipy.spatial import distance_matrix
    # M1 = np.zeros([mesh_d.ncells, mesh_d.ncells])
    # M2 = np.zeros([mesh_d.ncells, mesh_d.ncells])
    # M3 = np.zeros([mesh_d.ncells, mesh_d.ncells])
    # D = distance_matrix(cell_centers, cell_centers)
    # # print(D)
    # M1[D<0.05] = 1
    # M2[D<0.1] = 1
    # M3[D<0.2] = 1
    # m1 = np.sum(M1, axis=1)
    # m2 = np.sum(M2, axis=1)
    # m3 = np.sum(M3, axis=1)
    # m = np.concatenate([m1.reshape(-1, 1), m2.reshape(-1, 1), m3.reshape(-1, 1)], axis=1)
    # print(m)
    # m -= m.min(axis=0)
    # m /= (m.max(axis=0) - m.min(axis=0))
    # print(m)
    # print(m.shape)
    # mesh_d.celldata['Density'] = m
    # mesh_d.celldata['Density_mag'] = np.linalg.norm(m, axis=1)
    # mesh_d.write(('tmp_Example_03_d.vtp'))


    