
def plot_cases(pcs, backend='open3d', size=8, colours=None):
    """
    General purpose function for plotting point clouds
    
    Args:
        pcs (list): list of point clouds (N x 3)
        backend (str): currently either open3d or matplotlib
        size (int): point size
        colours (list): either a list of rgb values for each point (N x 3),
                        a single vector of rgb values for uniform colour (1 x 3),
                        or None for random colours
    """

    def pcd_with_colours(case, colours=None): # uses open3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(c)
        if colours is None: pc.paint_uniform_color(np.random.rand(3))
        elif colours.ndim == 1: pc.paint_uniform_color(colours.flatten())
        elif colours is not None: pc.colors = o3d.utility.Vector3dVector(colours)
        else: raise ValueError('Argument \'colours\' could not be understood. ' +
                               'Either input 1D arrays for uniform colours, '+
                               'or 2D arrays to colour individual points.')
        return pc

    if backend == 'open3d':
        import open3d as o3d
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = size
        for i, c in enumerate(pcs):
            if colours is not None:
                pcd = pcd_with_colours(c, colours[i%len(colours)])
            else: pcd = pcd_with_colours(c)
            vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()
    elif backend == 'matplotlib':
        import matplotlib.pyplot as pyplot
        from mpl_toolkits.mplot3d import Axes3D
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        for i, c in enumerate(pcs):
            if colours is None:
                ax.scatter(c[:,0], c[:,1], c[:,2], s=size, c=[np.random.rand(3)])
            else:
                ax.scatter(c[:,0], c[:,1], c[:,2], s=size, c=colours[i%len(colours)])
        return ax
    else:
        raise NotImplementedError('Unknown backend.')
